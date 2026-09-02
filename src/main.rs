//! voice-assistant: wake word -> VAD recording -> Whisper ASR -> kiro-cli.
//!
//! Usage:
//!   voice-assistant              run the full pipeline (first run enters setup)
//!   voice-assistant setup        (re)configure wake word / language / model
//!   voice-assistant devices      list audio input devices
//!   voice-assistant test-wake    print wake word scores
//!   voice-assistant test-vad     print voice activity probabilities
//!   voice-assistant test-asr     record one utterance and print transcription
//!
//! Settings persist in ~/.voice-assistant/config; models auto-download to
//! ~/.voice-assistant/models on first use.
//!
//! Env var overrides (take precedence over the config file):
//!   VA_MODELS_DIR      model directory
//!   VA_WAKE_THRESHOLD  detection threshold        (default: 0.5)
//!   VA_WHISPER_MODEL   explicit whisper ggml model path
//!   VA_LANG            ASR language: auto/zh/en/...
//!   VA_ASR_PROMPT      whisper initial prompt
//!   VA_KIRO_ARGS       extra kiro-cli args, space separated (e.g. "--agent foo")
//!   VA_HF_BASE         HuggingFace endpoint (default: https://hf-mirror.com)

mod agent;
mod asr;
mod audio;
mod setup;
mod vad;
mod wakeword;

use anyhow::Result;
use crossbeam_channel::Receiver;
use std::collections::VecDeque;
use std::io::IsTerminal;
use std::time::Duration;
use vad::VAD_CHUNK;

struct Config {
    wake_model: String,
    wake_display: String,
    wake_threshold: f32,
    whisper_model: String,
    lang: String,
    asr_prompt: Option<String>,
    kiro_args: Vec<String>,
}

impl Config {
    /// Load settings (running first-time setup if needed), download any
    /// missing models, and apply env var overrides.
    fn load() -> Result<Self> {
        let settings = match setup::load() {
            Some(s) => s,
            None if std::io::stdin().is_terminal() => setup::interactive_setup(None)?,
            None => setup::Settings::default(),
        };
        setup::ensure_models(&settings)?;

        let env = |k: &str, d: &str| std::env::var(k).unwrap_or_else(|_| d.to_string());
        let lang = env("VA_LANG", &settings.lang);
        // Bias whisper towards Simplified Chinese when transcribing zh,
        // otherwise it frequently emits Traditional characters.
        let default_prompt = if lang.starts_with("zh") {
            "以下是简体中文普通话的句子。"
        } else {
            ""
        };
        let asr_prompt = match env("VA_ASR_PROMPT", default_prompt) {
            s if s.is_empty() => None,
            s => Some(s),
        };
        let models_dir = setup::models_dir();
        Ok(Self {
            wake_model: setup::wake_model_path(&settings).to_string_lossy().into_owned(),
            wake_display: setup::wake_display(&settings),
            wake_threshold: env("VA_WAKE_THRESHOLD", &settings.threshold.to_string())
                .parse()
                .unwrap_or(0.5),
            whisper_model: env(
                "VA_WHISPER_MODEL",
                &models_dir
                    .join(format!("ggml-{}.bin", settings.whisper))
                    .to_string_lossy(),
            ),
            lang,
            asr_prompt,
            kiro_args: env("VA_KIRO_ARGS", &settings.kiro_args)
                .split_whitespace()
                .map(String::from)
                .collect(),
        })
    }

    fn wakeword(&self) -> Result<wakeword::WakeWord> {
        let dir = setup::models_dir();
        wakeword::WakeWord::new(
            &dir.join("melspectrogram.onnx").to_string_lossy(),
            &dir.join("embedding_model.onnx").to_string_lossy(),
            &self.wake_model,
        )
    }

    fn vad(&self) -> Result<vad::Vad> {
        vad::Vad::new(&setup::models_dir().join("silero_vad.onnx").to_string_lossy())
    }

    fn asr(&self) -> Result<asr::Asr> {
        eprintln!("[asr] loading whisper model {} ...", self.whisper_model);
        asr::Asr::new(&self.whisper_model, &self.lang, self.asr_prompt.clone())
    }
}

fn main() -> Result<()> {
    // Route whisper.cpp/ggml C-side logging into the `log` crate;
    // with no logger registered this silences the console spam.
    whisper_rs::install_logging_hooks();
    match std::env::args().nth(1).as_deref() {
        Some("devices") => return audio::list_devices(),
        Some("setup") => {
            let s = setup::interactive_setup(setup::load())?;
            setup::ensure_models(&s)?;
            return Ok(());
        }
        _ => {}
    }
    let cfg = Config::load()?;
    match std::env::args().nth(1).as_deref() {
        Some("selftest") => selftest(&cfg),
        Some("test-wake") => test_wake(&cfg),
        Some("test-vad") => test_vad(&cfg),
        Some("test-asr") => test_asr(&cfg),
        Some(other) => {
            eprintln!("unknown command: {other}");
            std::process::exit(2);
        }
        None => run(&cfg),
    }
}

fn start_capture() -> Result<(audio::Capture, Receiver<Vec<i16>>)> {
    let (tx, rx) = crossbeam_channel::bounded(256);
    let cap = audio::Capture::start(tx)?;
    Ok((cap, rx))
}

// ---------------- full pipeline ----------------

fn run(cfg: &Config) -> Result<()> {
    let mut wake = cfg.wakeword()?;
    let mut vad = cfg.vad()?;
    let asr = cfg.asr()?;
    let (_cap, rx) = start_capture()?;

    println!(
        "== voice assistant ready, say the wake word (\"{}\") ==",
        cfg.wake_display
    );
    let mut preroll: VecDeque<i16> = VecDeque::new(); // ~0.5s of pre-wake audio

    loop {
        let chunk = rx.recv()?;
        preroll.extend(chunk.iter().copied());
        while preroll.len() > 8000 {
            preroll.pop_front();
        }
        let Some(score) = wake.feed(&chunk)? else {
            continue;
        };
        if score < cfg.wake_threshold {
            continue;
        }
        println!("\x07>> wake word detected (score {score:.2}), listening...");
        vad.reset();
        let audio = record_utterance(&rx, &mut vad, Vec::new())?;
        println!(">> transcribing {:.1}s of audio...", audio.len() as f32 / 16000.0);
        match asr.transcribe(&audio) {
            Ok(text) if text.is_empty() => println!(">> heard nothing, back to listening"),
            Ok(text) => {
                println!(">> you said: {text}");
                println!(">> asking kiro-cli...\n");
                agent::ask_kiro(&text, &cfg.kiro_args)?;
                println!();
            }
            Err(e) => eprintln!(">> transcription failed: {e}"),
        }
        // drop anything captured while ASR/agent were running, then re-arm
        while rx.try_recv().is_ok() {}
        wake.reset();
        preroll.clear();
        println!("== listening for wake word ==");
    }
}

/// Record until the speaker stops talking (VAD-based endpointing).
fn record_utterance(
    rx: &Receiver<Vec<i16>>,
    vad: &mut vad::Vad,
    preroll: Vec<i16>,
) -> Result<Vec<i16>> {
    const CHUNK_MS: f32 = VAD_CHUNK as f32 * 1000.0 / 16000.0; // 32 ms
    let mut audio = preroll;
    let mut pending: Vec<i16> = Vec::new();
    let mut speech_started = false;
    let mut silence_ms = 0f32;
    let mut total_ms = 0f32;

    loop {
        let chunk = rx.recv_timeout(Duration::from_secs(2))?;
        audio.extend_from_slice(&chunk);
        pending.extend_from_slice(&chunk);
        while pending.len() >= VAD_CHUNK {
            let frame: Vec<i16> = pending.drain(..VAD_CHUNK).collect();
            let p = vad.predict(&frame)?;
            if p > 0.5 {
                speech_started = true;
                silence_ms = 0.0;
            } else {
                silence_ms += CHUNK_MS;
            }
            total_ms += CHUNK_MS;
        }
        if speech_started && silence_ms >= 1000.0 {
            break; // said something, then 1s of silence -> done
        }
        if !speech_started && total_ms >= 6000.0 {
            break; // never spoke -> give up
        }
        if total_ms >= 30_000.0 {
            break; // hard cap
        }
    }
    Ok(audio)
}

// ---------------- component tests ----------------

/// Headless sanity check: run every component on synthetic audio.
fn selftest(cfg: &Config) -> Result<()> {
    // 1. wake word pipeline on 3s of low-level noise -> expect a low score
    let mut wake = cfg.wakeword()?;
    let noise: Vec<i16> = (0..48000)
        .map(|i| (((i * 7919) % 997) as i16 - 498) / 4)
        .collect();
    let score = wake.feed(&noise)?;
    match score {
        Some(s) if s < 0.5 => println!("[ok] wakeword: noise score {s:.4} (< 0.5)"),
        Some(s) => println!("[??] wakeword: noise score {s:.4} unexpectedly high"),
        None => anyhow::bail!("wakeword produced no score on 3s of audio"),
    }

    // 2. VAD: silence -> low prob; loud tone -> model runs and returns a prob
    let mut vad = cfg.vad()?;
    let p_silence = vad.predict(&[0i16; VAD_CHUNK])?;
    anyhow::ensure!(p_silence < 0.3, "VAD silence prob too high: {p_silence}");
    println!("[ok] vad: silence prob {p_silence:.4} (< 0.3)");

    // 3. ASR: transcribe 1s of silence -> should not crash
    let asr = cfg.asr()?;
    let text = asr.transcribe(&vec![0i16; 16000])?;
    println!("[ok] asr: silence -> {:?}", text);

    // 4. kiro-cli present on PATH
    let out = std::process::Command::new("kiro-cli").arg("--version").output();
    match out {
        Ok(o) if o.status.success() => println!(
            "[ok] kiro-cli: {}",
            String::from_utf8_lossy(&o.stdout).trim()
        ),
        _ => println!("[!!] kiro-cli not found on PATH"),
    }

    println!("selftest done");
    Ok(())
}

fn test_wake(cfg: &Config) -> Result<()> {
    let mut wake = cfg.wakeword()?;
    let (_cap, rx) = start_capture()?;
    println!("say the wake word; scores > threshold are marked. Ctrl-C to quit.");
    loop {
        let chunk = rx.recv()?;
        if let Some(score) = wake.feed(&chunk)? {
            if score > 0.05 {
                let mark = if score >= cfg.wake_threshold { "  <== DETECTED" } else { "" };
                println!("score: {score:.3}{mark}");
            }
        }
    }
}

fn test_vad(cfg: &Config) -> Result<()> {
    let mut vad = cfg.vad()?;
    let (_cap, rx) = start_capture()?;
    let mut pending: Vec<i16> = Vec::new();
    println!("speak; showing VAD probability per 32ms frame. Ctrl-C to quit.");
    loop {
        let chunk = rx.recv()?;
        pending.extend_from_slice(&chunk);
        while pending.len() >= VAD_CHUNK {
            let frame: Vec<i16> = pending.drain(..VAD_CHUNK).collect();
            let p = vad.predict(&frame)?;
            let bar = "#".repeat((p * 40.0) as usize);
            println!("{p:.2} {bar}");
        }
    }
}

fn test_asr(cfg: &Config) -> Result<()> {
    let mut vad = cfg.vad()?;
    let asr = cfg.asr()?;
    let (_cap, rx) = start_capture()?;
    println!("speak now (recording until 1s of silence)...");
    let audio = record_utterance(&rx, &mut vad, Vec::new())?;
    println!("recorded {:.1}s, transcribing...", audio.len() as f32 / 16000.0);
    let text = asr.transcribe(&audio)?;
    println!("transcription: {text}");
    Ok(())
}
