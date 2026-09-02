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
//!   VA_AGENT_CMD       ACP agent launch command (default: "kiro-cli acp --agent voice")
//!   VA_HF_BASE         HuggingFace endpoint (default: https://hf-mirror.com)

mod acp;
mod agent;
mod asr;
mod audio;
mod setup;
mod vad;
mod wakeword;

use agent::{AgentHandle, AgentState};
use anyhow::Result;
use crossbeam_channel::Receiver;
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
    /// Full argv used to launch the ACP agent process (kept alive across turns).
    agent_cmd: Vec<String>,
    /// Auto-approve tool-permission requests (maps from the "full" trust mode).
    auto_approve: bool,
    /// kiro-cli permission mode (readonly/safe/full); used to refresh voice.json.
    agent_mode: String,
    /// Assistant persona name derived from the wake word (e.g. "Jarvis").
    persona: String,
    silence_ms: f32,
    no_speech_ms: f32,
    max_utterance_ms: f32,
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
            agent_cmd: env("VA_AGENT_CMD", &settings.agent_cmd)
                .split_whitespace()
                .map(String::from)
                .collect(),
            auto_approve: settings.agent_mode == "full",
            agent_mode: settings.agent_mode.clone(),
            persona: setup::persona_name(&settings.wake_word),
            silence_ms: env("VA_SILENCE_MS", &settings.silence_ms.to_string())
                .parse()
                .unwrap_or(1000.0),
            no_speech_ms: env("VA_NO_SPEECH_MS", &settings.no_speech_ms.to_string())
                .parse()
                .unwrap_or(6000.0),
            max_utterance_ms: env("VA_MAX_UTTERANCE_MS", &settings.max_utterance_ms.to_string())
                .parse()
                .unwrap_or(30000.0),
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
        Some("vad-wav") => vad_wav(&cfg),
        Some("test-wake") => test_wake(&cfg),
        Some("test-vad") => test_vad(&cfg),
        Some("test-asr") => test_asr(&cfg),
        Some("ask") => ask_cli(&cfg),
        Some("agent-test") => agent_test(&cfg),
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

/// For the kiro backend, regenerate ~/.kiro/agents/voice.json so the agent's
/// identity always matches the current wake word (the wake word is its name).
/// No-op for custom ACP backends, which manage their own persona.
fn sync_persona(cfg: &Config) {
    if cfg.agent_cmd.first().map(|s| s == "kiro-cli").unwrap_or(false) {
        if let Err(e) = setup::write_agent_config(&cfg.agent_mode, &cfg.persona) {
            eprintln!("[setup] warning: could not refresh voice.json: {e}");
        }
    }
}

/// Keywords that mean "interrupt the current task" rather than "new request".
fn is_cancel_intent(text: &str) -> bool {
    const WORDS: &[&str] = &["停", "取消", "别说了", "停下", "闭嘴", "打断", "stop", "cancel"];
    let t = text.to_lowercase();
    WORDS.iter().any(|w| t.contains(w))
}

/// Block until the agent finishes the current turn (or dies), printing restarts.
fn wait_for_idle(agent: &AgentHandle) {
    for st in agent.state_rx.iter() {
        match st {
            AgentState::Idle(_) => return,
            AgentState::Restarting(r) => eprintln!(">> agent 重启中: {r}"),
            _ => {}
        }
    }
}

/// Headless ACP check (no mic): spawn the supervised agent and send each
/// argument as a prompt in the SAME session, proving session continuity.
///   voice-assistant ask "remember 42" "what number did I say?"
fn ask_cli(cfg: &Config) -> Result<()> {
    let prompts: Vec<String> = std::env::args().skip(2).collect();
    anyhow::ensure!(!prompts.is_empty(), "usage: voice-assistant ask <text> [more text...]");
    sync_persona(cfg);
    eprintln!("[acp] starting agent: {}", cfg.agent_cmd.join(" "));
    let agent = AgentHandle::spawn(cfg.agent_cmd.clone(), cfg.auto_approve);
    for (i, p) in prompts.iter().enumerate() {
        println!("\n>> [{}] {p}", i + 1);
        agent.prompt(p.clone());
        wait_for_idle(&agent);
    }
    agent.shutdown();
    Ok(())
}

/// Hidden headless test of the supervisor: cancel an in-flight turn, then
/// continue on the same session (proves redirect + session survival). With a
/// bogus VA_AGENT_CMD it instead exercises the restart/backoff failsafe.
fn agent_test(cfg: &Config) -> Result<()> {
    sync_persona(cfg);
    eprintln!("[agent-test] launching: {}", cfg.agent_cmd.join(" "));
    let agent = AgentHandle::spawn(cfg.agent_cmd.clone(), cfg.auto_approve);

    // Consume states until the next Idle, printing each transition. Returns the
    // stopReason, or None if the agent channel closed.
    let drain_to_idle = |label: &str| -> Option<String> {
        for st in agent.state_rx.iter() {
            println!(">> [{label}] state: {st:?}");
            if let AgentState::Idle(reason) = st {
                return Some(reason);
            }
        }
        None
    };

    println!("\n>> [1] long task, will cancel after 1.2s");
    agent.prompt("从1数到100，每个数字占一行，并加一句简短说明。".into());
    std::thread::sleep(Duration::from_millis(1200));
    println!("\n>> sending cancel");
    agent.cancel();
    let r1 = drain_to_idle("cancel");
    println!(">> after cancel, stopReason = {r1:?} (expect \"cancelled\")");

    println!("\n>> [2] continue on same session");
    agent.prompt("刚才数到几了？一句话回答。".into());
    let r2 = drain_to_idle("continue");
    println!(">> after continue, stopReason = {r2:?} (expect \"end_turn\")");

    agent.shutdown();
    Ok(())
}

// ---------------- full pipeline ----------------

fn run(cfg: &Config) -> Result<()> {
    let mut wake = cfg.wakeword()?;
    let mut vad = cfg.vad()?;
    let asr = cfg.asr()?;
    let (_cap, rx) = start_capture()?;

    // Keep the managed kiro agent's identity in sync with the wake word.
    sync_persona(cfg);

    // The agent runs under a supervisor thread: the main loop below never
    // blocks on a reply, so it can always hear the wake word — including
    // "Jarvis, 停" to interrupt a running task. Exactly one agent is kept alive.
    eprintln!("[acp] starting agent: {}", cfg.agent_cmd.join(" "));
    let agent = AgentHandle::spawn(cfg.agent_cmd.clone(), cfg.auto_approve);

    println!(
        "== voice assistant ready, say the wake word (\"{}\") ==",
        cfg.wake_display
    );

    // `busy` tracks whether a turn is running; `followup` opens a no-wake
    // listening window right after a turn ends (multi-turn without re-waking).
    let mut busy = false;
    let mut followup = false;

    loop {
        // Absorb agent state without blocking: a finished turn arms a follow-up
        // window; a restart is announced.
        for st in agent.state_rx.try_iter() {
            match st {
                AgentState::Busy => busy = true,
                AgentState::Idle(reason) => {
                    if reason == "cancelled" {
                        println!(">> 已停止");
                    } else if busy {
                        followup = true; // open a no-wake window after a real reply
                    }
                    busy = false;
                }
                AgentState::Restarting(r) => {
                    eprintln!(">> agent 重启中: {r}");
                    busy = false;
                }
                AgentState::Ready => {}
            }
        }

        if followup {
            followup = false;
            vad.reset();
            match record_utterance(&rx, &mut vad, cfg)? {
                Some(audio) => handle_command(audio, &asr, &agent, &mut busy),
                None => println!(">> {}: 好的，我先下线待机了，需要时再叫我。", cfg.persona),
            }
            continue;
        }

        // Wake-word gate. `recv_timeout` so we periodically re-check agent
        // state instead of blocking forever on audio.
        match rx.recv_timeout(Duration::from_millis(200)) {
            Ok(chunk) => {
                let Some(score) = wake.feed(&chunk)? else {
                    continue;
                };
                if score < cfg.wake_threshold {
                    continue;
                }
                let hint = if busy { "，打断中" } else { "" };
                println!("\x07>> wake word detected (score {score:.2}){hint}, listening...");
                vad.reset();
                match record_utterance(&rx, &mut vad, cfg)? {
                    Some(audio) => handle_command(audio, &asr, &agent, &mut busy),
                    None => println!(">> 没听到指令，回到待机"),
                }
                wake.reset();
            }
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                anyhow::bail!("audio capture stopped (input device gone?)")
            }
        }
    }
}

/// Transcribe an utterance and dispatch it: a cancel keyword interrupts the
/// current task; anything else is sent as a prompt (the supervisor redirects
/// automatically if a turn is already running).
fn handle_command(audio: Vec<i16>, asr: &asr::Asr, agent: &AgentHandle, busy: &mut bool) {
    match asr.transcribe(&audio) {
        Ok(text) if text.is_empty() => println!(">> 没听清，请再说一次"),
        Ok(text) => {
            println!(">> you said: {text}");
            if is_cancel_intent(&text) {
                println!(">> 打断当前任务");
                agent.cancel();
                *busy = false;
            } else {
                println!(">> asking agent...\n");
                agent.prompt(text);
                *busy = true;
            }
        }
        Err(e) => eprintln!(">> transcription failed: {e}"),
    }
}

/// Record until the speaker stops talking (VAD-based endpointing).
/// Returns `None` if no speech started within `no_speech_ms` (the caller uses
/// this to fall back to wake-word mode), otherwise the captured utterance.
fn record_utterance(
    rx: &Receiver<Vec<i16>>,
    vad: &mut vad::Vad,
    cfg: &Config,
) -> Result<Option<Vec<i16>>> {
    const CHUNK_MS: f32 = VAD_CHUNK as f32 * 1000.0 / 16000.0; // 32 ms
    const STALL_MS: f32 = 500.0; // wait this long before assuming silence
    let mut audio: Vec<i16> = Vec::new();
    let mut pending: Vec<i16> = Vec::new();
    let mut speech_started = false;
    let mut silence_ms = 0f32;
    let mut total_ms = 0f32;

    loop {
        match rx.recv_timeout(Duration::from_millis(STALL_MS as u64)) {
            Ok(chunk) => {
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
            }
            // No audio arrived in time (device switch, system hiccup). Count it
            // as silence and let the normal timeouts end the turn: a momentary
            // stall must not take down the whole assistant.
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                silence_ms += STALL_MS;
                total_ms += STALL_MS;
            }
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                anyhow::bail!("audio capture stopped (input device gone?)")
            }
        }
        if speech_started && silence_ms >= cfg.silence_ms {
            break; // said something, then went quiet -> done
        }
        if !speech_started && total_ms >= cfg.no_speech_ms {
            break; // never spoke -> give up
        }
        if total_ms >= cfg.max_utterance_ms {
            break; // hard cap
        }
    }
    Ok(if speech_started { Some(audio) } else { None })
}

// ---------------- component tests ----------------

/// Debug: run VAD over a 16 kHz mono s16 WAV file, print per-frame stats.
fn vad_wav(cfg: &Config) -> Result<()> {
    let path = std::env::args().nth(2).expect("usage: vad-wav <file.wav>");
    let bytes = std::fs::read(&path)?;
    // minimal RIFF parse: locate the "data" chunk
    let pos = bytes
        .windows(4)
        .position(|w| w == b"data")
        .expect("no data chunk");
    let pcm = &bytes[pos + 8..];
    let samples: Vec<i16> = pcm
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]))
        .collect();
    let mut vad = cfg.vad()?;
    let (mut n, mut speech, mut max_p) = (0u32, 0u32, 0f32);
    for frame in samples.chunks_exact(VAD_CHUNK) {
        let p = vad.predict(frame)?;
        if p > 0.5 {
            speech += 1;
        }
        max_p = max_p.max(p);
        n += 1;
    }
    println!(
        "{n} frames, {speech} speech frames ({:.0}%), max prob {max_p:.3}",
        speech as f32 / n as f32 * 100.0
    );
    Ok(())
}

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
    match record_utterance(&rx, &mut vad, cfg)? {
        Some(audio) => {
            println!("recorded {:.1}s, transcribing...", audio.len() as f32 / 16000.0);
            let text = asr.transcribe(&audio)?;
            println!("transcription: {text}");
        }
        None => println!("no speech detected"),
    }
    Ok(())
}
