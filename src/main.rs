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
mod agents;
mod asr;
mod audio;
mod gui;
mod setup;
mod tts;
mod ui;
mod vad;
mod wakeword;

use agent::{AgentHandle, AgentState};
use anyhow::Result;
use crossbeam_channel::{never, select, Receiver};
use std::io::IsTerminal;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Duration;
use ui::{Ui, UiCommand};
use vad::VAD_CHUNK;

/// Everything the pipeline needs, resolved once at startup.
/// `Clone` because the window keeps the main thread and the pipeline runs on a
/// worker with its own copy.
#[derive(Clone)]
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
    /// Spoken-output engine (off / say).
    tts_engine: tts::Engine,
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
        let lang_for_tts = lang.clone();
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
            tts_engine: tts::Engine::resolve(
                &env("VA_TTS", &settings.tts),
                &env("VA_TTS_VOICE", &settings.tts_voice),
                env("VA_TTS_RATE", &settings.tts_rate.to_string())
                    .parse()
                    .unwrap_or(settings.tts_rate),
                &env("VA_TTS_CMD", &settings.tts_cmd),
                &lang_for_tts,
            ),
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
    // Reject an unknown subcommand *before* `Config::load`, which runs first-run
    // setup and can download up to 1.5GB of models — a silly thing to do on the
    // way to printing "unknown command". It also gives CI a cheap way to prove
    // the binary links and starts on a machine with no models and no mic.
    const CMDS: &[&str] = &[
        "selftest", "vad-wav", "test-wake", "test-vad", "test-asr", "test-tts", "ask",
        "agent-test", "events", "gui", "agents",
    ];
    if let Some(other) = std::env::args().nth(1) {
        if !CMDS.contains(&other.as_str()) {
            eprintln!("unknown command: {other}");
            std::process::exit(2);
        }
    }
    let cfg = Config::load()?;
    match std::env::args().nth(1).as_deref() {
        Some("selftest") => selftest(&cfg),
        Some("vad-wav") => vad_wav(&cfg),
        Some("test-wake") => test_wake(&cfg),
        Some("test-vad") => test_vad(&cfg),
        Some("test-asr") => test_asr(&cfg),
        Some("test-tts") => test_tts(&cfg),
        Some("ask") => ask_cli(&cfg),
        Some("agent-test") => agent_test(&cfg),
        Some("events") => events_cli(&cfg),
        Some("gui") => gui::run(cfg),
        Some("agents") => agents_cli(&cfg),
        Some(other) => unreachable!("unknown command {other} is rejected above"),
        None => run(&cfg),
    }
}

/// Start mic capture. `muted` is raised by the TTS player while it speaks, so
/// the assistant never hears its own voice (half duplex).
fn start_capture(muted: Arc<AtomicBool>) -> Result<(audio::Capture, Receiver<Vec<i16>>)> {
    let (tx, rx) = crossbeam_channel::bounded(256);
    let cap = audio::Capture::start(tx, muted)?;
    Ok((cap, rx))
}

/// Capture with no mute source (component test commands).
fn start_capture_unmuted() -> Result<(audio::Capture, Receiver<Vec<i16>>)> {
    start_capture(Arc::new(AtomicBool::new(false)))
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

/// What the user meant by a spoken command while a task exists.
#[derive(Debug, PartialEq)]
enum Intent {
    /// Abandon the current/paused task entirely (drop the resumable).
    Abandon,
    /// Resume the previously paused task.
    Resume,
    /// Interrupt the running task but remember it so it can be resumed.
    Pause,
    /// A normal request.
    New,
}

/// Classify a transcript. Order matters: "不用继续了" is Abandon, not Resume.
fn classify_intent(text: &str) -> Intent {
    const ABANDON: &[&str] = &["算了", "取消", "不用了", "不做了", "别做了", "别弄了", "不用继续"];
    const RESUME: &[&str] = &["继续", "接着", "接下去", "接上"];
    const PAUSE: &[&str] = &["暂停", "等等", "等一下", "等下", "稍等", "停一下", "先停", "停下", "停"];
    let t = text.to_lowercase();
    if ABANDON.iter().any(|w| t.contains(w)) {
        Intent::Abandon
    } else if RESUME.iter().any(|w| t.contains(w)) {
        Intent::Resume
    } else if PAUSE.iter().any(|w| t.contains(w)) {
        Intent::Pause
    } else {
        Intent::New
    }
}

/// Build the prompt that resumes a paused task. The session already retains the
/// interruption and how far the agent got; re-stating the original task makes
/// resume robust even after unrelated interjections in between.
fn resume_prompt(task: &str) -> String {
    format!(
        "接着继续你刚才被打断的任务：「{task}」。从上次停下的地方继续，已经完成的部分不要重复。"
    )
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
    let speaker = tts::Tts::spawn(cfg.tts_engine.clone(), Arc::new(AtomicBool::new(false)));
    let agent = AgentHandle::spawn(
        cfg.agent_cmd.clone(),
        cfg.auto_approve,
        speaker.clone(),
        Ui::terminal(&cfg.persona),
    );
    for (i, p) in prompts.iter().enumerate() {
        println!("\n>> [{}] {p}", i + 1);
        agent.prompt(p.clone());
        wait_for_idle(&agent);
    }
    // Let the spoken tail finish before we tear everything down.
    while speaker.is_speaking() {
        std::thread::sleep(Duration::from_millis(50));
    }
    speaker.shutdown();
    agent.shutdown();
    Ok(())
}

/// Hidden headless test of the supervisor: cancel an in-flight turn, then
/// continue on the same session (proves redirect + session survival). With a
/// bogus VA_AGENT_CMD it instead exercises the restart/backoff failsafe.
fn agent_test(cfg: &Config) -> Result<()> {
    sync_persona(cfg);
    eprintln!("[agent-test] launching: {}", cfg.agent_cmd.join(" "));
    let silent = tts::Tts::spawn(tts::Engine::Off, Arc::new(AtomicBool::new(false)));
    let agent = AgentHandle::spawn(
        cfg.agent_cmd.clone(),
        cfg.auto_approve,
        silent,
        Ui::terminal(&cfg.persona),
    );

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

    let task = "从1数到100，每个数字占一行，并加一句简短说明。";

    println!("\n>> [1] start task, pause after 1.2s");
    agent.prompt(task.into());
    std::thread::sleep(Duration::from_millis(1200));
    println!("\n>> PAUSE (cancel + remember)");
    agent.cancel();
    println!(">> paused, stopReason = {:?} (expect cancelled)", drain_to_idle("pause"));

    println!("\n>> [interject] unrelated quick question");
    agent.prompt("顺便说一句：星期三的英文怎么写？只回答单词。".into());
    println!(">> interject stopReason = {:?}", drain_to_idle("interject"));

    println!("\n>> [resume] 继续 the original task via resume_prompt");
    agent.prompt(resume_prompt(task));
    println!(">> resume stopReason = {:?} (expect end_turn)", drain_to_idle("resume"));

    agent.shutdown();
    Ok(())
}

// ---------------- full pipeline ----------------

/// Hidden headless check of the front-end seam: drive one prompt through the
/// agent with a *channel* front end instead of the terminal and dump every
/// `UiEvent` it produces. This is how the event stream is verified without a
/// mic and without a window — if a window shows nothing, compare against this.
///   voice-assistant events "说三句话，每句一行"
fn events_cli(cfg: &Config) -> Result<()> {
    let prompt = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "用一句话介绍你自己".to_string());
    sync_persona(cfg);
    eprintln!("[acp] starting agent: {}", cfg.agent_cmd.join(" "));
    let (ui, events) = Ui::channel();
    let silent = tts::Tts::spawn(tts::Engine::Off, Arc::new(AtomicBool::new(false)));
    let agent = AgentHandle::spawn(cfg.agent_cmd.clone(), cfg.auto_approve, silent, ui.clone());

    // A front end also sees what the state machine reports, not just the agent.
    ui.ready(&cfg.wake_display);
    ui.transcript(&prompt);
    ui.busy(true);
    agent.prompt(prompt);

    // Print events as they arrive until the turn ends.
    let mut n = 0usize;
    loop {
        match events.recv_timeout(Duration::from_secs(120)) {
            Ok(ev) => {
                n += 1;
                println!("[ev {n:>3}] {ev:?}");
                if let ui::UiEvent::TurnEnd { .. } = ev {
                    break;
                }
            }
            Err(_) => {
                eprintln!("[events] timed out waiting for the turn to end");
                break;
            }
        }
    }
    ui.busy(false);
    agent.shutdown();
    println!("[events] {n} events");
    Ok(())
}

/// Hidden: list the agents this machine can talk to, and optionally prove a hot
/// switch works without restarting the process.
///   voice-assistant agents              — what is installed, how it launches
///   voice-assistant agents switch <id>  — ask one agent its name, switch, ask again
fn agents_cli(cfg: &Config) -> Result<()> {
    println!("当前: {}", cfg.agent_cmd.join(" "));
    if let Some(id) = agents::id_of(&cfg.agent_cmd) {
        println!("识别为: {id}");
    }
    println!();
    for k in agents::KINDS {
        let st = agents::state(k);
        println!("[{}] {} — {}", if st.usable() { "ok" } else { "!!" }, k.label, st.label());
        println!("    启动: {}", agents::argv(k, &cfg.agent_mode).join(" "));
        match st {
            agents::State::NeedsCli => println!("    安装 CLI: {}", k.install_cli),
            agents::State::NeedsAdapter | agents::State::ViaNpx => {
                if let Some(cmd) = agents::install_argv(k) {
                    println!("    安装适配器: {}", cmd.join(" "));
                }
            }
            agents::State::Ready => {}
        }
    }

    let Some(target) = std::env::args().nth(2).filter(|a| a == "switch").and_then(|_| std::env::args().nth(3))
    else {
        return Ok(());
    };
    let Some(kind) = agents::find(&target) else {
        anyhow::bail!("未知 agent: {target}（可选: kiro/claude/codex/gemini）");
    };
    anyhow::ensure!(
        agents::state(kind).usable(),
        "{} 不可用: {}",
        kind.label,
        agents::state(kind).label()
    );

    // One question, a switch, the same question again: the answers should come
    // from different agents while this process never restarts.
    let ui = Ui::terminal(&cfg.persona);
    let silent = tts::Tts::spawn(tts::Engine::Off, Arc::new(AtomicBool::new(false)));
    eprintln!("\n[1] 当前 agent: {}", cfg.agent_cmd.join(" "));
    let agent = AgentHandle::spawn(cfg.agent_cmd.clone(), cfg.auto_approve, silent, ui.clone());
    let q = "你是什么模型？只回答模型或产品名，一行以内。";
    agent.prompt(q.into());
    wait_for_idle(&agent);

    let argv = agents::argv(kind, &cfg.agent_mode);
    eprintln!("\n[2] 热切换到 {}: {}", kind.label, argv.join(" "));
    agent.switch(argv);
    agent.prompt(q.into());
    wait_for_idle(&agent);
    agent.shutdown();
    eprintln!("\n[done] 进程未重启，两次回答若来自不同 agent 即证明切换生效");
    Ok(())
}

fn run(cfg: &Config) -> Result<()> {
    // The terminal is just one front end; a window swaps in here (v0.11.0) by
    // passing `Ui::channel()` and a real command receiver instead.
    run_with(cfg, Ui::terminal(&cfg.persona), never::<UiCommand>())
}

fn run_with(cfg: &Config, ui: Ui, commands: Receiver<UiCommand>) -> Result<()> {
    let mut wake = cfg.wakeword()?;
    let mut vad = cfg.vad()?;
    let asr = cfg.asr()?;

    // Spoken replies. The player owns the `muted` flag that the audio callback
    // honours, which is what keeps the assistant from hearing itself.
    let muted = Arc::new(AtomicBool::new(false));
    let speaker = tts::Tts::spawn(cfg.tts_engine.clone(), muted.clone());
    if speaker.enabled() {
        eprintln!("[tts] 语音回复已开启 ({:?})", cfg.tts_engine);
    }
    let (_cap, rx) = start_capture(muted)?;

    // Keep the managed kiro agent's identity in sync with the wake word.
    sync_persona(cfg);

    // The agent runs under a supervisor thread: the main loop below never
    // blocks on a reply, so it can always hear the wake word — including
    // "Jarvis, 停" to interrupt a running task. Exactly one agent is kept alive.
    eprintln!("[acp] starting agent: {}", cfg.agent_cmd.join(" "));
    let agent = AgentHandle::spawn(
        cfg.agent_cmd.clone(),
        cfg.auto_approve,
        speaker.clone(),
        ui.clone(),
    );

    ui.ready(&cfg.wake_display);

    // Live-adjustable parameters, seeded from config and moved by the settings UI.
    let mut tuning = Tuning::from(cfg);

    // `Conv` holds the conversation state (busy / follow-up window / the running
    // task / a paused-and-resumable task).
    let mut conv = Conv::default();

    loop {
        // Drain agent state without blocking: a finished reply arms a follow-up
        // window; a restart is announced. Cancellations are messaged by
        // handle_command (暂停/取消), so we stay quiet on those here.
        for st in agent.state_rx.try_iter() {
            match st {
                AgentState::Busy => {
                    conv.busy = true;
                    ui.busy(true);
                }
                AgentState::Idle(reason) => {
                    // Only a real answer earns a follow-up window. A cancelled
                    // turn was the user's choice, and a failed one has nothing
                    // to follow up on — opening the mic then just times out.
                    if reason != "cancelled" && reason != "error" && conv.busy {
                        conv.followup = true;
                    }
                    conv.busy = false;
                    ui.busy(false);
                }
                AgentState::Restarting(r) => {
                    ui.restarting(&r);
                    conv.busy = false;
                    ui.busy(false);
                }
                AgentState::Ready => {}
            }
        }

        if conv.followup {
            conv.followup = false;
            // The reply is still being read out: let it finish before opening
            // the window, otherwise the no-speech timer runs during speech.
            drain_while_speaking(&rx, &speaker)?;
            vad.reset();
            wake.reset();
            match record_utterance(&rx, &mut vad, &tuning)? {
                Some(audio) => handle_command(audio, &asr, &agent, &speaker, &ui, &mut conv),
                None => {
                    let bye = "好的，我先下线待机了，需要时再叫我。";
                    ui.spoken(bye);
                    speaker.say(bye);
                }
            }
            continue;
        }

        // Wake-word gate. Block on audio exactly like the proven pipeline so the
        // streaming detector is fed a continuous, gap-free stream (v0.6.1 broke
        // when this went non-blocking). A front-end command is the only other
        // thing that may wake this select, and those are rare, so the detector's
        // input stays gap-free.
        let chunk = select! {
            recv(rx) -> c => c?,
            recv(commands) -> c => {
                match c {
                    // The front end went away (window closed) -> shut down, same
                    // as an explicit Quit. In terminal mode `commands` is
                    // `never()`, so this arm is unreachable.
                    Ok(UiCommand::Quit) | Err(_) => {
                        speaker.stop();
                        speaker.shutdown();
                        agent.shutdown();
                        return Ok(());
                    }
                    Ok(UiCommand::SwitchAgent(id)) => {
                        switch_agent(&id, &agent, &ui, cfg);
                        continue;
                    }
                    Ok(UiCommand::Tune(change)) => {
                        tune(&mut tuning, change, &ui);
                        continue;
                    }
                    Ok(UiCommand::InstallAdapter(id)) => {
                        install_adapter(&id, &ui);
                        continue;
                    }
                    Ok(cmd) => {
                        let (intent, text) = from_command(cmd);
                        act(intent, text, &agent, &speaker, &ui, &mut conv);
                        continue;
                    }
                }
            }
        };
        let Some(score) = wake.feed(&chunk)? else {
            continue;
        };
        ui.wake_score(score);
        if score < tuning.wake_threshold {
            continue;
        }
        ui.wake(score, conv.busy);
        speaker.stop(); // stop talking the moment the user speaks up
        vad.reset();
        match record_utterance(&rx, &mut vad, &tuning)? {
            Some(audio) => handle_command(audio, &asr, &agent, &speaker, &ui, &mut conv),
            None => ui.no_speech(),
        }
        wake.reset();
    }
}

/// Keep consuming (muted) audio until the assistant has finished speaking, so
/// the capture queue doesn't back up and listening starts on a clean stream.
fn drain_while_speaking(rx: &Receiver<Vec<i16>>, speaker: &tts::Tts) -> Result<()> {
    while speaker.is_speaking() {
        match rx.recv_timeout(Duration::from_millis(200)) {
            Ok(_) => {}
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                anyhow::bail!("audio capture stopped (input device gone?)")
            }
        }
    }
    Ok(())
}

/// Conversation state for the main loop.
#[derive(Default)]
struct Conv {
    /// A turn is currently running.
    busy: bool,
    /// Open a no-wake listening window on the next loop (follow-up / after pause).
    followup: bool,
    /// Text of the task currently running (so pause can remember it).
    current_task: Option<String>,
    /// A paused task, resumed by "继续". Sticky: survives interjections.
    resumable: Option<String>,
}

/// Swap the running agent for another one from the registry.
///
/// No process restart: the supervisor kills the current connection and opens a
/// new one, which is the same path it uses to recover from a crash. The choice is
/// persisted so the next start keeps it — written as a launch command, so the
/// config format is unchanged and hand-edited custom commands still work.
fn switch_agent(id: &str, agent: &AgentHandle, ui: &Ui, cfg: &Config) {
    let Some(kind) = agents::find(id) else {
        ui.error(format!("未知 agent: {id}"));
        return;
    };
    let state = agents::state(kind);
    if !state.usable() {
        // Refuse rather than kill a working agent for one that cannot start.
        ui.error(format!("{} 现在不可用（{}）", kind.label, state.label()));
        return;
    }
    let argv = agents::argv(kind, &cfg.agent_mode);
    ui.notice(format!("切换到 {}：{}", kind.label, argv.join(" ")));
    agent.switch(argv.clone());
    if let Some(mut s) = setup::load() {
        s.agent_cmd = argv.join(" ");
        if let Err(e) = setup::save(&s) {
            ui.error(format!("agent 已切换，但写入配置失败: {e}"));
        }
    }
}

/// Install an agent's ACP adapter globally.
///
/// This runs `npm install -g <package>`, i.e. it fetches and executes
/// third-party code — so it only ever happens because the user asked for it, the
/// exact command is reported before it runs, and the result (including npm's own
/// error output) is reported back. It runs on its own thread: npm takes tens of
/// seconds and the pipeline must keep listening.
fn install_adapter(id: &str, ui: &Ui) {
    let Some(kind) = agents::find(id) else {
        ui.error(format!("未知 agent: {id}"));
        return;
    };
    let Some(cmd) = agents::install_argv(kind) else {
        ui.notice(format!("{} 无需适配器", kind.label));
        return;
    };
    ui.notice(format!("正在安装: {}（可能要几十秒）", cmd.join(" ")));
    let ui = ui.clone();
    let label = kind.label.to_string();
    std::thread::spawn(move || {
        match std::process::Command::new(&cmd[0]).args(&cmd[1..]).output() {
            Ok(out) if out.status.success() => {
                ui.notice(format!("{label} 适配器安装完成，现在可以切换过去"))
            }
            Ok(out) => {
                let err = String::from_utf8_lossy(&out.stderr);
                let tail = err.lines().rev().take(2).collect::<Vec<_>>().join(" ");
                ui.error(format!("安装失败: {}", ui::truncate(&tail)));
            }
            Err(e) => ui.error(format!("无法执行 npm: {e}")),
        }
    });
}

/// Transcribe an utterance and dispatch it. Everything after the transcript is
/// shared with front-end commands via `act`, so a button and a spoken phrase
/// take exactly the same path.
fn handle_command(
    audio: Vec<i16>,
    asr: &asr::Asr,
    agent: &AgentHandle,
    speaker: &tts::Tts,
    ui: &Ui,
    conv: &mut Conv,
) {
    let text = match asr.transcribe(&audio) {
        Ok(t) if t.is_empty() => {
            ui.notice("没听清，请再说一次");
            return;
        }
        Ok(t) => t,
        Err(e) => {
            ui.error(format!("transcription failed: {e}"));
            return;
        }
    };
    ui.transcript(&text);
    act(classify_intent(&text), text, agent, speaker, ui, conv);
}

/// Map a front-end command onto the same intents speech produces. `Quit` is
/// handled by the caller (it tears the pipeline down) and `SwitchAgent` by
/// `switch_agent`; neither is a conversation intent.
fn from_command(cmd: UiCommand) -> (Intent, String) {
    match cmd {
        UiCommand::Prompt(text) => (Intent::New, text),
        UiCommand::Pause => (Intent::Pause, String::new()),
        UiCommand::Resume => (Intent::Resume, String::new()),
        UiCommand::Abandon | UiCommand::Quit => (Intent::Abandon, String::new()),
        // Reached only if a caller forgets to intercept these; abandoning is the
        // conservative reading (do not leave a task running against an agent
        // that is being replaced, or through a settings change).
        UiCommand::SwitchAgent(_) => (Intent::Abandon, String::new()),
        UiCommand::Tune(_) | UiCommand::InstallAdapter(_) => (Intent::Abandon, String::new()),
    }
}

/// Apply an intent: pause remembers the running task for later; resume
/// re-launches it; abandon clears it; anything else is a new request (the
/// supervisor auto-redirects if a turn is already running). `text` is the
/// request for `Intent::New` and unused otherwise.
fn act(
    intent: Intent,
    text: String,
    agent: &AgentHandle,
    speaker: &tts::Tts,
    ui: &Ui,
    conv: &mut Conv,
) {
    // The user is doing something: stop reading the previous answer out loud.
    speaker.stop();
    match intent {
        Intent::Pause => {
            if conv.busy {
                agent.cancel();
                conv.resumable = conv.current_task.clone();
                conv.busy = false;
                conv.followup = true; // keep listening for the interjection or 继续
                ui.notice("已暂停，说“继续”接着做，或直接下别的指令");
            } else {
                ui.notice("现在没有进行中的任务");
            }
        }
        Intent::Resume => {
            if let Some(task) = conv.resumable.take() {
                ui.notice("继续刚才的任务");
                agent.prompt(resume_prompt(&task));
                conv.current_task = Some(task);
                conv.busy = true;
            } else {
                ui.notice("没有可继续的任务");
            }
        }
        Intent::Abandon => {
            if conv.busy {
                agent.cancel();
            }
            conv.busy = false;
            conv.current_task = None;
            conv.resumable = None;
            ui.notice("好的，取消了");
        }
        Intent::New => {
            ui.notice("asking agent...");
            agent.prompt(text.clone());
            conv.current_task = Some(text);
            conv.busy = true;
            // resumable stays sticky: this may be an interjection while paused
        }
    }
}

/// The slice of `Config` a front end may move while the pipeline runs.
///
/// These are re-read on every loop iteration, so changing them takes effect at
/// once. Anything needing a model reload (wake word, whisper size, language) is
/// deliberately absent: those go to the config file and wait for a restart.
#[derive(Clone, Copy)]
struct Tuning {
    wake_threshold: f32,
    silence_ms: f32,
    no_speech_ms: f32,
    max_utterance_ms: f32,
}

impl Tuning {
    fn from(cfg: &Config) -> Self {
        Tuning {
            wake_threshold: cfg.wake_threshold,
            silence_ms: cfg.silence_ms,
            no_speech_ms: cfg.no_speech_ms,
            max_utterance_ms: cfg.max_utterance_ms,
        }
    }
}

/// Apply one live parameter change, clamped to a range that cannot brick the
/// pipeline (a 0 threshold would fire constantly; a 0 silence would cut every
/// word), then persist it so a restart keeps the choice.
fn tune(t: &mut Tuning, change: ui::Tunable, ui: &Ui) {
    use ui::Tunable;
    let what = match change {
        Tunable::WakeThreshold(v) => {
            t.wake_threshold = v.clamp(0.05, 0.95);
            format!("唤醒阈值 = {:.2}", t.wake_threshold)
        }
        Tunable::SilenceMs(v) => {
            t.silence_ms = v.clamp(200.0, 5000.0);
            format!("停顿判定 = {:.0}ms", t.silence_ms)
        }
        Tunable::NoSpeechMs(v) => {
            t.no_speech_ms = v.clamp(1000.0, 300_000.0);
            format!("等待开口 = {:.0}ms", t.no_speech_ms)
        }
    };
    ui.notice(what);
    if let Some(mut s) = setup::load() {
        s.threshold = t.wake_threshold;
        s.silence_ms = t.silence_ms as u32;
        s.no_speech_ms = t.no_speech_ms as u32;
        if let Err(e) = setup::save(&s) {
            ui.error(format!("已生效，但写入配置失败: {e}"));
        }
    }
}

/// Record until the speaker stops talking (VAD-based endpointing).
/// Returns `None` if no speech started within `no_speech_ms` (the caller uses
/// this to fall back to wake-word mode), otherwise the captured utterance.
fn record_utterance(
    rx: &Receiver<Vec<i16>>,
    vad: &mut vad::Vad,
    t: &Tuning,
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
        if speech_started && silence_ms >= t.silence_ms {
            break; // said something, then went quiet -> done
        }
        if !speech_started && total_ms >= t.no_speech_ms {
            break; // never spoke -> give up
        }
        if total_ms >= t.max_utterance_ms {
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

    // 4. the *configured* agent backend is reachable. Hardcoding kiro-cli here
    //    made every custom ACP backend look broken.
    let bin = cfg.agent_cmd.first().cloned().unwrap_or_default();
    if bin.is_empty() {
        println!("[!!] agent 命令为空 (检查 agent_cmd / VA_AGENT_CMD)");
    } else if !setup::which_on_path(&bin) {
        println!("[!!] agent 后端 '{bin}' 不在 PATH 上");
    } else if bin == "kiro-cli" {
        // Only the backend we manage is probed with --version: an arbitrary
        // agent might not know the flag and could block waiting on stdin.
        let out = std::process::Command::new(&bin).arg("--version").output();
        match out {
            Ok(o) if o.status.success() => {
                println!("[ok] kiro-cli: {}", String::from_utf8_lossy(&o.stdout).trim())
            }
            _ => println!("[!!] kiro-cli 在 PATH 上但 --version 失败"),
        }
    } else {
        println!("[ok] agent 后端: {bin} (在 PATH 上)");
    }

    // 5. spoken output: is the configured engine actually usable here?
    if cfg.tts_engine.enabled() {
        println!("[ok] tts: {:?}", cfg.tts_engine);
    } else {
        println!("[--] tts: 关闭 (平台默认 = {})", setup::default_tts_id());
    }

    println!("selftest done");
    Ok(())
}

fn test_wake(cfg: &Config) -> Result<()> {
    let mut wake = cfg.wakeword()?;
    let (_cap, rx) = start_capture_unmuted()?;
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
    let (_cap, rx) = start_capture_unmuted()?;
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
    let (_cap, rx) = start_capture_unmuted()?;
    println!("speak now (recording until 1s of silence)...");
    let tuning = Tuning::from(cfg);
    match record_utterance(&rx, &mut vad, &tuning)? {
        Some(audio) => {
            println!("recorded {:.1}s, transcribing...", audio.len() as f32 / 16000.0);
            let text = asr.transcribe(&audio)?;
            println!("transcription: {text}");
        }
        None => println!("no speech detected"),
    }
    Ok(())
}

/// TTS debug: feed a reply through the streaming buffer character by character,
/// exactly as it would arrive from the agent, and speak it. Shows that speech
/// starts at the first finished sentence (not at the end of the reply), that
/// code blocks are skipped, and — with `--interrupt` — that "停" cuts it off.
fn test_tts(cfg: &Config) -> Result<()> {
    anyhow::ensure!(
        cfg.tts_engine.enabled(),
        "语音回复当前关闭，先运行 `voice-assistant setup` 打开 (或设 VA_TTS=say)"
    );
    let args: Vec<String> = std::env::args().skip(2).collect();
    let interrupt = args.iter().any(|a| a == "--interrupt");
    let custom: Vec<&String> = args.iter().filter(|a| !a.starts_with("--")).collect();
    let reply = if custom.is_empty() {
        "好的，我看了一下代码。问题出在音频回调里，播放的时候麦克风还在听。\n\
         ```rust\nlet _ = tx.try_send(out); // 这句不该在静音时执行\n```\n\
         修好了，现在播放期间会把输入通道静音，所以不会再自己听自己说话了。"
            .to_string()
    } else {
        custom.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(" ")
    };

    let muted = Arc::new(AtomicBool::new(false));
    let speaker = tts::Tts::spawn(cfg.tts_engine.clone(), muted);
    let mut buf = tts::SpeechBuffer::default();
    println!("[tts] engine {:?}", cfg.tts_engine);
    // Simulate streaming: one char at a time, ~25ms apart.
    for ch in reply.chars() {
        for sentence in buf.push(&ch.to_string()) {
            println!("  ♪ {sentence}");
            speaker.say(sentence);
        }
        std::thread::sleep(Duration::from_millis(25));
    }
    if let Some(rest) = buf.flush() {
        println!("  ♪ {rest}");
        speaker.say(rest);
    }
    if interrupt {
        std::thread::sleep(Duration::from_millis(1500));
        println!("  ⏹  模拟用户说“停”");
        speaker.stop();
        std::thread::sleep(Duration::from_millis(300));
        println!("  speaking = {} (应为 false)", speaker.is_speaking());
    }
    while speaker.is_speaking() {
        std::thread::sleep(Duration::from_millis(50));
    }
    speaker.shutdown();
    println!("[tts] done");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pause_words() {
        for s in ["暂停", "等等", "等一下", "稍等", "停一下", "停", "先停一下再说"] {
            assert_eq!(classify_intent(s), Intent::Pause, "{s}");
        }
    }

    #[test]
    fn resume_words() {
        for s in ["继续", "接着刚才的", "接下去", "你继续吧"] {
            assert_eq!(classify_intent(s), Intent::Resume, "{s}");
        }
    }

    #[test]
    fn abandon_words() {
        // Abandon must win even when the phrase also contains 继续.
        for s in ["算了", "取消", "不用了", "不用继续了", "别做了"] {
            assert_eq!(classify_intent(s), Intent::Abandon, "{s}");
        }
    }

    #[test]
    fn new_requests() {
        for s in ["帮我看下今天天气", "列出当前目录的文件", "打开日历"] {
            assert_eq!(classify_intent(s), Intent::New, "{s}");
        }
    }

    #[test]
    fn resume_prompt_restates_task() {
        let p = resume_prompt("把报告写完");
        assert!(p.contains("把报告写完"));
        assert!(p.contains("继续"));
    }

    /// A button and a spoken phrase must reach the same intent, otherwise the
    /// window and the voice path would drift apart.
    #[test]
    fn front_end_commands_map_onto_voice_intents() {
        assert_eq!(
            from_command(UiCommand::Prompt("列出文件".into())),
            (Intent::New, "列出文件".to_string())
        );
        assert_eq!(from_command(UiCommand::Pause).0, Intent::Pause);
        assert_eq!(from_command(UiCommand::Resume).0, Intent::Resume);
        assert_eq!(from_command(UiCommand::Abandon).0, Intent::Abandon);
        // Quit tears the pipeline down in the caller; if it ever gets here it
        // must at least not leave a task running.
        assert_eq!(from_command(UiCommand::Quit).0, Intent::Abandon);
    }
}
