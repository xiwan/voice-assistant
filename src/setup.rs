//! First-run setup, persisted settings, and automatic model download.
//!
//! Settings live in ~/.voice-assistant/config (simple key=value).
//! Models live in ~/.voice-assistant/models (override with VA_MODELS_DIR).
//! Re-run `voice-assistant setup` at any time to change the wake word etc.

use anyhow::{anyhow, Context, Result};
use std::fs;
use std::io::{BufRead, Write};
use std::path::PathBuf;

/// Pretrained openWakeWord models: (id, display name).
pub const WAKE_WORDS: &[(&str, &str)] = &[
    ("hey_jarvis", "Hey Jarvis"),
    ("alexa", "Alexa"),
    ("hey_mycroft", "Hey Mycroft"),
    ("hey_rhasspy", "Hey Rhasspy"),
];

pub const WHISPER_SIZES: &[(&str, &str)] = &[
    ("base", "base   (141MB, 快, 中文精度一般)"),
    ("small", "small  (466MB, 推荐, 中文明显更准)"),
    ("medium", "medium (1.5GB, 最准, 稍慢)"),
];

/// kiro-cli agent permission modes: (id, description).
pub const AGENT_MODES: &[(&str, &str)] = &[
    ("readonly", "只读       (只能读文件, 最安全)"),
    ("safe", "安全命令   (可执行 pwd/ls/git status 等只读命令白名单)"),
    ("full", "完全信任   (可执行任意命令和写文件, 听错指令有风险!)"),
];

const OWW_RELEASE: &str = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1";
const SILERO_URL: &str =
    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx";

#[derive(Clone, Debug)]
pub struct Settings {
    /// Wake word id from WAKE_WORDS, or an absolute path to a custom .onnx.
    pub wake_word: String,
    pub lang: String,
    /// Whisper model size (base/small/medium).
    pub whisper: String,
    pub threshold: f32,
    /// Full command used to launch the ACP agent process. Agent-agnostic:
    /// point this at any ACP-speaking backend. Default: kiro-cli acp --agent voice.
    pub agent_cmd: String,
    /// kiro-cli agent permission mode: readonly / safe / full.
    pub agent_mode: String,
    /// End the utterance after this much trailing silence.
    pub silence_ms: u32,
    /// Give up (back to wake word) if no speech starts within this window.
    pub no_speech_ms: u32,
    /// Hard cap on a single utterance.
    pub max_utterance_ms: u32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            wake_word: "hey_jarvis".into(),
            lang: "auto".into(),
            whisper: "base".into(),
            threshold: 0.5,
            agent_cmd: "kiro-cli acp --agent voice".into(),
            agent_mode: "readonly".into(),
            silence_ms: 1000,
            no_speech_ms: 6000,
            max_utterance_ms: 30000,
        }
    }
}

pub fn base_dir() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".into())).join(".voice-assistant")
}

pub fn models_dir() -> PathBuf {
    std::env::var("VA_MODELS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| base_dir().join("models"))
}

fn config_path() -> PathBuf {
    base_dir().join("config")
}

/// Whether `bin` resolves to an existing file (absolute/relative path) or is
/// found on $PATH. Unix-only lookup, which matches this project's targets.
fn which_on_path(bin: &str) -> bool {
    if bin.contains('/') {
        return PathBuf::from(bin).exists();
    }
    std::env::var("PATH")
        .map(|path| {
            path.split(':')
                .any(|dir| PathBuf::from(dir).join(bin).exists())
        })
        .unwrap_or(false)
}

/// Load persisted settings, if a config file exists.
pub fn load() -> Option<Settings> {
    let text = fs::read_to_string(config_path()).ok()?;
    let mut s = Settings::default();
    for line in text.lines() {
        let Some((k, v)) = line.split_once('=') else {
            continue;
        };
        let v = v.trim();
        match k.trim() {
            "wake_word" => s.wake_word = v.into(),
            "lang" => s.lang = v.into(),
            "whisper" => s.whisper = v.into(),
            "threshold" => s.threshold = v.parse().unwrap_or(0.5),
            "agent_cmd" => s.agent_cmd = v.into(),
            "agent_mode" => s.agent_mode = v.into(),
            "silence_ms" => s.silence_ms = v.parse().unwrap_or(1000),
            "no_speech_ms" => s.no_speech_ms = v.parse().unwrap_or(6000),
            "max_utterance_ms" => s.max_utterance_ms = v.parse().unwrap_or(30000),
            _ => {}
        }
    }
    Some(s)
}

pub fn save(s: &Settings) -> Result<()> {
    fs::create_dir_all(base_dir())?;
    fs::write(
        config_path(),
        format!(
            "wake_word={}\nlang={}\nwhisper={}\nthreshold={}\nagent_cmd={}\n\
             agent_mode={}\nsilence_ms={}\nno_speech_ms={}\nmax_utterance_ms={}\n",
            s.wake_word,
            s.lang,
            s.whisper,
            s.threshold,
            s.agent_cmd,
            s.agent_mode,
            s.silence_ms,
            s.no_speech_ms,
            s.max_utterance_ms
        ),
    )?;
    Ok(())
}

/// Interactive setup (first run, or `voice-assistant setup`). Saves the result.
pub fn interactive_setup(existing: Option<Settings>) -> Result<Settings> {
    let cur = existing.unwrap_or_default();
    let stdin = std::io::stdin();
    let mut lines = stdin.lock().lines();
    let mut ask = |prompt: &str, default: &str| -> String {
        print!("{prompt} [{default}]: ");
        let _ = std::io::stdout().flush();
        match lines.next() {
            Some(Ok(l)) if !l.trim().is_empty() => l.trim().to_string(),
            _ => default.to_string(),
        }
    };

    println!("== voice-assistant 初始设置 (直接回车用默认值) ==\n");

    println!("1) 选择唤醒词:");
    for (i, (_, name)) in WAKE_WORDS.iter().enumerate() {
        println!("   {}. {name}", i + 1);
    }
    println!("   {}. 自定义模型 (输入你自己训练的 .onnx 路径)", WAKE_WORDS.len() + 1);
    let cur_idx = WAKE_WORDS
        .iter()
        .position(|(id, _)| *id == cur.wake_word)
        .map(|i| i + 1)
        .unwrap_or(1);
    let choice = ask("唤醒词编号", &cur_idx.to_string());
    let wake_word = match choice.parse::<usize>() {
        Ok(n) if n >= 1 && n <= WAKE_WORDS.len() => WAKE_WORDS[n - 1].0.to_string(),
        Ok(n) if n == WAKE_WORDS.len() + 1 => {
            let p = ask("自定义 .onnx 模型路径", "");
            anyhow::ensure!(
                p.ends_with(".onnx") && PathBuf::from(&p).exists(),
                "自定义模型文件不存在: {p}"
            );
            p
        }
        _ => choice, // allow typing an id like "alexa" directly
    };

    println!("\n2) 识别语言: zh=中文  en=English  auto=自动检测");
    let lang = ask("语言", &cur.lang);

    println!("\n3) Whisper 模型 (语音转文字精度/速度权衡):");
    for (i, (_, desc)) in WHISPER_SIZES.iter().enumerate() {
        println!("   {}. {desc}", i + 1);
    }
    let cur_widx = WHISPER_SIZES
        .iter()
        .position(|(id, _)| *id == cur.whisper)
        .map(|i| i + 1)
        .unwrap_or(1);
    let wchoice = ask("模型编号", &cur_widx.to_string());
    let whisper = match wchoice.parse::<usize>() {
        Ok(n) if n >= 1 && n <= WHISPER_SIZES.len() => WHISPER_SIZES[n - 1].0.to_string(),
        _ => wchoice,
    };

    // 4) Agent backend. ACP is agent-agnostic, so besides kiro-cli the user
    //    can point voice-assistant at any other ACP-speaking agent on this box.
    println!("\n4) Agent 后端 (通过 ACP 协议对接):");
    println!("   1. kiro-cli            (默认, 自动管理权限)");
    println!("   2. 自定义 ACP 命令     (本机其它 agent, 手填启动命令)");
    let cur_is_kiro = cur.agent_cmd.is_empty() || cur.agent_cmd.starts_with("kiro-cli");
    let bchoice = ask("后端编号", if cur_is_kiro { "1" } else { "2" });

    let (agent_cmd, agent_mode) = if bchoice.trim() == "2" {
        // Custom ACP backend: user supplies the full launch command.
        let cmd = ask("完整启动命令 (例: my-agent acp)", &cur.agent_cmd);
        anyhow::ensure!(!cmd.trim().is_empty(), "启动命令不能为空");
        if let Some(bin) = cmd.split_whitespace().next() {
            if !which_on_path(bin) {
                println!("   ⚠️  未在 PATH 找到 '{bin}'，请确认已安装 (仍会保存)");
            }
        }
        // Voice has no interactive confirmation, so decide tool-approval upfront.
        // We reuse the "full" mode marker to drive auto-approve in the client.
        let approve = ask("自动批准工具授权? (语音无法交互确认) [y/N]", "N");
        let mode = if approve.eq_ignore_ascii_case("y") { "full" } else { "safe" };
        (cmd, mode.to_string())
    } else {
        // kiro-cli: pick a permission mode and generate the managed agent file.
        println!("\n   kiro-cli 权限 (语音指令允许 kiro 做什么):");
        for (i, (_, desc)) in AGENT_MODES.iter().enumerate() {
            println!("     {}. {desc}", i + 1);
        }
        let cur_midx = AGENT_MODES
            .iter()
            .position(|(id, _)| *id == cur.agent_mode)
            .map(|i| i + 1)
            .unwrap_or(1);
        let mchoice = ask("权限编号", &cur_midx.to_string());
        let mode = match mchoice.parse::<usize>() {
            Ok(n) if n >= 1 && n <= AGENT_MODES.len() => AGENT_MODES[n - 1].0.to_string(),
            _ => mchoice,
        };
        write_agent_config(&mode, &persona_name(&wake_word))?;
        // "full" adds -a so the agent auto-approves tool use.
        let mut c = "kiro-cli acp --agent voice".to_string();
        if mode == "full" {
            c.push_str(" -a");
        }
        (c, mode)
    };

    // 5) Runtime tuning. Enter keeps the current value.
    println!("\n5) 运行参数 (回车保留当前值):");
    let threshold = ask("唤醒阈值 (0-1, 越低越灵敏)", &cur.threshold.to_string())
        .parse()
        .unwrap_or(cur.threshold);
    let silence_ms = ask("说完停顿多久算结束 (ms)", &cur.silence_ms.to_string())
        .parse()
        .unwrap_or(cur.silence_ms);
    let no_speech_ms = ask("唤醒后多久没开口就放弃 (ms)", &cur.no_speech_ms.to_string())
        .parse()
        .unwrap_or(cur.no_speech_ms);
    let max_utterance_ms = ask("单次录音上限 (ms)", &cur.max_utterance_ms.to_string())
        .parse()
        .unwrap_or(cur.max_utterance_ms);

    let settings = Settings {
        wake_word,
        lang,
        whisper,
        threshold,
        agent_cmd,
        agent_mode,
        silence_ms,
        no_speech_ms,
        max_utterance_ms,
    };
    save(&settings)?;
    println!("\n已保存到 {} (随时可用 `voice-assistant setup` 修改)\n", config_path().display());
    Ok(settings)
}

/// Persona name for the connected agent, derived from the wake word
/// (e.g. `hey_jarvis` -> "Jarvis"). This makes the agent identify as the wake
/// word rather than as "kiro": the wake word IS the assistant's name.
pub fn persona_name(wake_word: &str) -> String {
    if let Some((_, display)) = WAKE_WORDS.iter().find(|(id, _)| *id == wake_word) {
        return display.strip_prefix("Hey ").unwrap_or(display).to_string();
    }
    // Custom .onnx path or unknown id: derive a name from the file stem.
    let stem = std::path::Path::new(wake_word)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(wake_word)
        .trim_end_matches("_v0.1");
    let mut chars = stem.chars();
    match chars.next() {
        Some(c) => c.to_uppercase().collect::<String>() + chars.as_str(),
        None => "Jarvis".to_string(),
    }
}

/// Write the managed kiro-cli agent (~/.kiro/agents/voice.json) for the chosen
/// permission mode, with the agent's identity bound to `persona` (the wake
/// word). Overwrites previous content: this file is managed by voice-assistant.
pub fn write_agent_config(mode: &str, persona: &str) -> Result<()> {
    let prompt = format!(
        "你的名字是 {persona}，是用户的私人语音助手。有人问你是谁，就回答你是 {persona}。\
         用户的输入来自语音转文字，可能存在识别错误（同音字、专有名词错拼，如 kiro 被识别成 \
         Kerro/Q row、目录被识别成末路），请结合上下文推断真实意图后再回答。回答尽量简短、\
         口语化，适合朗读和快速浏览，避免长篇代码和表格。"
    );
    // Read-only shell commands auto-approved in "safe" mode (regex match).
    const SAFE_COMMANDS: &str = r#""pwd.*", "ls .*", "ls", "cat .*", "head .*", "tail .*", "grep .*", "find .*", "df.*", "du .*", "ps.*", "date.*", "whoami", "uname.*", "which .*", "echo .*", "git status.*", "git log.*", "git diff.*", "git branch.*""#;
    let body = match mode {
        "safe" => format!(
            r#"{{
  "name": "voice",
  "description": "Voice assistant agent (managed by voice-assistant setup, mode: safe)",
  "mcpServers": {{}},
  "tools": ["fs_read", "execute_bash"],
  "allowedTools": ["fs_read"],
  "toolsSettings": {{
    "execute_bash": {{ "allowedCommands": [{SAFE_COMMANDS}] }}
  }},
  "prompt": "{prompt}"
}}
"#
        ),
        "full" => format!(
            r#"{{
  "name": "voice",
  "description": "Voice assistant agent (managed by voice-assistant setup, mode: full)",
  "mcpServers": {{}},
  "tools": ["fs_read", "fs_write", "execute_bash"],
  "allowedTools": ["fs_read", "fs_write", "execute_bash"],
  "prompt": "{prompt}"
}}
"#
        ),
        _ => format!(
            r#"{{
  "name": "voice",
  "description": "Voice assistant agent (managed by voice-assistant setup, mode: readonly)",
  "mcpServers": {{}},
  "tools": ["fs_read"],
  "allowedTools": ["fs_read"],
  "prompt": "{prompt}"
}}
"#
        ),
    };
    let dir = PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".into()))
        .join(".kiro")
        .join("agents");
    fs::create_dir_all(&dir)?;
    fs::write(dir.join("voice.json"), body)?;
    Ok(())
}

/// Resolve the wake word classifier path (custom path or managed model file).
pub fn wake_model_path(s: &Settings) -> PathBuf {
    if s.wake_word.ends_with(".onnx") {
        PathBuf::from(&s.wake_word)
    } else {
        models_dir().join(format!("{}.onnx", s.wake_word))
    }
}

/// Display name of the configured wake word.
pub fn wake_display(s: &Settings) -> String {
    WAKE_WORDS
        .iter()
        .find(|(id, _)| *id == s.wake_word)
        .map(|(_, name)| name.to_string())
        .unwrap_or_else(|| s.wake_word.clone())
}

/// Check for required model files and download any that are missing.
pub fn ensure_models(s: &Settings) -> Result<()> {
    let dir = models_dir();
    fs::create_dir_all(&dir)?;
    let hf_base =
        std::env::var("VA_HF_BASE").unwrap_or_else(|_| "https://hf-mirror.com".to_string());

    let mut needed: Vec<(PathBuf, String)> = vec![
        (
            dir.join("melspectrogram.onnx"),
            format!("{OWW_RELEASE}/melspectrogram.onnx"),
        ),
        (
            dir.join("embedding_model.onnx"),
            format!("{OWW_RELEASE}/embedding_model.onnx"),
        ),
        (dir.join("silero_vad.onnx"), SILERO_URL.to_string()),
    ];
    if !s.wake_word.ends_with(".onnx") {
        anyhow::ensure!(
            WAKE_WORDS.iter().any(|(id, _)| *id == s.wake_word),
            "未知唤醒词 '{}'，可选: {:?}",
            s.wake_word,
            WAKE_WORDS.iter().map(|(id, _)| *id).collect::<Vec<_>>()
        );
        needed.push((
            wake_model_path(s),
            format!("{OWW_RELEASE}/{}_v0.1.onnx", s.wake_word),
        ));
    }
    // A user-supplied VA_WHISPER_MODEL path is managed by the user, not us.
    if std::env::var("VA_WHISPER_MODEL").is_err() {
        needed.push((
            dir.join(format!("ggml-{}.bin", s.whisper)),
            format!("{hf_base}/ggerganov/whisper.cpp/resolve/main/ggml-{}.bin", s.whisper),
        ));
    }

    for (path, url) in needed {
        if !path.exists() {
            download(&url, &path)?;
        }
    }
    Ok(())
}

fn download(url: &str, path: &std::path::Path) -> Result<()> {
    let name = path.file_name().unwrap_or_default().to_string_lossy();
    eprintln!("[setup] 下载 {name} <- {url}");
    let resp = ureq::get(url)
        .call()
        .map_err(|e| anyhow!("下载失败 {url}: {e}"))?;
    let tmp = path.with_extension("part");
    let mut file = fs::File::create(&tmp).with_context(|| format!("创建 {tmp:?}"))?;
    let bytes = std::io::copy(&mut resp.into_reader(), &mut file)
        .map_err(|e| anyhow!("下载中断 {url}: {e}"))?;
    fs::rename(&tmp, path)?;
    eprintln!("[setup] 完成 {name} ({:.1} MB)", bytes as f64 / 1_048_576.0);
    Ok(())
}
