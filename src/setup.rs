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
    pub kiro_args: String,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            wake_word: "hey_jarvis".into(),
            lang: "auto".into(),
            whisper: "base".into(),
            threshold: 0.5,
            kiro_args: String::new(),
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
            "kiro_args" => s.kiro_args = v.into(),
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
            "wake_word={}\nlang={}\nwhisper={}\nthreshold={}\nkiro_args={}\n",
            s.wake_word, s.lang, s.whisper, s.threshold, s.kiro_args
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

    let settings = Settings {
        wake_word,
        lang,
        whisper,
        threshold: cur.threshold,
        kiro_args: cur.kiro_args,
    };
    save(&settings)?;
    println!("\n已保存到 {} (随时可用 `voice-assistant setup` 修改)\n", config_path().display());
    Ok(settings)
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
