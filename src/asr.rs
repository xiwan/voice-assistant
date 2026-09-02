//! Speech-to-text using whisper.cpp via whisper-rs (Metal-accelerated on macOS).

use anyhow::{anyhow, Result};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

pub struct Asr {
    ctx: WhisperContext,
    language: String,
    prompt: Option<String>,
}

impl Asr {
    /// `language`: "auto", "zh", "en", ...
    /// `prompt`: optional initial prompt to bias decoding (e.g. towards
    /// Simplified Chinese, since whisper otherwise often emits Traditional).
    pub fn new(model_path: &str, language: &str, prompt: Option<String>) -> Result<Self> {
        let ctx = WhisperContext::new_with_params(model_path, WhisperContextParameters::default())
            .map_err(|e| anyhow!("failed to load whisper model {model_path}: {e:?}"))?;
        Ok(Self {
            ctx,
            language: language.to_string(),
            prompt,
        })
    }

    /// Transcribe 16 kHz mono i16 audio to text.
    pub fn transcribe(&self, audio: &[i16]) -> Result<String> {
        let mut samples: Vec<f32> = audio.iter().map(|&s| s as f32 / 32768.0).collect();
        // whisper needs at least ~1s of audio; pad with silence
        if samples.len() < 16000 {
            samples.resize(16000, 0.0);
        }
        let mut state = self.ctx.create_state()?;
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_language(Some(&self.language));
        if let Some(p) = &self.prompt {
            params.set_initial_prompt(p);
        }
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_suppress_blank(true);
        params.set_no_timestamps(true);
        state.full(params, &samples)?;

        let n = state.full_n_segments()?;
        let mut text = String::new();
        for i in 0..n {
            text.push_str(&state.full_get_segment_text(i)?);
        }
        Ok(clean_transcript(&text))
    }
}

/// Strip whisper non-speech markers like "[BLANK_AUDIO]", "(music)", "♪".
/// Returns an empty string if nothing but markers remained.
fn clean_transcript(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut depth = 0usize;
    for c in text.chars() {
        match c {
            '[' | '(' => depth += 1,
            ']' | ')' => depth = depth.saturating_sub(1),
            '♪' | '♫' => {}
            _ if depth == 0 => out.push(c),
            _ => {}
        }
    }
    out.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::clean_transcript;

    #[test]
    fn strips_non_speech_markers() {
        assert_eq!(clean_transcript("[BLANK_AUDIO]"), "");
        assert_eq!(clean_transcript(" (music) ♪ "), "");
        assert_eq!(clean_transcript("你好 [笑声] 世界"), "你好  世界");
        assert_eq!(clean_transcript("普通句子。"), "普通句子。");
    }
}
