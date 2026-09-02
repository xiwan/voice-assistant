//! Streaming text-to-speech output.
//!
//! Three problems have to be solved together for spoken replies to feel right:
//!
//! 1. **Only speak the reply.** The agent streams thoughts, tool progress and
//!    reply text; only `agent_message_chunk` text reaches this module, and
//!    `SpeechBuffer` additionally drops code blocks, tables and other things
//!    that are unlistenable.
//! 2. **Speak while the text is still arriving.** `SpeechBuffer` cuts the
//!    stream at sentence boundaries (with a comma fallback for long clauses)
//!    and hands each finished sentence to the player immediately, so speech
//!    starts a sentence after generation instead of a reply after it.
//! 3. **Don't listen to ourselves.** Playback would otherwise be re-captured by
//!    the mic and treated as a wake word or a command (an endless loop). The
//!    player raises a shared `muted` flag that the audio callback honours by
//!    zeroing samples — half duplex. Samples keep flowing (just silent) so the
//!    streaming wake-word detector never sees a gap.
//!
//! The engine is a replaceable layer: `Engine::Say` shells out to macOS `say`,
//! and a local neural engine (Piper) can be added without touching the
//! pipeline. Playback runs in its own thread and is killable, so "停" can cut
//! speech off mid-sentence.

use anyhow::{anyhow, Result};
use crossbeam_channel::{unbounded, Receiver, RecvTimeoutError, Sender};
use std::collections::VecDeque;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

/// How often the player checks on the running child / new commands.
const POLL: Duration = Duration::from_millis(20);
/// Keep the mic muted this long after the last word, so room echo dies out
/// before we start listening again.
const TAIL: Duration = Duration::from_millis(250);
/// Emit a partial clause once a run of text without sentence punctuation grows
/// past this many chars (keeps the first words coming quickly).
const SOFT_LIMIT: usize = 48;
/// Never cut a clause shorter than this (avoids choppy one-word utterances).
const MIN_CLAUSE: usize = 12;

/// Which synthesizer to use. Add variants (e.g. Piper) without touching the
/// rest of the pipeline.
#[derive(Clone, Debug)]
pub enum Engine {
    /// TTS disabled: text output only.
    Off,
    /// macOS `say`. Zero dependencies, mediocre voice — the bootstrap engine.
    Say { voice: Option<String>, rate: Option<u32> },
}

impl Engine {
    /// Resolve the configured engine id into an engine, picking a sensible
    /// default voice for the ASR language when none was configured.
    pub fn resolve(id: &str, voice: &str, rate: u32, lang: &str) -> Self {
        match id {
            "say" => {
                let voice = match voice.trim() {
                    "" if lang.starts_with("zh") => Some("Tingting".to_string()),
                    "" => None,
                    v => Some(v.to_string()),
                };
                Engine::Say {
                    voice,
                    rate: (rate > 0).then_some(rate),
                }
            }
            _ => Engine::Off,
        }
    }

    pub fn enabled(&self) -> bool {
        !matches!(self, Engine::Off)
    }

    /// Start speaking `text`, returning the playback process so the caller can
    /// wait on it or kill it (interruption).
    fn start(&self, text: &str) -> Result<Child> {
        match self {
            Engine::Off => Err(anyhow!("tts disabled")),
            Engine::Say { voice, rate } => {
                let mut c = Command::new("say");
                if let Some(v) = voice {
                    c.arg("-v").arg(v);
                }
                if let Some(r) = rate {
                    c.arg("-r").arg(r.to_string());
                }
                // A leading '-' would be parsed as a flag; a leading space
                // makes `say` treat the whole thing as text.
                let safe = if text.starts_with('-') {
                    format!(" {text}")
                } else {
                    text.to_string()
                };
                c.arg(safe)
                    .stdin(Stdio::null())
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .spawn()
                    .map_err(|e| anyhow!("failed to run `say`: {e}"))
            }
        }
    }
}

enum Cmd {
    /// Speak this text. Carries the epoch it was queued in, so utterances
    /// already in flight when the user says "停" are discarded.
    Say(String, u64),
    Stop,
    Shutdown,
}

/// Handle to the playback thread. Cheap to clone; the ACP connection uses a
/// clone to push sentences while the main loop uses one to stop playback.
#[derive(Clone)]
pub struct Tts {
    tx: Option<Sender<Cmd>>,
    /// Number of queued/playing utterances of the current epoch.
    pending: Arc<AtomicUsize>,
    /// Bumped by `stop()` to invalidate everything queued so far.
    epoch: Arc<AtomicU64>,
}

impl Tts {
    /// Start the player thread. `muted` is the flag the audio callback reads:
    /// the player raises it while speaking (half duplex) and clears it a short
    /// tail after the last word.
    pub fn spawn(engine: Engine, muted: Arc<AtomicBool>) -> Self {
        let pending = Arc::new(AtomicUsize::new(0));
        let epoch = Arc::new(AtomicU64::new(0));
        if !engine.enabled() {
            return Self { tx: None, pending, epoch };
        }
        let (tx, rx) = unbounded::<Cmd>();
        let (p, e) = (pending.clone(), epoch.clone());
        thread::spawn(move || player(engine, rx, p, e, muted));
        Self { tx: Some(tx), pending, epoch }
    }

    pub fn enabled(&self) -> bool {
        self.tx.is_some()
    }

    /// Queue one utterance (non-blocking).
    pub fn say(&self, text: impl Into<String>) {
        if let Some(tx) = &self.tx {
            let epoch = self.epoch.load(Ordering::SeqCst);
            self.pending.fetch_add(1, Ordering::SeqCst);
            if tx.send(Cmd::Say(text.into(), epoch)).is_err() {
                self.pending.store(0, Ordering::SeqCst);
            }
        }
    }

    /// Cut playback off now and drop anything queued (used by 停/取消/新指令).
    pub fn stop(&self) {
        if let Some(tx) = &self.tx {
            self.epoch.fetch_add(1, Ordering::SeqCst);
            self.pending.store(0, Ordering::SeqCst);
            let _ = tx.send(Cmd::Stop);
        }
    }

    /// True while something is queued or playing (mic is muted meanwhile).
    pub fn is_speaking(&self) -> bool {
        self.pending.load(Ordering::SeqCst) > 0
    }

    pub fn shutdown(&self) {
        if let Some(tx) = &self.tx {
            self.epoch.fetch_add(1, Ordering::SeqCst);
            self.pending.store(0, Ordering::SeqCst);
            let _ = tx.send(Cmd::Shutdown);
        }
    }
}

/// Playback thread: one utterance at a time, always interruptible.
fn player(
    engine: Engine,
    rx: Receiver<Cmd>,
    pending: Arc<AtomicUsize>,
    epoch: Arc<AtomicU64>,
    muted: Arc<AtomicBool>,
) {
    let mut queue: VecDeque<(String, u64)> = VecDeque::new();
    loop {
        // Idle: wait for work.
        let Some((text, gen)) = queue.pop_front() else {
            match rx.recv() {
                Ok(Cmd::Say(t, g)) => queue.push_back((t, g)),
                Ok(Cmd::Stop) => {}
                Ok(Cmd::Shutdown) | Err(_) => {
                    muted.store(false, Ordering::SeqCst);
                    return;
                }
            }
            continue;
        };
        // Stale utterance (queued before a stop): drop without speaking.
        if gen != epoch.load(Ordering::SeqCst) {
            continue;
        }

        muted.store(true, Ordering::SeqCst);
        let mut child = match engine.start(&text) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[tts] {e}");
                done(&pending, &queue, &muted);
                continue;
            }
        };

        // Wait for playback while staying responsive to Stop and to further
        // sentences arriving from the still-streaming reply.
        let mut interrupted = false;
        loop {
            if matches!(child.try_wait(), Ok(Some(_)) | Err(_)) {
                break;
            }
            match rx.recv_timeout(POLL) {
                Ok(Cmd::Say(t, g)) => queue.push_back((t, g)),
                Ok(Cmd::Stop) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    queue.clear();
                    interrupted = true;
                    break;
                }
                Ok(Cmd::Shutdown) | Err(RecvTimeoutError::Disconnected) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    muted.store(false, Ordering::SeqCst);
                    return;
                }
                Err(RecvTimeoutError::Timeout) => {}
            }
        }

        if interrupted {
            pending.store(0, Ordering::SeqCst);
            muted.store(false, Ordering::SeqCst);
        } else {
            let _ = pending.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| {
                Some(n.saturating_sub(1))
            });
            done(&pending, &queue, &muted);
        }
    }
}

/// Unmute once nothing is left to say (after a short tail for room echo).
fn done(pending: &AtomicUsize, queue: &VecDeque<(String, u64)>, muted: &AtomicBool) {
    if queue.is_empty() && pending.load(Ordering::SeqCst) == 0 {
        thread::sleep(TAIL);
        if pending.load(Ordering::SeqCst) == 0 {
            muted.store(false, Ordering::SeqCst);
        }
    }
}

// ---------------- streaming text -> speakable sentences ----------------

/// Accumulates streamed reply text and yields speakable sentences as soon as
/// each one is complete, so TTS tracks generation instead of waiting for the
/// whole reply. Also the content filter: fenced code blocks, table rows and
/// markdown decoration never reach the synthesizer.
#[derive(Default)]
pub struct SpeechBuffer {
    buf: String,
    in_code: bool,
}

/// Characters that end a spoken sentence.
const SENTENCE_END: &[char] = &['。', '！', '？', '；', '!', '?', ';'];
/// Fallback cut points for a long clause with no sentence end yet.
const CLAUSE_END: &[char] = &['，', '、', ',', '：', ':'];

impl SpeechBuffer {
    /// Feed a streamed chunk; returns the sentences that are now complete.
    pub fn push(&mut self, chunk: &str) -> Vec<String> {
        self.buf.push_str(chunk);
        let mut out = Vec::new();
        loop {
            // Inside a code block: swallow whole lines until the closing fence.
            if self.in_code {
                match self.buf.find('\n') {
                    Some(i) => {
                        let line: String = self.buf.drain(..=i).collect();
                        if line.trim_start().starts_with("```") {
                            self.in_code = false;
                        }
                    }
                    None => break,
                }
                continue;
            }
            let nl = self.buf.find('\n');
            let end = self.buf.find(SENTENCE_END);
            // A sentence ends before the next line break: emit it now.
            if let Some(e) = end.filter(|e| nl.map_or(true, |n| *e < n)) {
                let upto = e + self.buf[e..].chars().next().map_or(1, char::len_utf8);
                let seg: String = self.buf.drain(..upto).collect();
                out.extend(speakable(&seg));
            } else if let Some(n) = nl {
                // A full line with no sentence punctuation (heading, bullet,
                // fence, table row): decide what to do with the whole line.
                let seg: String = self.buf.drain(..=n).collect();
                if seg.trim_start().starts_with("```") {
                    self.in_code = true;
                } else {
                    out.extend(speakable(&seg));
                }
            } else {
                // Nothing terminated yet: cut a long clause at a comma so the
                // first words don't wait for the sentence to finish.
                match self.soft_cut() {
                    Some(seg) => out.extend(speakable(&seg)),
                    None => break,
                }
            }
        }
        out
    }

    /// End of turn: speak whatever is left (unless it's an unterminated code
    /// block). Resets the buffer.
    pub fn flush(&mut self) -> Option<String> {
        let rest = std::mem::take(&mut self.buf);
        let was_code = std::mem::take(&mut self.in_code);
        if was_code || rest.trim_start().starts_with("```") {
            return None;
        }
        speakable(&rest)
    }

    /// Drop buffered text without speaking it (new turn / cancelled turn).
    pub fn reset(&mut self) {
        self.buf.clear();
        self.in_code = false;
    }

    /// If the pending text is long enough, cut it at the last clause boundary.
    fn soft_cut(&mut self) -> Option<String> {
        if self.buf.chars().count() < SOFT_LIMIT {
            return None;
        }
        let cut = self
            .buf
            .char_indices()
            .filter(|(i, c)| CLAUSE_END.contains(c) && self.buf[..*i].chars().count() >= MIN_CLAUSE)
            .map(|(i, c)| i + c.len_utf8())
            .last()?;
        Some(self.buf.drain(..cut).collect())
    }
}

/// Turn one raw segment into something worth speaking, or `None` if it is not
/// listenable (table row, rule, pure punctuation, empty).
fn speakable(seg: &str) -> Option<String> {
    let mut s = seg.trim().to_string();
    if s.is_empty() {
        return None;
    }
    // Markdown table rows and horizontal rules read as noise.
    if s.starts_with('|') || s.chars().all(|c| "-=_*#| \t".contains(c)) {
        return None;
    }
    // Leading list / heading / quote markers.
    s = s
        .trim_start_matches(|c: char| "#>*-+ \t".contains(c))
        .to_string();
    // Inline decoration: `code`, **bold**, _italic_.
    s = s.replace("**", "").replace('`', "").replace('*', "");
    // URLs are unspeakable; say that there is one instead.
    s = replace_urls(&s);
    // Collapse whitespace so `say` doesn't pause oddly.
    s = s.split_whitespace().collect::<Vec<_>>().join(" ");
    // Require at least one letter, digit or CJK char.
    if !s.chars().any(|c| c.is_alphanumeric()) {
        return None;
    }
    Some(s)
}

/// Replace every http(s) URL with "链接".
fn replace_urls(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(i) = rest.find("http") {
        let (head, tail) = rest.split_at(i);
        if !(tail.starts_with("http://") || tail.starts_with("https://")) {
            out.push_str(&rest[..i + 4]);
            rest = &rest[i + 4..];
            continue;
        }
        out.push_str(head);
        out.push_str("链接");
        let end = tail.find(char::is_whitespace).unwrap_or(tail.len());
        rest = &tail[end..];
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Feed a whole reply one char at a time (the streaming worst case) and
    /// collect everything that would be spoken.
    fn stream(text: &str) -> Vec<String> {
        let mut b = SpeechBuffer::default();
        let mut out = Vec::new();
        for ch in text.chars() {
            out.extend(b.push(&ch.to_string()));
        }
        out.extend(b.flush());
        out
    }

    #[test]
    fn sentences_emit_as_they_complete() {
        let mut b = SpeechBuffer::default();
        assert!(b.push("好的，我先看一下").is_empty()); // no boundary yet
        assert_eq!(b.push("代码。"), vec!["好的，我先看一下代码。"]);
        assert_eq!(b.push("找到了两处问题！"), vec!["找到了两处问题！"]);
    }

    #[test]
    fn code_blocks_are_not_spoken() {
        let spoken = stream("这是修复：\n```rust\nfn main() { panic!() }\n```\n改完了。");
        assert_eq!(spoken, vec!["这是修复：", "改完了。"]);
    }

    #[test]
    fn tables_and_rules_are_dropped() {
        let spoken = stream("结果如下\n| a | b |\n|---|---|\n| 1 | 2 |\n就这些。");
        assert_eq!(spoken, vec!["结果如下", "就这些。"]);
    }

    #[test]
    fn markdown_decoration_stripped() {
        let spoken = stream("## 小结\n- **重点**是 `cargo build` 能过。\n");
        assert_eq!(spoken, vec!["小结", "重点是 cargo build 能过。"]);
    }

    #[test]
    fn urls_become_a_word() {
        assert_eq!(
            speakable("详见 https://example.com/x?y=1 这个页面").unwrap(),
            "详见 链接 这个页面"
        );
    }

    #[test]
    fn long_clause_cuts_early_so_speech_starts() {
        // No sentence end anywhere, but commas let us start speaking before
        // the clause is finished (SOFT_LIMIT chars is the trigger).
        let long = "第一件事情是先把音频通道静音掉，第二件事情是把回答按句子切开边收边念，\
                    第三件事情是让一句停可以同时掐掉播放和任务，第四件事情是把代码块过滤掉";
        assert!(long.chars().count() > SOFT_LIMIT);
        let spoken = stream(long);
        assert!(spoken.len() > 1, "expected an early cut, got {spoken:?}");
        assert!(spoken[0].ends_with('，'), "cut should land on a comma: {:?}", spoken[0]);
        // Nothing dropped: the pieces still add up to the original text.
        assert_eq!(spoken.concat(), long.replace(char::is_whitespace, ""));
    }

    #[test]
    fn nothing_is_lost_across_chunk_boundaries() {
        let reply = "第一句话。第二句话！第三句话？收尾";
        let spoken = stream(reply);
        assert_eq!(spoken, vec!["第一句话。", "第二句话！", "第三句话？", "收尾"]);
    }

    #[test]
    fn reset_drops_pending_text() {
        let mut b = SpeechBuffer::default();
        b.push("半句话");
        b.reset();
        assert!(b.flush().is_none());
    }

    #[test]
    fn disabled_engine_is_inert() {
        let muted = Arc::new(AtomicBool::new(false));
        let tts = Tts::spawn(Engine::Off, muted.clone());
        assert!(!tts.enabled());
        tts.say("这句话不会被说出来");
        assert!(!tts.is_speaking());
        assert!(!muted.load(Ordering::SeqCst), "disabled TTS must never mute the mic");
    }

    #[test]
    fn engine_resolves_chinese_voice_by_default() {
        match Engine::resolve("say", "", 0, "zh") {
            Engine::Say { voice, rate } => {
                assert_eq!(voice.as_deref(), Some("Tingting"));
                assert!(rate.is_none());
            }
            e => panic!("unexpected engine {e:?}"),
        }
        assert!(!Engine::resolve("off", "", 0, "zh").enabled());
    }
}
