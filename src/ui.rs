//! Front-end boundary.
//!
//! Before this module the pipeline *was* its terminal output: `println!` calls
//! were scattered across the state machine (main.rs) and the ACP renderer
//! (acp.rs), so "what is the assistant doing right now" existed only as text on
//! a tty. A window cannot subscribe to that.
//!
//! Now the pipeline emits `UiEvent`s and accepts `UiCommand`s, and the terminal
//! is just one front end: `Ui::terminal()` spawns a consumer thread that renders
//! the same lines as before. A desktop window uses `Ui::channel()` instead and
//! receives the identical event stream.
//!
//! Two rules keep this honest:
//!
//! - **Events are semantic, formatting lives in the front end.** Callers say
//!   `ui.tool("read_file", ToolState::Completed)`, not "  ✓ read_file". Line
//!   discipline (when a newline is needed because the agent switched from reply
//!   text to a tool line) is the terminal's business, which is why `Term` owns
//!   the `Stream` state that used to live in `AcpConnection`.
//! - **Sending never blocks and never fails.** The pipeline must not stall or
//!   error because a front end is slow or gone, so the channel is unbounded and
//!   send errors are dropped. `Ui::silent()` (tests) discards everything.
//!
//! Startup diagnostics (`[asr] loading ...`, `[acp] starting agent: ...`) stay
//! on stderr as plain `eprintln!`: they happen before a front end exists and
//! describe the process, not the conversation.

use crossbeam_channel::{unbounded, Receiver, Sender};
use std::io::{IsTerminal, Write};
use std::thread;

/// Longest tool title / status line a front end echoes on one line.
const MAX_STATUS_CHARS: usize = 96;

/// Progress of a single agent tool call.
#[derive(Clone, Debug, PartialEq)]
pub enum ToolState {
    Started,
    Completed,
    Failed,
    /// The agent asked for permission; `approved` reflects the trust mode.
    Permission { approved: bool },
}

/// Everything the pipeline tells a front end. Ordering is preserved: all events
/// travel one channel, so a window sees them in the same order a terminal does.
///
/// `allow(dead_code)`: the terminal deliberately ignores some payloads (per-window
/// wake scores, busy flags) and the `events` command only `Debug`-prints them,
/// which dead-code analysis does not count as a read. A window front end reads
/// all of them.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub enum UiEvent {
    /// Pipeline is up and waiting for the wake word.
    Ready { wake_word: String },
    /// Score of an evaluated wake-word window, including the ones below the
    /// threshold — this is the data a level meter needs. Terminals ignore it.
    WakeScore(f32),
    /// The wake word fired. `interrupting` = a turn was already running.
    Wake { score: f32, interrupting: bool },
    /// Wake/follow-up window closed without any speech.
    NoSpeech,
    /// What the user was heard to say.
    Transcript(String),
    /// User-facing status line (paused, cancelled, nothing to resume, ...).
    Notice(String),
    /// Something went wrong but the assistant stays up.
    Error(String),
    /// Reply text streaming from the agent, in arrival-sized chunks.
    Reply(String),
    /// The agent's thinking, streamed. Never spoken.
    Thought(String),
    /// Tool call progress.
    Tool { title: String, state: ToolState },
    /// The turn ended; `reason` is the ACP stopReason (end_turn / cancelled).
    TurnEnd { reason: String },
    /// The supervisor is replacing a dead or wedged agent.
    AgentRestarting(String),
    /// A turn is / is no longer running.
    Busy(bool),
    /// The assistant spoke on its own initiative (e.g. the sign-off).
    Spoken(String),
}

/// What a front end can send back. Voice stops being the only input once there
/// are buttons; both paths funnel through the same dispatch in main.rs.
///
/// `allow(dead_code)`: nothing constructs these in terminal mode (the command
/// receiver there is `never()`); a window front end does.
#[allow(dead_code)]
#[derive(Clone, Debug, PartialEq)]
pub enum UiCommand {
    /// Type a request instead of speaking it.
    Prompt(String),
    /// Interrupt but remember the task (equivalent to saying 暂停).
    Pause,
    /// Resume the paused task (继续).
    Resume,
    /// Interrupt and forget the task (算了).
    Abandon,
    /// Switch to another ACP agent by registry id (kiro / claude / codex /
    /// gemini). Handled without restarting the process: the supervisor drops the
    /// current connection and opens one with the new command.
    SwitchAgent(String),
    /// Shut the pipeline down.
    Quit,
}

/// Handle the pipeline holds. Cloneable and `Send`, because the ACP supervisor
/// thread emits events too. Dropping the receiving end makes every subsequent
/// emit a no-op, so there is no separate "silent" mode to keep in sync.
#[derive(Clone)]
pub struct Ui {
    tx: Sender<UiEvent>,
}

impl Ui {
    /// Render events to the terminal, reproducing the pre-0.10.0 output.
    /// `persona` is the assistant's name, used when it speaks unprompted.
    pub fn terminal(persona: &str) -> Self {
        let (tx, rx) = unbounded();
        let mut term = Term::new(persona, std::io::stdout().is_terminal());
        thread::spawn(move || {
            for ev in rx.iter() {
                term.render(&ev, &mut std::io::stdout());
            }
        });
        Ui { tx }
    }

    /// Feed a front end of your own (a window, or the `events` debug command)
    /// instead of the tty.
    pub fn channel() -> (Self, Receiver<UiEvent>) {
        let (tx, rx) = unbounded();
        (Ui { tx }, rx)
    }

    /// Never blocks, never fails: a slow or absent front end must not be able
    /// to stall the audio pipeline.
    pub fn emit(&self, ev: UiEvent) {
        let _ = self.tx.send(ev);
    }

    pub fn ready(&self, wake_word: &str) {
        self.emit(UiEvent::Ready { wake_word: wake_word.to_string() });
    }
    pub fn wake_score(&self, score: f32) {
        self.emit(UiEvent::WakeScore(score));
    }
    pub fn wake(&self, score: f32, interrupting: bool) {
        self.emit(UiEvent::Wake { score, interrupting });
    }
    pub fn no_speech(&self) {
        self.emit(UiEvent::NoSpeech);
    }
    pub fn transcript(&self, text: &str) {
        self.emit(UiEvent::Transcript(text.to_string()));
    }
    pub fn notice<S: Into<String>>(&self, text: S) {
        self.emit(UiEvent::Notice(text.into()));
    }
    pub fn error<S: Into<String>>(&self, text: S) {
        self.emit(UiEvent::Error(text.into()));
    }
    pub fn reply(&self, text: &str) {
        self.emit(UiEvent::Reply(text.to_string()));
    }
    pub fn thought(&self, text: &str) {
        self.emit(UiEvent::Thought(text.to_string()));
    }
    pub fn tool(&self, title: &str, state: ToolState) {
        self.emit(UiEvent::Tool { title: title.to_string(), state });
    }
    pub fn turn_end(&self, reason: &str) {
        self.emit(UiEvent::TurnEnd { reason: reason.to_string() });
    }
    pub fn restarting(&self, why: &str) {
        self.emit(UiEvent::AgentRestarting(why.to_string()));
    }
    pub fn busy(&self, busy: bool) {
        self.emit(UiEvent::Busy(busy));
    }
    pub fn spoken(&self, text: &str) {
        self.emit(UiEvent::Spoken(text.to_string()));
    }
}

/// What the agent was last streaming, so separators are inserted only when the
/// output switches between reply text, thinking, and single-line status.
/// (Moved here from `AcpConnection`: it is a rendering concern.)
#[derive(PartialEq, Clone, Copy)]
enum Stream {
    Idle,
    Message,
    Thought,
}

/// Terminal renderer. Split out from the thread so it can be unit-tested
/// against a `Vec<u8>` instead of a tty.
struct Term {
    persona: String,
    color: bool,
    stream: Stream,
}

impl Term {
    fn new(persona: &str, color: bool) -> Self {
        Term { persona: persona.to_string(), color, stream: Stream::Idle }
    }

    fn render<W: Write>(&mut self, ev: &UiEvent, w: &mut W) {
        match ev {
            UiEvent::Ready { wake_word } => self.line(
                w,
                &format!("== voice assistant ready, say the wake word (\"{wake_word}\") =="),
            ),
            // A meter's worth of data per audio window: far too chatty for a tty.
            UiEvent::WakeScore(_) | UiEvent::Busy(_) => {}
            UiEvent::Wake { score, interrupting } => {
                let hint = if *interrupting { "，打断中" } else { "" };
                // \x07 (BEL) is the audible "I'm listening" cue.
                self.line(
                    w,
                    &format!("\x07>> wake word detected (score {score:.2}){hint}, listening..."),
                );
            }
            UiEvent::NoSpeech => self.line(w, ">> 没听到指令，回到待机"),
            UiEvent::Transcript(text) => self.line(w, &format!(">> you said: {text}")),
            UiEvent::Notice(text) => self.line(w, &format!(">> {text}")),
            UiEvent::Spoken(text) => self.line(w, &format!(">> {}: {text}", self.persona)),
            UiEvent::Reply(text) => {
                self.enter(w, Stream::Message, "");
                self.out(w, text);
            }
            UiEvent::Thought(text) => {
                self.enter(w, Stream::Thought, "  · 思考: ");
                self.out(w, text);
            }
            UiEvent::Tool { title, state } => {
                let t = truncate(title);
                let line = match state {
                    ToolState::Started => format!("  · {t}..."),
                    ToolState::Completed => format!("  ✓ {t}"),
                    ToolState::Failed => format!("  ✗ {t} 失败"),
                    ToolState::Permission { approved: true } => {
                        format!("  · {t} 请求授权 → 已批准")
                    }
                    ToolState::Permission { approved: false } => {
                        format!("  · {t} 请求授权 → 已拒绝（权限模式）")
                    }
                };
                self.status(w, &line);
            }
            // Close a half-written line so the next prompt starts clean.
            UiEvent::TurnEnd { .. } => {
                if self.stream != Stream::Idle {
                    self.out(w, "\n");
                    self.stream = Stream::Idle;
                }
            }
            // Diagnostics keep going to stderr, as before.
            UiEvent::AgentRestarting(why) => eprintln!(">> agent 重启中: {why}"),
            UiEvent::Error(msg) => eprintln!(">> {msg}"),
        }
    }

    /// A standalone line: break out of any streaming text first.
    fn line<W: Write>(&mut self, w: &mut W, s: &str) {
        if self.stream != Stream::Idle {
            self.out(w, "\n");
            self.stream = Stream::Idle;
        }
        self.out(w, s);
        self.out(w, "\n");
    }

    fn enter<W: Write>(&mut self, w: &mut W, next: Stream, lead: &str) {
        if self.stream == next {
            return;
        }
        if self.stream != Stream::Idle {
            self.out(w, "\n");
        }
        self.stream = next;
        if !lead.is_empty() {
            self.out(w, lead);
        }
    }

    fn status<W: Write>(&mut self, w: &mut W, line: &str) {
        if self.stream != Stream::Idle {
            self.out(w, "\n");
            self.stream = Stream::Idle;
        }
        if self.color {
            self.out(w, &format!("\x1b[2m{line}\x1b[0m\n"));
        } else {
            self.out(w, &format!("{line}\n"));
        }
    }

    fn out<W: Write>(&self, w: &mut W, s: &str) {
        let _ = w.write_all(s.as_bytes());
        let _ = w.flush();
    }
}

/// Clip a status line to `MAX_STATUS_CHARS` chars (char-wise, so multi-byte
/// titles are never split) and flatten embedded newlines.
pub fn truncate(s: &str) -> String {
    let flat = s.replace(['\n', '\r'], " ");
    if flat.chars().count() <= MAX_STATUS_CHARS {
        return flat;
    }
    flat.chars().take(MAX_STATUS_CHARS).collect::<String>() + "…"
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Render a sequence with color off and return what a terminal would show.
    fn render(events: &[UiEvent]) -> String {
        let mut term = Term::new("Jarvis", false);
        let mut out: Vec<u8> = Vec::new();
        for ev in events {
            term.render(ev, &mut out);
        }
        String::from_utf8(out).unwrap()
    }

    #[test]
    fn reply_chunks_stream_without_extra_breaks() {
        let out = render(&[
            UiEvent::Reply("你好".into()),
            UiEvent::Reply("，世界".into()),
            UiEvent::TurnEnd { reason: "end_turn".into() },
        ]);
        assert_eq!(out, "你好，世界\n");
    }

    #[test]
    fn switching_stream_inserts_a_break_and_lead() {
        let out = render(&[
            UiEvent::Reply("答案".into()),
            UiEvent::Thought("再想想".into()),
        ]);
        assert_eq!(out, "答案\n  · 思考: 再想想");
    }

    #[test]
    fn tool_lines_stand_alone() {
        let out = render(&[
            UiEvent::Reply("看一下".into()),
            UiEvent::Tool { title: "read_file".into(), state: ToolState::Started },
            UiEvent::Tool { title: "read_file".into(), state: ToolState::Completed },
        ]);
        assert_eq!(out, "看一下\n  · read_file...\n  ✓ read_file\n");
    }

    #[test]
    fn permission_verdict_reflects_trust_mode() {
        let allowed = render(&[UiEvent::Tool {
            title: "run".into(),
            state: ToolState::Permission { approved: true },
        }]);
        assert!(allowed.contains("已批准"), "{allowed}");
        let denied = render(&[UiEvent::Tool {
            title: "run".into(),
            state: ToolState::Permission { approved: false },
        }]);
        assert!(denied.contains("已拒绝"), "{denied}");
    }

    #[test]
    fn turn_end_on_idle_stream_prints_nothing() {
        assert_eq!(render(&[UiEvent::TurnEnd { reason: "cancelled".into() }]), "");
    }

    #[test]
    fn notice_after_reply_starts_on_its_own_line() {
        let out = render(&[
            UiEvent::Reply("正在做".into()),
            UiEvent::Notice("已暂停".into()),
        ]);
        assert_eq!(out, "正在做\n>> 已暂停\n");
    }

    #[test]
    fn spoken_lines_are_attributed_to_the_persona() {
        let out = render(&[UiEvent::Spoken("我先下线待机了".into())]);
        assert_eq!(out, ">> Jarvis: 我先下线待机了\n");
    }

    #[test]
    fn meter_events_are_not_printed() {
        let out = render(&[UiEvent::WakeScore(0.42), UiEvent::Busy(true)]);
        assert_eq!(out, "");
    }

    #[test]
    fn wake_line_marks_an_interruption() {
        let out = render(&[UiEvent::Wake { score: 0.87, interrupting: true }]);
        assert!(out.contains("score 0.87"), "{out}");
        assert!(out.contains("打断中"), "{out}");
    }

    // ---- truncate (moved from acp.rs with the rest of the formatting) ----

    #[test]
    fn short_titles_pass_through() {
        assert_eq!(truncate("read_file"), "read_file");
    }

    #[test]
    fn long_titles_are_clipped() {
        let long = "x".repeat(MAX_STATUS_CHARS + 20);
        let out = truncate(&long);
        assert_eq!(out.chars().count(), MAX_STATUS_CHARS + 1); // + ellipsis
        assert!(out.ends_with('…'));
    }

    #[test]
    fn newlines_flattened() {
        assert_eq!(truncate("a\nb\rc"), "a b c");
    }
}
