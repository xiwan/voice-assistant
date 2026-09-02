//! Agent-agnostic connection for the Agent Client Protocol (ACP).
//!
//! ACP is a transport standard (JSON-RPC 2.0 over newline-delimited stdio),
//! not something specific to any one agent. We launch a configurable agent
//! command (default `kiro-cli acp --agent voice`) and drive it; swapping to
//! another ACP-speaking backend is a config change, not a code change.
//!
//! This is the LOW-LEVEL half: one `AcpConnection` owns one child process and
//! its stdio. A dedicated reader subthread turns the child's stdout into an
//! `Incoming` channel, so the owner (the supervisor in `agent.rs`) can wait on
//! agent output and control commands at the same time and cancel a running
//! turn — the whole point of not blocking the main loop.
//!
//! Flow (validated in scratch/try_acp.py + try_acp_cancel.py against kiro-cli
//! 2.21.0): initialize -> session/new -> session/prompt; `session/cancel`
//! stops an in-flight turn in ~30ms (stopReason "cancelled") and the session
//! survives for the next prompt. Assistant text arrives as `session/update`
//! notifications (`agent_message_chunk`); `agent_thought_chunk`, `tool_call`
//! and `tool_call_update` are surfaced as dimmed progress lines so a turn that
//! spends time thinking or in tools never looks like a hang. Agent-specific
//! notifications (kiro's `_kiro.dev/*`) are ignored.

use crate::tts::{SpeechBuffer, Tts};
use anyhow::{anyhow, bail, Context, Result};
use crossbeam_channel::{unbounded, Receiver};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, IsTerminal, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

/// Protocol version we speak. ACP is currently at 1.
const PROTOCOL_VERSION: i64 = 1;

/// Longest tool title / thought line we echo, so progress stays one line.
const MAX_STATUS_CHARS: usize = 96;

/// A message read off the agent's stdout, or a signal that the stream ended.
pub enum Incoming {
    Msg(Value),
    /// The agent closed stdout / the reader hit an error — the process is gone.
    Closed,
}

/// What the agent was last streaming, so we insert separators only when the
/// output switches between reply text, thinking, and tool progress.
#[derive(PartialEq)]
enum Stream {
    Idle,
    Message,
    Thought,
}

pub struct AcpConnection {
    child: Child,
    stdin: ChildStdin,
    next_id: i64,
    session_id: String,
    /// Request id of the in-flight `session/prompt`, if a turn is running.
    active_prompt: Option<i64>,
    /// Approve tool-permission requests (the "full" trust mode); otherwise
    /// reject anything the launched agent's own allow-list didn't permit.
    auto_approve: bool,
    stream: Stream,
    color: bool,
    /// Spoken output. Only reply text is routed here — never thoughts or tool
    /// progress — and it is spoken sentence by sentence as the reply streams.
    tts: Tts,
    speech: SpeechBuffer,
}

impl AcpConnection {
    /// Launch `cmd`, start the reader subthread, and complete the ACP handshake
    /// (initialize + session/new). Returns the connection plus the `Incoming`
    /// receiver the caller waits on. Keeping the receiver separate from the
    /// connection lets the supervisor `select!` on it while still holding
    /// `&mut AcpConnection` to handle each message.
    pub fn connect(cmd: &[String], auto_approve: bool, tts: Tts) -> Result<(Self, Receiver<Incoming>)> {
        anyhow::ensure!(!cmd.is_empty(), "agent command is empty");
        let mut child = Command::new(&cmd[0])
            .args(&cmd[1..])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| anyhow!("failed to launch agent `{}`: {e}", cmd.join(" ")))?;
        let stdin = child.stdin.take().context("agent stdin unavailable")?;
        let stdout = child.stdout.take().context("agent stdout unavailable")?;
        let incoming = spawn_reader(stdout);
        let mut c = Self {
            child,
            stdin,
            next_id: 1,
            session_id: String::new(),
            active_prompt: None,
            auto_approve,
            stream: Stream::Idle,
            color: std::io::stdout().is_terminal(),
            tts,
            speech: SpeechBuffer::default(),
        };
        c.handshake(&incoming)?;
        Ok((c, incoming))
    }

    fn handshake(&mut self, incoming: &Receiver<Incoming>) -> Result<()> {
        let init = self.request_blocking(
            incoming,
            "initialize",
            json!({
                "protocolVersion": PROTOCOL_VERSION,
                "clientCapabilities": { "fs": { "readTextFile": false, "writeTextFile": false } },
            }),
        )?;
        let ver = init.get("protocolVersion").and_then(Value::as_i64);
        if ver != Some(PROTOCOL_VERSION) {
            eprintln!("[acp] warning: agent protocolVersion {ver:?}, expected {PROTOCOL_VERSION}");
        }
        let cwd = std::env::current_dir()
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|_| ".".to_string());
        let newsess =
            self.request_blocking(incoming, "session/new", json!({ "cwd": cwd, "mcpServers": [] }))?;
        self.session_id = newsess
            .get("sessionId")
            .and_then(Value::as_str)
            .context("session/new returned no sessionId")?
            .to_string();
        Ok(())
    }

    /// Send a request and block until its response arrives, handling
    /// notifications / agent requests in between. Used only for the handshake;
    /// once running, the supervisor pumps `Incoming` and calls `handle`.
    fn request_blocking(
        &mut self,
        incoming: &Receiver<Incoming>,
        method: &str,
        params: Value,
    ) -> Result<Value> {
        let id = self.next_id;
        self.next_id += 1;
        self.send(&json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params }))?;
        loop {
            match incoming.recv() {
                Ok(Incoming::Msg(v)) => {
                    if v.get("id").and_then(Value::as_i64) == Some(id) && v.get("method").is_none() {
                        if let Some(err) = v.get("error") {
                            bail!("agent error on {method}: {err}");
                        }
                        return Ok(v.get("result").cloned().unwrap_or(Value::Null));
                    }
                    self.handle(&v);
                }
                Ok(Incoming::Closed) | Err(_) => {
                    bail!("agent stream closed during {method}")
                }
            }
        }
    }

    /// Start a new prompt turn (non-blocking). Records the request id so the
    /// matching response can be recognised as the turn's end.
    pub fn send_prompt(&mut self, text: &str) -> Result<()> {
        let id = self.next_id;
        self.next_id += 1;
        self.active_prompt = Some(id);
        self.stream = Stream::Idle;
        self.speech.reset();
        let session_id = self.session_id.clone();
        self.send(&json!({
            "jsonrpc": "2.0", "id": id, "method": "session/prompt",
            "params": { "sessionId": session_id, "prompt": [{ "type": "text", "text": text }] },
        }))
    }

    /// Ask the agent to cancel the in-flight turn (ACP notification, no id).
    pub fn send_cancel(&mut self) -> Result<()> {
        let session_id = self.session_id.clone();
        self.send(&json!({
            "jsonrpc": "2.0", "method": "session/cancel",
            "params": { "sessionId": session_id },
        }))
    }

    pub fn is_busy(&self) -> bool {
        self.active_prompt.is_some()
    }

    /// Cancel the running turn and pump messages until it actually resolves,
    /// or `grace` elapses. On timeout / stream close this returns an error so
    /// the supervisor can escalate to kill + respawn (the hard failsafe).
    pub fn cancel_and_wait(&mut self, incoming: &Receiver<Incoming>, grace: Duration) -> Result<()> {
        if !self.is_busy() {
            return Ok(());
        }
        self.send_cancel()?;
        let deadline = Instant::now() + grace;
        while self.is_busy() {
            let remaining = deadline
                .checked_duration_since(Instant::now())
                .ok_or_else(|| anyhow!("cancel timed out after {grace:?}"))?;
            match incoming.recv_timeout(remaining) {
                Ok(Incoming::Msg(v)) => {
                    self.handle(&v);
                }
                Ok(Incoming::Closed) => bail!("agent stream closed while cancelling"),
                Err(_) => bail!("cancel timed out after {grace:?}"),
            }
        }
        Ok(())
    }

    /// Process one incoming message. Renders reply text / progress, answers
    /// permission requests, and — when the active turn's response arrives —
    /// returns its `stopReason`.
    pub fn handle(&mut self, msg: &Value) -> Option<String> {
        if let Some(id) = msg.get("id").and_then(Value::as_i64) {
            if msg.get("method").is_none() {
                // Response to a request we sent.
                if self.active_prompt == Some(id) {
                    self.active_prompt = None;
                    self.finish_line();
                    let stop = msg["result"]["stopReason"]
                        .as_str()
                        .unwrap_or("end_turn")
                        .to_string();
                    // Speak the tail of the reply (a last sentence without
                    // final punctuation). A cancelled turn stays silent.
                    match self.speech.flush() {
                        Some(rest) if stop != "cancelled" => self.tts.say(rest),
                        _ => {}
                    }
                    return Some(stop);
                }
                return None;
            }
            // Request FROM the agent (permission).
            self.answer_agent_request(id, msg);
            return None;
        }
        if msg.get("method").and_then(Value::as_str) == Some("session/update") {
            self.render_update(&msg["params"]["update"]);
        }
        None
    }

    // ---- rendering ----

    fn render_update(&mut self, update: &Value) {
        match update.get("sessionUpdate").and_then(Value::as_str).unwrap_or("") {
            "agent_message_chunk" => {
                if let Some(text) = update["content"]["text"].as_str() {
                    self.enter(Stream::Message, "");
                    self.out(text);
                    // Reply text is the ONLY thing spoken; each sentence goes
                    // out as soon as it is complete, not at end of turn.
                    for sentence in self.speech.push(text) {
                        self.tts.say(sentence);
                    }
                }
            }
            "agent_thought_chunk" => {
                if let Some(text) = update["content"]["text"].as_str() {
                    self.enter(Stream::Thought, "  · 思考: ");
                    self.out(text);
                }
            }
            "tool_call" => {
                let title = update.get("title").and_then(Value::as_str).unwrap_or("tool");
                self.status(&format!("  · {}...", truncate(title)));
            }
            "tool_call_update" => {
                let status = update.get("status").and_then(Value::as_str).unwrap_or("");
                let title = update.get("title").and_then(Value::as_str).unwrap_or("tool");
                match status {
                    "completed" => self.status(&format!("  ✓ {}", truncate(title))),
                    "failed" => self.status(&format!("  ✗ {} 失败", truncate(title))),
                    _ => {}
                }
            }
            _ => {}
        }
    }

    /// Emit a trailing newline if a turn ended mid-line.
    fn finish_line(&mut self) {
        if self.stream != Stream::Idle {
            self.out("\n");
            self.stream = Stream::Idle;
        }
    }

    fn enter(&mut self, next: Stream, lead: &str) {
        if self.stream == next {
            return;
        }
        if self.stream != Stream::Idle {
            self.out("\n");
        }
        self.stream = next;
        if !lead.is_empty() {
            self.out(lead);
        }
    }

    fn status(&mut self, line: &str) {
        if self.stream != Stream::Idle {
            self.out("\n");
            self.stream = Stream::Idle;
        }
        if self.color {
            self.out(&format!("\x1b[2m{line}\x1b[0m\n"));
        } else {
            self.out(&format!("{line}\n"));
        }
    }

    fn out(&self, s: &str) {
        print!("{s}");
        let _ = std::io::stdout().flush();
    }

    fn send(&mut self, msg: &Value) -> Result<()> {
        let line = serde_json::to_string(msg)?;
        self.stdin
            .write_all(line.as_bytes())
            .and_then(|_| self.stdin.write_all(b"\n"))
            .and_then(|_| self.stdin.flush())
            .context("failed writing to agent stdin (process gone?)")?;
        Ok(())
    }

    fn answer_agent_request(&mut self, id: i64, msg: &Value) {
        let method = msg.get("method").and_then(Value::as_str).unwrap_or("");
        let result = if method == "session/request_permission" {
            let options = msg["params"]["options"].as_array().cloned().unwrap_or_default();
            let tool = msg["params"]["toolCall"]["title"].as_str().unwrap_or("工具").to_string();
            // Voice has no interactive confirmation. In "full" mode we approve;
            // otherwise we reject anything the launched agent's own allow-list
            // did not already permit (so readonly/safe can't be escalated).
            let want = if self.auto_approve { "allow" } else { "reject" };
            let picked = options.iter().find_map(|o| {
                let kind = o.get("kind").or_else(|| o.get("name")).and_then(Value::as_str)?;
                if kind.contains(want) {
                    o.get("optionId").or_else(|| o.get("id")).and_then(Value::as_str)
                } else {
                    None
                }
            });
            let verdict = if self.auto_approve { "已批准" } else { "已拒绝（权限模式）" };
            self.status(&format!("  · {} 请求授权 → {verdict}", truncate(&tool)));
            match picked {
                Some(opt) => json!({ "outcome": { "outcome": "selected", "optionId": opt } }),
                None => json!({ "outcome": { "outcome": "cancelled" } }),
            }
        } else {
            Value::Null
        };
        let _ = self.send(&json!({ "jsonrpc": "2.0", "id": id, "result": result }));
    }
}

impl Drop for AcpConnection {
    fn drop(&mut self) {
        // Terminate the agent process and reap it (no zombies). This is what
        // enforces the "at most one" half of the supervisor's invariant: a
        // connection is never replaced without its child being killed+waited.
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Reader subthread: turn the child's stdout into a channel of `Incoming`.
/// It only reads (never writes), so the owner can write to stdin concurrently.
fn spawn_reader(stdout: ChildStdout) -> Receiver<Incoming> {
    let (tx, rx) = unbounded();
    thread::spawn(move || {
        let mut reader = BufReader::new(stdout);
        loop {
            let mut line = String::new();
            match reader.read_line(&mut line) {
                Ok(0) => {
                    let _ = tx.send(Incoming::Closed);
                    break;
                }
                Ok(_) => {
                    let l = line.trim();
                    if l.is_empty() {
                        continue;
                    }
                    match serde_json::from_str::<Value>(l) {
                        Ok(v) => {
                            if tx.send(Incoming::Msg(v)).is_err() {
                                break; // owner gone
                            }
                        }
                        Err(_) => { /* skip a malformed line rather than die */ }
                    }
                }
                Err(_) => {
                    let _ = tx.send(Incoming::Closed);
                    break;
                }
            }
        }
    });
    rx
}

/// Clip a status line to `MAX_STATUS_CHARS` chars (char-wise, so multi-byte
/// titles are never split) and flatten embedded newlines.
fn truncate(s: &str) -> String {
    let flat = s.replace(['\n', '\r'], " ");
    if flat.chars().count() <= MAX_STATUS_CHARS {
        return flat;
    }
    flat.chars().take(MAX_STATUS_CHARS).collect::<String>() + "…"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_titles_pass_through() {
        assert_eq!(truncate("hello"), "hello");
    }

    #[test]
    fn long_titles_are_clipped() {
        let s = "x".repeat(200);
        let t = truncate(&s);
        assert_eq!(t.chars().count(), MAX_STATUS_CHARS + 1); // + the ellipsis
        assert!(t.ends_with('…'));
    }

    #[test]
    fn newlines_flattened() {
        assert_eq!(truncate("a\nb\rc"), "a b c");
    }
}
