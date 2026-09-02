//! Agent-agnostic client for the Agent Client Protocol (ACP).
//!
//! ACP is a transport standard (JSON-RPC 2.0 over newline-delimited stdio),
//! not something specific to any one agent. We launch a configurable agent
//! command (default `kiro-cli acp --agent voice`) and drive it; swapping to
//! another ACP-speaking backend is a config change, not a code change.
//!
//! Why this instead of spawning `kiro-cli chat` per utterance: the agent
//! process is started ONCE and kept alive, so (a) there is no per-turn cold
//! start and (b) the conversation keeps its context across wake cycles — the
//! session lives in the running process.
//!
//! Flow (validated in scratch/try_acp.py against kiro-cli 2.21.0):
//!   initialize -> session/new -> session/prompt (repeat per turn)
//! Assistant text arrives as `session/update` notifications carrying
//! `agent_message_chunk` content; `agent_thought_chunk`, `tool_call` and
//! `tool_call_update` are surfaced as dimmed progress lines so a turn that
//! spends time in tools doesn't look like a hang. The prompt request resolves
//! with a `stopReason` when the turn ends. Agent-specific notifications (kiro
//! uses the `_kiro.dev/*` namespace) are ignored.

use anyhow::{anyhow, bail, Context, Result};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, IsTerminal, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

/// Protocol version we speak. ACP is currently at 1.
const PROTOCOL_VERSION: i64 = 1;

/// Longest tool title / thought line we echo, so progress stays one line.
const MAX_STATUS_CHARS: usize = 96;

/// What the agent was last streaming, so we can insert separators only when the
/// output actually switches between reply text, thinking, and tool progress.
#[derive(PartialEq)]
enum Stream {
    /// Nothing streamed yet, or the last thing printed ended with a newline.
    Idle,
    /// Mid-line inside the assistant's reply text.
    Message,
    /// Mid-line inside the agent's thinking text.
    Thought,
}

pub struct AcpClient {
    /// The launch command (argv), kept so we can respawn on failure.
    cmd: Vec<String>,
    auto_approve: bool,
    child: Child,
    stdin: ChildStdin,
    reader: BufReader<ChildStdout>,
    next_id: i64,
    session_id: String,
    /// Tracks what is currently being streamed (see `Stream`).
    stream: Stream,
    /// Dim the progress lines only when stdout is a real terminal.
    color: bool,
}

impl AcpClient {
    /// Launch `cmd` and complete the ACP handshake (initialize + session/new).
    ///
    /// `auto_approve` decides how tool-permission requests are answered: when
    /// true we approve them (used for the "full" trust mode); otherwise we
    /// reject anything the agent's own allow-list did not already permit,
    /// since there is no human at the keyboard to ask in voice mode.
    pub fn spawn(cmd: &[String], auto_approve: bool) -> Result<Self> {
        anyhow::ensure!(!cmd.is_empty(), "agent command is empty");
        let (child, stdin, reader) = Self::start_process(cmd)?;
        let mut c = Self {
            cmd: cmd.to_vec(),
            auto_approve,
            child,
            stdin,
            reader,
            next_id: 1,
            session_id: String::new(),
            stream: Stream::Idle,
            color: std::io::stdout().is_terminal(),
        };
        c.handshake()?;
        Ok(c)
    }

    /// Kill the current agent process and start a fresh one with a new session.
    /// Used to recover from a crashed/broken agent without losing the pipeline.
    pub fn respawn(&mut self) -> Result<()> {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let (child, stdin, reader) = Self::start_process(&self.cmd)?;
        self.child = child;
        self.stdin = stdin;
        self.reader = reader;
        self.next_id = 1;
        self.session_id.clear();
        self.stream = Stream::Idle;
        self.handshake()
    }

    fn start_process(cmd: &[String]) -> Result<(Child, ChildStdin, BufReader<ChildStdout>)> {
        let mut child = Command::new(&cmd[0])
            .args(&cmd[1..])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| anyhow!("failed to launch agent `{}`: {e}", cmd.join(" ")))?;
        let stdin = child.stdin.take().context("agent stdin unavailable")?;
        let stdout = child.stdout.take().context("agent stdout unavailable")?;
        Ok((child, stdin, BufReader::new(stdout)))
    }

    fn handshake(&mut self) -> Result<()> {
        let init = self.request(
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
        let newsess = self.request(
            "session/new",
            json!({ "cwd": cwd, "mcpServers": [] }),
        )?;
        self.session_id = newsess
            .get("sessionId")
            .and_then(Value::as_str)
            .context("session/new returned no sessionId")?
            .to_string();
        Ok(())
    }

    /// Send one prompt turn and stream the assistant's reply to stdout, along
    /// with progress lines for thinking and tool calls (so a long turn never
    /// looks like a hang).
    /// Returns when the agent finishes the turn (`stopReason`).
    pub fn prompt(&mut self, text: &str) -> Result<()> {
        let session_id = self.session_id.clone();
        self.stream = Stream::Idle;
        // Immediate feedback: a turn can spend seconds thinking before the
        // first token (kiro doesn't stream thoughts by default), and tool
        // progress only appears once a tool starts. Without this the terminal
        // looks frozen right after the prompt.
        self.status("  · 思考中…");
        self.request(
            "session/prompt",
            json!({
                "sessionId": session_id,
                "prompt": [{ "type": "text", "text": text }],
            }),
        )?;
        if self.stream != Stream::Idle {
            println!();
        }
        self.stream = Stream::Idle;
        println!();
        Ok(())
    }

    // ---- JSON-RPC plumbing ----

    fn send(&mut self, msg: &Value) -> Result<()> {
        let line = serde_json::to_string(msg)?;
        self.stdin
            .write_all(line.as_bytes())
            .and_then(|_| self.stdin.write_all(b"\n"))
            .and_then(|_| self.stdin.flush())
            .context("failed writing to agent stdin (process gone?)")?;
        Ok(())
    }

    /// Send a request and pump the stream until its response arrives, handling
    /// notifications (streamed text) and agent-initiated requests (permissions)
    /// along the way. Returns the `result` object of our request.
    fn request(&mut self, method: &str, params: Value) -> Result<Value> {
        let id = self.next_id;
        self.next_id += 1;
        self.send(&json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params }))?;

        loop {
            let msg = self.read_message()?;

            // Response to one of our requests.
            if let Some(mid) = msg.get("id").and_then(Value::as_i64) {
                if msg.get("method").is_none() {
                    if mid != id {
                        continue; // stale/interleaved response id, ignore
                    }
                    if let Some(err) = msg.get("error") {
                        bail!("agent error on {method}: {err}");
                    }
                    return Ok(msg.get("result").cloned().unwrap_or(Value::Null));
                }
                // Has both id and method => a request FROM the agent.
                self.answer_agent_request(mid, &msg);
                continue;
            }

            // Notification (no id).
            if let Some(m) = msg.get("method").and_then(Value::as_str) {
                self.handle_notification(m, &msg);
            }
        }
    }

    fn read_message(&mut self) -> Result<Value> {
        loop {
            let mut line = String::new();
            let n = self
                .reader
                .read_line(&mut line)
                .context("failed reading from agent stdout")?;
            if n == 0 {
                bail!("agent closed its output stream (process exited?)");
            }
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            return serde_json::from_str(line)
                .with_context(|| format!("agent emitted non-JSON line: {line}"));
        }
    }

    fn handle_notification(&mut self, method: &str, msg: &Value) {
        // Standard ACP updates only. Agent-specific namespaces (kiro's
        // `_kiro.dev/*`, e.g. tool_call_chunk / metadata) are ignored: the
        // standard `tool_call` notification carries the same information.
        if method != "session/update" {
            return;
        }
        let update = &msg["params"]["update"];
        let kind = update.get("sessionUpdate").and_then(Value::as_str).unwrap_or("");
        match kind {
            // The reply itself.
            "agent_message_chunk" => {
                if let Some(text) = update["content"]["text"].as_str() {
                    self.enter(Stream::Message, "");
                    self.out(text);
                }
            }
            // Reasoning, when the model exposes it. Dimmed and prefixed so it
            // is clearly not part of the answer.
            "agent_thought_chunk" => {
                if let Some(text) = update["content"]["text"].as_str() {
                    self.enter(Stream::Thought, "  · 思考: ");
                    self.out(text);
                }
            }
            // A tool is starting: this is where the old client went silent.
            "tool_call" => {
                let title = update.get("title").and_then(Value::as_str).unwrap_or("tool");
                self.status(&format!("  · {}...", truncate(title)));
            }
            // Tool finished (or failed). `in_progress` updates are skipped to
            // avoid one line per chunk of tool output.
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

    /// Switch the output stream, emitting a newline + `lead` label when the
    /// kind of content changes (so reply text and thoughts never run together).
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

    /// Print a one-off progress line (tool activity), on its own line.
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

    /// Respond to an agent-initiated request. The only one we expect is
    /// `session/request_permission`; anything else gets an empty result.
    fn answer_agent_request(&mut self, id: i64, msg: &Value) {
        let method = msg.get("method").and_then(Value::as_str).unwrap_or("");
        let result = if method == "session/request_permission" {
            let options = msg["params"]["options"].as_array().cloned().unwrap_or_default();
            let tool = msg["params"]["toolCall"]["title"]
                .as_str()
                .unwrap_or("工具")
                .to_string();
            // Prefer an allow/reject option matching our policy; fall back to
            // cancelling the request if we can't find a suitable option.
            let want = if self.auto_approve { "allow" } else { "reject" };
            let picked = options.iter().find_map(|o| {
                let kind = o.get("kind").or_else(|| o.get("name")).and_then(Value::as_str)?;
                if kind.contains(want) {
                    o.get("optionId").or_else(|| o.get("id")).and_then(Value::as_str)
                } else {
                    None
                }
            });
            // Make the decision visible: a silent rejection otherwise looks
            // like the agent simply refused for no reason.
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

/// Clip a status line to `MAX_STATUS_CHARS` characters (char-wise, so multi-byte
/// titles are never split mid-character) and flatten embedded newlines.
fn truncate(s: &str) -> String {
    let flat = s.replace(['\n', '\r'], " ");
    if flat.chars().count() <= MAX_STATUS_CHARS {
        return flat;
    }
    flat.chars().take(MAX_STATUS_CHARS).collect::<String>() + "…"
}

#[cfg(test)]
mod tests {
    use super::{truncate, MAX_STATUS_CHARS};

    #[test]
    fn flattens_newlines() {
        assert_eq!(truncate("read\nfile"), "read file");
    }

    #[test]
    fn short_titles_pass_through() {
        assert_eq!(truncate("读取目录"), "读取目录");
    }

    #[test]
    fn clips_on_char_boundaries() {
        // Byte-wise slicing would panic here; char-wise must not.
        let long = "读".repeat(MAX_STATUS_CHARS + 20);
        let out = truncate(&long);
        assert_eq!(out.chars().count(), MAX_STATUS_CHARS + 1); // + the ellipsis
        assert!(out.ends_with('…'));
    }
}

impl Drop for AcpClient {
    fn drop(&mut self) {
        // Terminate the agent process; self.stdin (and thus the agent's input)
        // is closed when the struct's fields drop right after this returns.
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
