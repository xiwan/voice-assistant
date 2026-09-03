//! Supervised, single-instance agent host.
//!
//! The main (voice) loop must never block on the agent, and there must always
//! be *exactly one* agent that can respond to the user — no zombies, and a dead
//! or wedged agent must be replaced automatically. This module provides that:
//!
//! - `AgentHandle` is what the main loop keeps: a command channel in, a state
//!   channel out. Sending a prompt is non-blocking; the main loop stays free to
//!   listen for the wake word and for "停/cancel".
//! - A single supervisor thread owns exactly one `AcpConnection` at a time.
//!   Because only this thread spawns/kills the child (and every replacement
//!   kills+reaps the previous one via `Drop`), the "at most one" invariant is
//!   structural. An outer loop reconnects with exponential backoff, so the
//!   "at least one" invariant self-heals: crashes, closed streams, and wedged
//!   cancels all funnel into a fresh connection.
//!
//! Failsafe ladder for "always responsive":
//!   1. soft: `session/cancel` (validated ~30ms), used for Cancel and for the
//!      redirect that precedes a new prompt while busy;
//!   2. hard: if the cancel doesn't resolve within `CANCEL_GRACE`, or any
//!      stdio write/read fails, the connection is dropped (kill+reap) and
//!      respawned with a fresh session.

use crate::acp::{AcpConnection, Incoming};
use crate::tts::Tts;
use crate::ui::Ui;
use crossbeam_channel::{select, unbounded, Receiver, RecvTimeoutError, Sender};
use std::thread;
use std::time::Duration;

/// Commands the main loop sends to the agent.
pub enum AgentCmd {
    Prompt(String),
    Cancel,
    /// Replace the agent with a different one, without restarting this process.
    /// The supervisor already kills and respawns connections to recover from
    /// crashes, so switching backends is that same path with a new argv — the
    /// "exactly one agent" invariant is unchanged.
    Switch(Vec<String>),
    Shutdown,
}

/// State the supervisor reports back so the main loop can drive UX (e.g. open a
/// follow-up window on `Idle`, announce a restart).
#[derive(Clone, Debug)]
pub enum AgentState {
    Ready,
    Busy,
    Idle(String), // stopReason: end_turn / cancelled / ...
    Restarting(String),
}

/// How long a `session/cancel` may take before we escalate to kill+respawn.
const CANCEL_GRACE: Duration = Duration::from_secs(3);
const BACKOFF_START: Duration = Duration::from_millis(300);
const BACKOFF_MAX: Duration = Duration::from_secs(5);

pub struct AgentHandle {
    cmd_tx: Sender<AgentCmd>,
    pub state_rx: Receiver<AgentState>,
}

impl AgentHandle {
    /// Start the supervisor thread and return a handle to it.
    /// `tts` is handed to each connection so reply text can be spoken as it
    /// streams; it survives agent restarts because the handle is cloneable.
    /// `ui` is likewise handed down, so progress reaches whatever front end is
    /// attached (terminal or window) instead of being printed here.
    pub fn spawn(cmd: Vec<String>, auto_approve: bool, tts: Tts, ui: Ui) -> Self {
        let (cmd_tx, cmd_rx) = unbounded::<AgentCmd>();
        let (state_tx, state_rx) = unbounded::<AgentState>();
        thread::spawn(move || supervisor(cmd, auto_approve, tts, ui, cmd_rx, state_tx));
        AgentHandle { cmd_tx, state_rx }
    }

    pub fn prompt(&self, text: String) {
        let _ = self.cmd_tx.send(AgentCmd::Prompt(text));
    }
    pub fn cancel(&self) {
        let _ = self.cmd_tx.send(AgentCmd::Cancel);
    }
    /// Swap in a different agent. Takes effect on the next connection, which the
    /// supervisor opens immediately.
    pub fn switch(&self, argv: Vec<String>) {
        let _ = self.cmd_tx.send(AgentCmd::Switch(argv));
    }
    pub fn shutdown(&self) {
        let _ = self.cmd_tx.send(AgentCmd::Shutdown);
    }
}

/// Outcome of running one connection: reconnect (agent died / wedged), replace
/// it with a different agent, or stop.
enum Exit {
    Reconnect,
    Switch(Vec<String>),
    Shutdown,
}

fn supervisor(
    cmd: Vec<String>,
    auto_approve: bool,
    tts: Tts,
    ui: Ui,
    cmd_rx: Receiver<AgentCmd>,
    state_tx: Sender<AgentState>,
) {
    let mut cmd = cmd; // replaced in place by AgentCmd::Switch
    let mut backoff = BACKOFF_START;
    loop {
        match AcpConnection::connect(&cmd, auto_approve, tts.clone(), ui.clone()) {
            Ok((mut conn, incoming)) => {
                backoff = BACKOFF_START; // healthy connection resets backoff
                let _ = state_tx.send(AgentState::Ready);
                match run_connection(&mut conn, &incoming, &cmd_rx, &state_tx) {
                    Exit::Shutdown => {
                        drop(conn); // kill + reap
                        return;
                    }
                    // A switch is a deliberate replacement, so it skips the
                    // backoff a crash would earn and connects straight away.
                    Exit::Switch(next) => {
                        drop(conn); // kill + reap before replacing (invariant)
                        cmd = next;
                        let _ = state_tx.send(AgentState::Restarting(format!(
                            "切换 agent: {}",
                            cmd.join(" ")
                        )));
                        backoff = BACKOFF_START;
                    }
                    Exit::Reconnect => {
                        drop(conn); // kill + reap before replacing (invariant)
                        let _ = state_tx.send(AgentState::Restarting("agent 连接断开，重连中".into()));
                        match wait_or_shutdown(&cmd_rx, backoff) {
                            Downtime::Shutdown => return,
                            Downtime::Switch(next) => {
                                cmd = next;
                                backoff = BACKOFF_START;
                            }
                            Downtime::Elapsed => backoff = (backoff * 2).min(BACKOFF_MAX),
                        }
                    }
                }
            }
            Err(e) => {
                let _ = state_tx.send(AgentState::Restarting(format!("启动失败: {e}")));
                match wait_or_shutdown(&cmd_rx, backoff) {
                    Downtime::Shutdown => return,
                    Downtime::Switch(next) => {
                        cmd = next;
                        backoff = BACKOFF_START;
                    }
                    Downtime::Elapsed => backoff = (backoff * 2).min(BACKOFF_MAX),
                }
            }
        }
    }
}

/// Drive one live connection until it must be replaced or shut down.
fn run_connection(
    conn: &mut AcpConnection,
    incoming: &Receiver<Incoming>,
    cmd_rx: &Receiver<AgentCmd>,
    state_tx: &Sender<AgentState>,
) -> Exit {
    loop {
        select! {
            recv(cmd_rx) -> cmd => match cmd {
                Err(_) => return Exit::Shutdown, // handle dropped: main is gone
                Ok(AgentCmd::Shutdown) => return Exit::Shutdown,
                // Stop the running turn before walking away from this agent, so
                // it isn't killed mid-tool-call if a soft cancel would do.
                Ok(AgentCmd::Switch(next)) => {
                    if conn.is_busy() {
                        let _ = conn.cancel_and_wait(incoming, CANCEL_GRACE);
                    }
                    return Exit::Switch(next);
                }
                Ok(AgentCmd::Cancel) => {
                    if conn.is_busy() {
                        if conn.cancel_and_wait(incoming, CANCEL_GRACE).is_err() {
                            return Exit::Reconnect; // hard failsafe
                        }
                        let _ = state_tx.send(AgentState::Idle("cancelled".into()));
                    }
                }
                Ok(AgentCmd::Prompt(text)) => {
                    // "Redirect": interrupting with a new command cancels the
                    // current turn first (session survives, context kept).
                    if conn.is_busy() && conn.cancel_and_wait(incoming, CANCEL_GRACE).is_err() {
                        return Exit::Reconnect;
                    }
                    if conn.send_prompt(&text).is_err() {
                        return Exit::Reconnect;
                    }
                    let _ = state_tx.send(AgentState::Busy);
                }
            },
            recv(incoming) -> inc => match inc {
                Ok(Incoming::Msg(v)) => {
                    if let Some(stop) = conn.handle(&v) {
                        let _ = state_tx.send(AgentState::Idle(stop));
                    }
                }
                Ok(Incoming::Closed) | Err(_) => return Exit::Reconnect,
            },
        }
    }
}

/// Sleep for `dur` while still honoring commands that arrive during downtime.
/// A switch matters most exactly here: the usual reason to change agents is that
/// the current one is crash-looping, so dropping it would ignore the user at the
/// worst moment.
enum Downtime {
    Elapsed,
    Switch(Vec<String>),
    Shutdown,
}

fn wait_or_shutdown(cmd_rx: &Receiver<AgentCmd>, dur: Duration) -> Downtime {
    match cmd_rx.recv_timeout(dur) {
        Ok(AgentCmd::Shutdown) => Downtime::Shutdown,
        Ok(AgentCmd::Switch(next)) => Downtime::Switch(next),
        Err(RecvTimeoutError::Disconnected) => Downtime::Shutdown, // handle dropped
        // Timeout, or a prompt/cancel we drop while there is no agent to take it.
        _ => Downtime::Elapsed,
    }
}
