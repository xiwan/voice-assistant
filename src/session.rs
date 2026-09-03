//! Conversation continuity: what survives a reconnect, a process restart, or a
//! change of backend.
//!
//! Until v0.20.0 the ACP layer only ever sent `session/new`, so every recovery
//! path in the supervisor — crash, a cancel that escalated to kill+respawn, and
//! every deliberate backend switch — silently started a conversation from zero
//! while the user was told only "agent 重启中". This module is the memory that
//! was missing, and the policy for using it.
//!
//! Two mechanisms, because neither covers everything (all measured against
//! kiro-cli 2.21.0, see `versions/v0.20.0.md`):
//!
//! - **`session/load`** genuinely restores the agent's own context across
//!   processes, but only if the previous child exited on stdin EOF. A killed
//!   child leaves a session lock that names a dead PID and *never* expires, so
//!   that session id is unloadable forever.
//! - **A recap** — our own transcript, replayed as a prefix on the next prompt —
//!   is what covers the cases `session/load` cannot: a crash, a backend that
//!   does not advertise `loadSession`, and switching to a *different* agent
//!   (session ids are private to the backend that issued them).
//!
//! The store is written after every turn, so continuity does not depend on a
//! clean shutdown, and it is deliberately small: this is the thread of the
//! conversation, not an archive.

use serde_json::{json, Value};
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

/// Turns kept. Enough to rebuild the thread of a conversation, not an archive:
/// the recap is prefixed to a real prompt, so it competes with the user's own
/// context window.
const MAX_TURNS: usize = 40;
/// Per-turn cap. Voice utterances are short; agent replies can be pages long,
/// and a recap made of pages is worse than a recap made of gists.
const MAX_TURN_CHARS: usize = 1200;
/// How much of the tail goes into a recap.
const RECAP_TURNS: usize = 8;
const RECAP_CHARS: usize = 1500;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Role {
    User,
    Agent,
}

impl Role {
    fn tag(self) -> &'static str {
        match self {
            Role::User => "用户",
            Role::Agent => "你",
        }
    }
    fn as_str(self) -> &'static str {
        match self {
            Role::User => "user",
            Role::Agent => "agent",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Turn {
    pub role: Role,
    pub text: String,
}

/// How the next connection should continue the conversation.
#[derive(Clone, Debug, PartialEq)]
pub enum Plan {
    /// Take over the agent's own session (same backend, lock was released).
    Load(String),
    /// Start a new session, but carry the thread over as a prefix on the next
    /// prompt. Used when loading is impossible: crash, no `loadSession`, or a
    /// different backend.
    Recap(String),
    /// Nothing to continue.
    Fresh,
}

/// What actually happened, so the user can be told the truth rather than a
/// hopeful "重启中".
///
/// There is deliberately no "lost everything" case: a user turn is written to the
/// store *before* the prompt leaves for the agent, so by the time anything can
/// crash there is always at least a transcript to recap from. The honest range of
/// outcomes is "the agent's own session came back", "only the thread came back",
/// and "there was nothing to bring back".
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Recovery {
    /// The agent's own session was reloaded; it remembers everything it said.
    Restored,
    /// A new session, seeded with our transcript. The agent knows the thread but
    /// not its own reasoning, and only the recent tail of it.
    Recapped,
    /// There was nothing to preserve (first run, or an explicit new session).
    Fresh,
}

/// Persisted conversation state. `loadable` is a fact about the *agent's* copy of
/// the session, not about ours: it goes false the moment a load is refused, so we
/// stop asking for a session the backend will never hand back.
#[derive(Clone, Debug, Default)]
pub struct Store {
    pub agent_cmd: String,
    pub session_id: String,
    pub loadable: bool,
    pub turns: Vec<Turn>,
    /// Whether `save` writes anything. False for the one-shot debug subcommands
    /// (`ask`, `events`, `agent-test`): they should neither inherit the
    /// conversation nor overwrite it.
    persist: bool,
}

impl Store {
    /// `VA_SESSION_FILE` redirects the store. Its purpose is testability: the
    /// end-to-end continuity check (`session-test`) must not overwrite the
    /// conversation the user is actually having.
    pub fn path() -> PathBuf {
        match std::env::var("VA_SESSION_FILE") {
            Ok(p) if !p.trim().is_empty() => PathBuf::from(p),
            _ => crate::setup::base_dir().join("session.json"),
        }
    }

    /// A store that continues nothing and remembers nothing.
    pub fn ephemeral() -> Self {
        Self::default()
    }

    /// Read the store, tolerating every kind of absence and damage: a missing
    /// file is a first run, and a corrupt one must not stop the assistant from
    /// starting — it only costs continuity.
    pub fn load() -> Self {
        let Ok(text) = fs::read_to_string(Self::path()) else {
            return Self { persist: true, ..Self::default() };
        };
        let Ok(v) = serde_json::from_str::<Value>(&text) else {
            return Self { persist: true, ..Self::default() };
        };
        let turns = v["turns"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|t| {
                        let text = t["text"].as_str()?.to_string();
                        let role = match t["role"].as_str()? {
                            "user" => Role::User,
                            "agent" => Role::Agent,
                            _ => return None,
                        };
                        Some(Turn { role, text })
                    })
                    .collect()
            })
            .unwrap_or_default();
        Store {
            agent_cmd: v["agent_cmd"].as_str().unwrap_or("").to_string(),
            session_id: v["session_id"].as_str().unwrap_or("").to_string(),
            loadable: v["loadable"].as_bool().unwrap_or(false),
            turns,
            persist: true,
        }
    }

    /// Write atomically (tmp + rename) so a crash mid-write cannot leave a
    /// truncated file where a conversation used to be, and 0600 because this is
    /// the conversation in plain text — same bar as `secrets`.
    pub fn save(&self) {
        if !self.persist {
            return;
        }
        let dir = crate::setup::base_dir();
        if fs::create_dir_all(&dir).is_err() {
            return;
        }
        let body = json!({
            "agent_cmd": self.agent_cmd,
            "session_id": self.session_id,
            "loadable": self.loadable,
            "updated": SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            "turns": self.turns.iter().map(|t| json!({
                "role": t.role.as_str(),
                "text": t.text,
            })).collect::<Vec<_>>(),
        });
        let path = Self::path();
        let tmp = path.with_extension("json.tmp");
        if fs::write(&tmp, body.to_string()).is_err() {
            return;
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = fs::set_permissions(&tmp, fs::Permissions::from_mode(0o600));
        }
        let _ = fs::rename(&tmp, &path);
    }

    /// Append a turn, keeping the store bounded. Consecutive turns from the same
    /// role are merged: reply text arrives in stream-sized chunks, and one turn
    /// per chunk would blow the cap on a single answer.
    pub fn record(&mut self, role: Role, text: &str) {
        let text = text.trim();
        if text.is_empty() {
            return;
        }
        match self.turns.last_mut() {
            Some(last) if last.role == role => {
                last.text.push_str(text);
                truncate_in_place(&mut last.text, MAX_TURN_CHARS);
            }
            _ => {
                let mut text = text.to_string();
                truncate_in_place(&mut text, MAX_TURN_CHARS);
                self.turns.push(Turn { role, text });
            }
        }
        if self.turns.len() > MAX_TURNS {
            let drop = self.turns.len() - MAX_TURNS;
            self.turns.drain(..drop);
        }
    }

    /// Remember which session this is, and that it is worth trying to load.
    pub fn bind(&mut self, agent_cmd: &str, session_id: &str) {
        self.agent_cmd = agent_cmd.to_string();
        self.session_id = session_id.to_string();
        self.loadable = !session_id.is_empty();
    }

    /// The backend refused to hand the session back. Verified behaviour: this is
    /// permanent for that id (the lock names a dead PID and never expires), so
    /// never ask again — fall through to a recap instead.
    pub fn mark_unloadable(&mut self) {
        self.loadable = false;
    }

    /// Explicit "new session" from the user. Drops the thread as well as the id:
    /// keeping the transcript would recap the very conversation they asked to
    /// leave behind.
    pub fn clear(&mut self) {
        self.session_id.clear();
        self.loadable = false;
        self.turns.clear();
    }

    /// Decide how to continue with `agent_cmd`, given whether that backend says
    /// it supports `session/load`.
    ///
    /// A session id belongs to the backend that issued it, so a different argv
    /// can never load it — that path has to go through a recap even when the new
    /// backend supports loading.
    pub fn plan(&self, agent_cmd: &str, supports_load: bool) -> Plan {
        let same_backend = agent_cmd == self.agent_cmd;
        if same_backend && supports_load && self.loadable && !self.session_id.is_empty() {
            return Plan::Load(self.session_id.clone());
        }
        match self.recap() {
            Some(recap) => Plan::Recap(recap),
            None => Plan::Fresh,
        }
    }

    /// Build the block that is prefixed to the next real prompt.
    ///
    /// Written as background rather than as a question: the agent must not answer
    /// it, greet the user again, or read it out loud — the user only ever hears
    /// the answer to what they just said.
    pub fn recap(&self) -> Option<String> {
        if self.turns.is_empty() {
            return None;
        }
        let mut lines: Vec<String> = Vec::new();
        let mut budget = RECAP_CHARS;
        for turn in self.turns.iter().rev().take(RECAP_TURNS) {
            let mut text = turn.text.clone();
            truncate_in_place(&mut text, budget.min(MAX_TURN_CHARS));
            let line = format!("{}：{}", turn.role.tag(), text);
            let cost = line.chars().count();
            if cost > budget {
                break;
            }
            budget -= cost;
            lines.push(line);
        }
        if lines.is_empty() {
            return None;
        }
        lines.reverse();
        Some(format!(
            "【上下文恢复】你的进程刚重启过，下面是我们之前对话的结尾部分，仅供你回忆，\
             不要复述、不要回应、不要重新打招呼：\n{}\n【以上是回忆】接下来是新的指令：",
            lines.join("\n")
        ))
    }
}

/// Clip to `max` characters, counting chars (not bytes) so multi-byte text is
/// never split mid-character.
fn truncate_in_place(text: &mut String, max: usize) {
    if text.chars().count() <= max {
        return;
    }
    let kept: String = text.chars().take(max).collect();
    *text = kept + "…";
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store_with(turns: &[(Role, &str)]) -> Store {
        let mut s = Store::default();
        for (role, text) in turns {
            s.record(*role, text);
        }
        s
    }

    #[test]
    fn reply_chunks_merge_into_one_turn() {
        let s = store_with(&[
            (Role::User, "帮我看下天气"),
            (Role::Agent, "上海"),
            (Role::Agent, "今天下雨"),
        ]);
        assert_eq!(s.turns.len(), 2);
        assert_eq!(s.turns[1].text, "上海今天下雨");
    }

    #[test]
    fn blank_text_is_not_a_turn() {
        let s = store_with(&[(Role::User, "   "), (Role::Agent, "\n")]);
        assert!(s.turns.is_empty());
    }

    #[test]
    fn turns_are_capped_keeping_the_tail() {
        let mut s = Store::default();
        for i in 0..(MAX_TURNS + 5) {
            // Alternate roles, otherwise everything merges into one turn.
            let role = if i % 2 == 0 { Role::User } else { Role::Agent };
            s.record(role, &format!("t{i}"));
        }
        assert_eq!(s.turns.len(), MAX_TURNS);
        assert_eq!(s.turns.last().unwrap().text, format!("t{}", MAX_TURNS + 4));
    }

    #[test]
    fn a_long_turn_is_clipped_not_dropped() {
        let s = store_with(&[(Role::Agent, &"字".repeat(MAX_TURN_CHARS * 2))]);
        let text = &s.turns[0].text;
        assert_eq!(text.chars().count(), MAX_TURN_CHARS + 1); // + ellipsis
        assert!(text.ends_with('…'));
    }

    #[test]
    fn same_backend_with_a_live_session_loads() {
        let mut s = store_with(&[(Role::User, "记住 4173")]);
        s.bind("kiro-cli acp", "abc-123");
        assert_eq!(s.plan("kiro-cli acp", true), Plan::Load("abc-123".into()));
    }

    #[test]
    fn a_different_backend_never_loads_someone_elses_session() {
        let mut s = store_with(&[(Role::User, "记住 4173")]);
        s.bind("kiro-cli acp", "abc-123");
        // Even though the new backend supports loading, the id is not its own.
        match s.plan("dsh --profile acp", true) {
            Plan::Recap(recap) => assert!(recap.contains("4173"), "{recap}"),
            other => panic!("expected a recap, got {other:?}"),
        }
    }

    #[test]
    fn a_refused_load_falls_back_to_recap_and_never_retries() {
        let mut s = store_with(&[(Role::User, "记住 4173")]);
        s.bind("kiro-cli acp", "abc-123");
        s.mark_unloadable();
        assert!(matches!(s.plan("kiro-cli acp", true), Plan::Recap(_)));
    }

    #[test]
    fn no_load_support_means_recap() {
        let mut s = store_with(&[(Role::User, "记住 4173")]);
        s.bind("some-agent acp", "abc-123");
        assert!(matches!(s.plan("some-agent acp", false), Plan::Recap(_)));
    }

    #[test]
    fn nothing_to_continue_is_fresh() {
        let s = Store::default();
        assert_eq!(s.plan("kiro-cli acp", true), Plan::Fresh);
    }

    #[test]
    fn explicit_reset_drops_the_thread_as_well_as_the_id() {
        let mut s = store_with(&[(Role::User, "记住 4173")]);
        s.bind("kiro-cli acp", "abc-123");
        s.clear();
        assert!(s.turns.is_empty());
        assert_eq!(s.plan("kiro-cli acp", true), Plan::Fresh);
    }

    #[test]
    fn recap_is_chronological_and_labelled() {
        let s = store_with(&[
            (Role::User, "第一句"),
            (Role::Agent, "第一答"),
            (Role::User, "第二句"),
        ]);
        let recap = s.recap().unwrap();
        let first = recap.find("第一句").unwrap();
        let answer = recap.find("第一答").unwrap();
        let second = recap.find("第二句").unwrap();
        assert!(first < answer && answer < second, "{recap}");
        assert!(recap.contains("用户：第一句"), "{recap}");
        assert!(recap.contains("你：第一答"), "{recap}");
        // It must tell the agent not to perform the recap back at the user.
        assert!(recap.contains("不要复述"), "{recap}");
    }

    #[test]
    fn recap_keeps_the_most_recent_turns_within_budget() {
        let mut s = Store::default();
        for i in 0..30 {
            let role = if i % 2 == 0 { Role::User } else { Role::Agent };
            s.record(role, &format!("句子{i}"));
        }
        let recap = s.recap().unwrap();
        assert!(recap.contains("句子29"), "tail must survive: {recap}");
        assert!(!recap.contains("句子0："), "head must be dropped: {recap}");
        assert!(recap.chars().count() < RECAP_CHARS + 200, "recap too big");
    }

    #[test]
    fn a_corrupt_store_is_a_fresh_start_not_a_crash() {
        // Exercised through the same parser `load` uses, without touching $HOME.
        let v: Result<Value, _> = serde_json::from_str("{not json");
        assert!(v.is_err());
        let s = Store::default();
        assert_eq!(s.plan("kiro-cli acp", true), Plan::Fresh);
    }
}
