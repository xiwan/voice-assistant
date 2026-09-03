//! Which model an agent runs, when the agent lets you choose.
//!
//! Kept out of `agents.rs` deliberately. That module is Protected because every
//! launch command and probe path in it is an externally verified fact, and mixing a
//! second kind of fact (what models a backend offers, which changes weekly) into it
//! is exactly the hazard the protection is about. So the registry still says how to
//! *launch* an agent, and this module says how to *pick a model* for one that
//! supports it — the argv is composed by the caller from both.
//!
//! Verified 2026-09-03 against the installed CLIs:
//!
//! - `kiro-cli acp --model <id>` — "Model ID to use when starting the first
//!   session" (`kiro-cli acp --help`). Ids come from `kiro-cli chat --list-models`.
//! - `dsh` has no model flag at all (`dsh --help`): a Harness profile picks its own
//!   model, so the panel must not offer a choice that would be silently dropped.
//! - Claude Code / Codex: their adapters take their own flags; not verified here,
//!   so they are reported as "no choice from here" rather than guessed at.

/// One selectable model.
#[derive(Clone, Debug, PartialEq)]
pub struct Model {
    pub id: String,
    /// Relative cost as the CLI reports it (e.g. "2.20x credits"). Shown because
    /// picking a model is partly a cost decision.
    pub cost: String,
    /// The CLI's one-line description.
    pub note: String,
    /// The backend's own default.
    pub default: bool,
}

/// How to talk to a backend about models. `None` = it does not take a model flag,
/// which is a fact about that backend rather than a gap in this table.
struct Support {
    /// Flag used at launch, e.g. `--model`.
    flag: &'static str,
    /// argv that prints the available models.
    list: &'static [&'static str],
}

fn support(agent_id: &str) -> Option<Support> {
    match agent_id {
        "kiro" => Some(Support {
            flag: "--model",
            list: &["kiro-cli", "chat", "--list-models"],
        }),
        _ => None,
    }
}

/// Whether the panel should offer a model choice for this agent.
pub fn selectable(agent_id: &str) -> bool {
    support(agent_id).is_some()
}

/// Why there is no choice, for the panel to show instead of an empty control.
pub fn unsupported_note(agent_id: &str) -> &'static str {
    match agent_id {
        "dsh" => "DeepSeek Harness 的模型由 profile 自己决定，命令行没有 --model",
        "claude" | "codex" => "该后端的模型由它自己的配置决定，这里不接管",
        _ => "这个后端没有可选的模型",
    }
}

/// Append the model flag to a launch argv, if the agent takes one and a model was
/// chosen. An empty model means "the backend's default", which is expressed by
/// passing nothing at all rather than by guessing an id.
pub fn apply(argv: &mut Vec<String>, agent_id: &str, model: &str) {
    let model = model.trim();
    if model.is_empty() {
        return;
    }
    if let Some(s) = support(agent_id) {
        argv.push(s.flag.to_string());
        argv.push(model.to_string());
    }
}

/// Ask the backend what it offers. Shells out, so callers do this on demand (when
/// the panel opens), never per frame. An unavailable CLI yields an empty list,
/// which the panel shows as "拿不到列表" instead of pretending.
pub fn list(agent_id: &str) -> Vec<Model> {
    let Some(s) = support(agent_id) else {
        return Vec::new();
    };
    let Ok(out) = std::process::Command::new(s.list[0]).args(&s.list[1..]).output() else {
        return Vec::new();
    };
    parse(&String::from_utf8_lossy(&out.stdout))
}

/// Parse `kiro-cli chat --list-models`:
///
/// ```text
/// Available models (* = default):
///
/// * auto                 1.00x credits      Models chosen by task for optimal usage
///   claude-opus-5        2.20x credits      Claude Opus 5 model with 1M context window
/// ```
fn parse(text: &str) -> Vec<Model> {
    text.lines()
        .filter_map(|line| {
            let default = line.starts_with('*');
            let rest = line.trim_start_matches('*').trim();
            if rest.is_empty() || rest.starts_with("Available models") {
                return None;
            }
            let mut parts = rest.split_whitespace();
            let id = parts.next()?.to_string();
            // Model ids never contain spaces, and the cost column is "<n>x credits".
            let cost = match (parts.next(), parts.next()) {
                (Some(n), Some(unit)) if n.ends_with('x') => format!("{n} {unit}"),
                _ => String::new(),
            };
            let note = parts.collect::<Vec<_>>().join(" ");
            Some(Model { id, cost, note, default })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "\
Available models (* = default):

* auto                 1.00x credits      Models chosen by task for optimal usage
  claude-opus-5        2.20x credits      Claude Opus 5 model with 1M context window
  qwen3-coder-next     0.05x credits      Experimental preview of Qwen3 Coder Next
";

    #[test]
    fn the_default_is_marked_and_not_part_of_the_id() {
        let models = parse(SAMPLE);
        assert_eq!(models.len(), 3);
        assert_eq!(models[0].id, "auto");
        assert!(models[0].default, "the starred line is the default");
        assert!(!models[1].default);
        assert_eq!(models[1].id, "claude-opus-5");
    }

    #[test]
    fn cost_and_description_are_kept_apart() {
        let m = &parse(SAMPLE)[1];
        assert_eq!(m.cost, "2.20x credits");
        assert!(m.note.starts_with("Claude Opus 5"), "{}", m.note);
    }

    #[test]
    fn the_header_and_blank_lines_are_not_models() {
        assert!(parse("Available models (* = default):\n\n").is_empty());
        assert!(parse("").is_empty());
    }

    /// Only backends whose flag was actually verified may receive one; a wrong flag
    /// fails silently at runtime (the launch succeeds, the model is ignored).
    #[test]
    fn only_verified_backends_take_a_model_flag() {
        assert!(selectable("kiro"));
        for other in ["dsh", "claude", "codex", "custom"] {
            assert!(!selectable(other), "{other}");
            let mut argv = vec!["x".to_string()];
            apply(&mut argv, other, "claude-opus-5");
            assert_eq!(argv, vec!["x"], "{other} must not get a model flag");
        }
    }

    #[test]
    fn an_empty_choice_means_the_backend_default() {
        let mut argv = vec!["kiro-cli".into(), "acp".into()];
        apply(&mut argv, "kiro", "   ");
        assert_eq!(argv, vec!["kiro-cli", "acp"]);
        apply(&mut argv, "kiro", "claude-opus-4.8");
        assert_eq!(argv, vec!["kiro-cli", "acp", "--model", "claude-opus-4.8"]);
    }
}
