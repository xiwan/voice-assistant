//! Which ACP agents this machine can talk to, and how to launch them.
//!
//! ACP is a protocol, so any agent that speaks it works — but "type the right
//! argv" is a poor user experience when the right argv differs per agent and
//! half of them need an adapter package. This module is the small amount of
//! knowledge needed to offer a *list* instead: what CLI must exist, whether an
//! adapter stands between it and ACP, and what to install when it is missing.
//!
//! Two rules, both learned the hard way in this project:
//!
//! - **argv is built in code, never parsed from config.** `agent_cmd` is split on
//!   whitespace, so a path with a space shatters. Config stores an *id*; the
//!   argv comes from here (same reasoning as the `sapi`/`espeak` TTS engines).
//! - **The platform is a parameter, not an ambient fact.** `state_with` and
//!   `argv_with` take the PATH lookup as a closure so every branch is unit-tested
//!   from any machine — v0.11.1 exists because two tests assumed the host.
//!
//! Nothing here installs anything on its own. `install_argv` returns a command
//! for the caller to *show*; running it requires an explicit user action, because
//! it fetches and executes third-party code.

use crate::setup;

/// An agent this build knows how to launch.
pub struct Kind {
    pub id: &'static str,
    pub label: &'static str,
    /// The CLI that must be installed and logged in.
    pub cli: &'static str,
    /// Global binary of the ACP adapter, when the CLI does not speak ACP itself.
    pub adapter_bin: Option<&'static str>,
    /// npm package that provides that adapter.
    pub adapter_pkg: Option<&'static str>,
    /// How the underlying CLI gets installed.
    pub install: Install,
    /// Some agents expose ACP as a *profile* of their own CLI rather than as a
    /// separate binary. dsh is one: `dsh --profile acp` fails with
    /// `profile "acp" does not exist` until the bundle is added, so having the
    /// CLI on PATH is not enough to call the agent usable.
    pub profile: Option<Profile>,
    /// Environment variable this agent reads its API key from, when it needs one.
    /// The others authenticate through their own CLI login, so there is nothing
    /// for this program to hold.
    pub api_key_env: Option<&'static str>,
}

/// An ACP entry point that lives inside the agent's own CLI.
pub struct Profile {
    /// Path under $HOME that exists once the profile is provisioned.
    pub marker: &'static str,
    /// Command that provisions it.
    pub install: &'static [&'static str],
}

/// Whether this program can install the agent's CLI itself.
///
/// npm-distributed CLIs can be installed with one command; kiro-cli is a signed
/// download plus a login, so pretending there is a button for it would be a lie.
pub enum Install {
    /// `npm install -g <pkg>`.
    Npm(&'static str),
    /// Nothing to run — show this to the user instead.
    Manual(&'static str),
}

impl Install {
    /// Human-readable description, shown either way.
    pub fn hint(&self) -> String {
        match self {
            Install::Npm(pkg) => format!("npm install -g {pkg}"),
            Install::Manual(text) => text.to_string(),
        }
    }

    pub fn argv(&self) -> Option<Vec<String>> {
        match self {
            Install::Npm(pkg) => Some(
                ["npm", "install", "-g", pkg]
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
            ),
            Install::Manual(_) => None,
        }
    }
}


/// Package names and flags verified against the npm registry and each project's
/// own docs on 2026-09-03, not from memory:
/// `claude-agent-acp` 0.73.0 and `codex-acp` 1.8.0 under the `agentclientprotocol`
/// scope (the `@zed-industries/*` originals sit at 0.16.x and look superseded),
/// and Gemini CLI's ACP mode is `--experimental-acp`.
pub const KINDS: &[Kind] = &[
    Kind {
        id: "kiro",
        label: "kiro-cli",
        cli: "kiro-cli",
        // Speaks ACP natively, and this program manages its agent file.
        adapter_bin: None,
        adapter_pkg: None,
        install: Install::Manual("从 https://kiro.dev 下载安装，然后 kiro-cli login"),
        profile: None,
        api_key_env: None,
    },
    Kind {
        id: "claude",
        label: "Claude Code",
        cli: "claude",
        adapter_bin: Some("claude-agent-acp"),
        adapter_pkg: Some("@agentclientprotocol/claude-agent-acp"),
        install: Install::Npm("@anthropic-ai/claude-code"),
        profile: None,
        api_key_env: None,
    },
    Kind {
        id: "codex",
        label: "Codex",
        cli: "codex",
        adapter_bin: Some("codex-acp"),
        adapter_pkg: Some("@agentclientprotocol/codex-acp"),
        install: Install::Npm("@openai/codex"),
        profile: None,
        api_key_env: None,
    },
    Kind {
        id: "deepseek",
        label: "DeepSeek Harness",
        cli: "dsh",
        // ACP is not a separate binary here: `@deepseek-ai/dsh-acp` is a plugin
        // with peer deps on dsh internals, surfaced as the `acp` profile. The
        // official docs are explicit — "An ACP v1 SDK client initializes
        // `dsh --profile acp`" — and stdout is reserved for ACP JSON-RPC frames.
        adapter_bin: None,
        adapter_pkg: None,
        install: Install::Npm("@deepseek-ai/dsh"),
        // Verified by running it: without this the launch dies during initialize.
        profile: Some(Profile {
            marker: ".dsh/profiles/acp",
            install: &[
                "dsh",
                "plugin",
                "--profile",
                "acp",
                "add",
                "@deepseek-ai/dsh-acp-app",
            ],
        }),
        // dsh's own config tree declares `apiKeyEnv: DEEPSEEK_API_KEY`; without
        // it the ACP handshake succeeds and then nothing ever answers.
        api_key_env: Some("DEEPSEEK_API_KEY"),
    },
];

/// How ready an agent is, in the order a UI should present it.
#[derive(PartialEq, Eq, Debug, Clone, Copy)]
pub enum State {
    /// Launchable right now with no network round trip.
    Ready,
    /// Launchable, but the adapter runs through `npx` — first start pays a
    /// download, so installing it globally is worth offering.
    ViaNpx,
    /// The CLI is here but its adapter is not, and there is no `npx` to fall
    /// back on.
    NeedsAdapter,
    /// The underlying CLI is missing; the adapter alone would be useless.
    NeedsCli,
}

impl State {
    pub fn label(self) -> &'static str {
        match self {
            State::Ready => "可用",
            State::ViaNpx => "可用 (npx)",
            State::NeedsAdapter => "缺适配器",
            State::NeedsCli => "未安装",
        }
    }

    pub fn usable(self) -> bool {
        matches!(self, State::Ready | State::ViaNpx)
    }
}

pub fn find(id: &str) -> Option<&'static Kind> {
    KINDS.iter().find(|k| k.id == id)
}

pub fn state(k: &Kind) -> State {
    state_with(k, setup::which_on_path, |rel| setup::home().join(rel).exists())
}

/// `home_has` answers "does this path exist under $HOME", kept as a parameter for
/// the same reason as `on_path`: so the profile branch is testable anywhere.
pub fn state_with(
    k: &Kind,
    on_path: impl Fn(&str) -> bool,
    home_has: impl Fn(&str) -> bool,
) -> State {
    if !on_path(k.cli) {
        return State::NeedsCli;
    }
    if let Some(pf) = &k.profile {
        // The CLI is here but its ACP profile may not be provisioned yet.
        return if home_has(pf.marker) {
            State::Ready
        } else {
            State::NeedsAdapter
        };
    }
    match k.adapter_bin {
        None => State::Ready, // the CLI speaks ACP itself
        Some(bin) if on_path(bin) => State::Ready,
        Some(_) if on_path("npx") => State::ViaNpx,
        Some(_) => State::NeedsAdapter,
    }
}

/// The argv to launch this agent over ACP. `mode` is the kiro permission mode;
/// other agents carry their own trust configuration.
pub fn argv(k: &Kind, mode: &str) -> Vec<String> {
    argv_with(k, mode, setup::which_on_path)
}

pub fn argv_with(k: &Kind, mode: &str, on_path: impl Fn(&str) -> bool) -> Vec<String> {
    let s = |v: &str| v.to_string();
    match k.id {
        "kiro" => {
            let mut argv = vec![s("kiro-cli"), s("acp"), s("--agent"), s("voice")];
            if mode == "full" {
                argv.push(s("-a")); // auto-approve tool permissions
            }
            argv
        }
        // The profile flag is what turns dsh into an ACP server.
        "deepseek" => vec![s("dsh"), s("--profile"), s("acp")],
        _ => match (k.adapter_bin, k.adapter_pkg) {
            // Prefer the installed binary: no npx resolution on every launch,
            // and the supervisor may respawn it after a crash.
            (Some(bin), _) if on_path(bin) => vec![s(bin)],
            // `-y` so a missing package installs without an interactive prompt
            // that nobody can answer — this process has no terminal to type in.
            (_, Some(pkg)) => vec![s("npx"), s("-y"), s(pkg)],
            _ => vec![s(k.cli)],
        },
    }
}

/// Which agent an existing `agent_cmd` refers to, if any.
///
/// Config keeps storing the launch command rather than an id, so nothing about
/// the file format changes and a hand-written custom command still works. The id
/// is recovered by looking at the argv instead: the first token identifies the
/// three direct cases, and an `npx` invocation is identified by its package.
pub fn id_of(argv: &[String]) -> Option<&'static str> {
    let head = argv.first()?.as_str();
    // Compare on the file name so an absolute path still matches.
    let head = head.rsplit(['/', '\\']).next().unwrap_or(head);
    let head = head.strip_suffix(".exe").unwrap_or(head);
    for k in KINDS {
        if head == k.cli || Some(head) == k.adapter_bin {
            return Some(k.id);
        }
    }
    if head == "npx" {
        // `?` here would abort the whole function on the first adapter-less kind.
        for k in KINDS {
            if let Some(pkg) = k.adapter_pkg {
                if argv.iter().any(|a| a.starts_with(pkg)) {
                    return Some(k.id);
                }
            }
        }
    }
    None
}

/// A repair hint for a launch failure we recognise.
///
/// Detection can only prove *necessary* conditions — dsh's `acp` profile can exist
/// and the launch still die. This turns the one failure that actually happened
/// into an instruction instead of a stack trace.
///
/// The dsh case is worth spelling out because it is a trap anyone will fall into:
/// `@deepseek-ai/dsh` and `@deepseek-ai/dsh-acp-app` do **not** share a `latest`
/// tag. `npm i -g @deepseek-ai/dsh` installs 0.1.1-rc.2 while `dsh plugin add`
/// pulls acp-app 0.1.2-*, and the newer app then asks the older launcher for
/// hooks (`ctx.appExit` / `ctx.appReady`) it does not have. Aligning both on the
/// 0.1.2 line fixes it.
pub fn repair_hint(id: &str, err: &str) -> Option<&'static str> {
    match id {
        "deepseek"
            if err.contains("appExit")
                || err.contains("appReady")
                || err.contains("plugin tree failed to load") =>
        {
            Some("dsh 与 acp-app 版本错配（两个包的 latest 不同线）。对齐后重试: npm i -g @deepseek-ai/dsh@next")
        }
        "deepseek" if err.contains("does not exist") => {
            Some("acp profile 没配好: dsh plugin --profile acp add @deepseek-ai/dsh-acp-app（需要 pnpm）")
        }
        _ => None,
    }
}

/// Command that installs the *adapter* globally, if this agent needs one.
/// Returned for display; the caller decides whether to run it.
pub fn install_argv(k: &Kind) -> Option<Vec<String>> {
    if let Some(pf) = &k.profile {
        return Some(pf.install.iter().map(|s| s.to_string()).collect());
    }
    k.adapter_pkg.map(|pkg| {
        vec![
            "npm".to_string(),
            "install".to_string(),
            "-g".to_string(),
            pkg.to_string(),
        ]
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pretend exactly these binaries are on PATH.
    fn path<'a>(bins: &'a [&'a str]) -> impl Fn(&str) -> bool + 'a {
        move |b| bins.contains(&b)
    }

    #[test]
    fn every_kind_is_addressable_and_documented() {
        for k in KINDS {
            assert!(find(k.id).is_some(), "{} not findable", k.id);
            assert!(!k.install.hint().is_empty(), "{} has no install hint", k.id);
            // An adapter binary without its package would be undiagnosable.
            assert_eq!(
                k.adapter_bin.is_some(),
                k.adapter_pkg.is_some(),
                "{} adapter is half-specified",
                k.id
            );
        }
    }

    #[test]
    fn missing_cli_beats_a_present_adapter() {
        let k = find("claude").unwrap();
        // The adapter alone cannot talk to a CLI that isn't there.
        assert_eq!(
            state_with(k, path(&["claude-agent-acp", "npx"]), |_| true),
            State::NeedsCli
        );
    }

    #[test]
    fn installed_adapter_is_launched_directly() {
        let k = find("claude").unwrap();
        let p = path(&["claude", "claude-agent-acp", "npx"]);
        assert_eq!(state_with(k, &p, |_| true), State::Ready);
        assert_eq!(argv_with(k, "full", &p), vec!["claude-agent-acp"]);
    }

    #[test]
    fn npx_is_the_fallback_and_is_marked_as_slower() {
        let k = find("codex").unwrap();
        let p = path(&["codex", "npx"]);
        assert_eq!(state_with(k, &p, |_| true), State::ViaNpx);
        assert_eq!(
            argv_with(k, "safe", &p),
            vec!["npx", "-y", "@agentclientprotocol/codex-acp"]
        );
    }

    #[test]
    fn without_npx_the_adapter_must_be_installed() {
        let k = find("codex").unwrap();
        assert_eq!(state_with(k, path(&["codex"]), |_| true), State::NeedsAdapter);
        assert_eq!(
            install_argv(k).unwrap(),
            vec!["npm", "install", "-g", "@agentclientprotocol/codex-acp"]
        );
    }

    /// Agents that speak ACP themselves must never be routed through npx — even
    /// dsh, whose ACP lives in a profile of its own CLI.
    #[test]
    fn native_acp_agents_need_no_adapter() {
        for id in ["kiro", "deepseek"] {
            let k = find(id).unwrap();
            assert!(k.adapter_pkg.is_none(), "{id} should need no npm adapter");
            let argv = argv_with(k, "safe", path(&[k.cli]));
            assert_eq!(argv[0], k.cli);
            assert!(!argv.contains(&"npx".to_string()), "{id}: {argv:?}");
        }
    }

    #[test]
    fn kiro_permission_mode_maps_to_the_auto_approve_flag() {
        let k = find("kiro").unwrap();
        let p = path(&["kiro-cli"]);
        assert!(argv_with(k, "full", &p).contains(&"-a".to_string()));
        for mode in ["safe", "readonly"] {
            assert!(!argv_with(k, mode, &p).contains(&"-a".to_string()), "{mode}");
        }
    }

    #[test]
    fn deepseek_is_launched_through_its_acp_profile() {
        let k = find("deepseek").unwrap();
        assert_eq!(
            argv_with(k, "safe", path(&["dsh"])),
            vec!["dsh", "--profile", "acp"]
        );
    }

    fn v(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|s| s.to_string()).collect()
    }

    /// Every argv this module produces must be recognisable again, otherwise the
    /// settings UI could not show which agent is currently selected.
    #[test]
    fn generated_argv_round_trips_through_id_of() {
        for k in KINDS {
            let direct = argv_with(k, "full", path(&[k.cli, k.adapter_bin.unwrap_or(k.cli)]));
            assert_eq!(id_of(&direct), Some(k.id), "{}: {direct:?}", k.id);
            let via_npx = argv_with(k, "full", path(&[k.cli, "npx"]));
            assert_eq!(id_of(&via_npx), Some(k.id), "{}: {via_npx:?}", k.id);
        }
    }

    #[test]
    fn id_of_tolerates_paths_exe_and_versions() {
        assert_eq!(id_of(&v(&["/Users/x/.local/bin/kiro-cli", "acp"])), Some("kiro"));
        assert_eq!(id_of(&v(&["C:\\tools\\codex-acp.exe"])), Some("codex"));
        assert_eq!(
            id_of(&v(&["npx", "-y", "@agentclientprotocol/claude-agent-acp@latest"])),
            Some("claude")
        );
    }

    /// A CLI on PATH is not enough when ACP lives in a profile: dsh dies with
    /// `profile "acp" does not exist` until the bundle is added, which is exactly
    /// what this used to report as 可用.
    #[test]
    fn a_profile_agent_needs_its_profile_provisioned() {
        let k = find("deepseek").unwrap();
        let p = path(&["dsh"]);
        assert_eq!(state_with(k, &p, |_| false), State::NeedsAdapter);
        assert_eq!(state_with(k, &p, |_| true), State::Ready);
        let cmd = install_argv(k).unwrap();
        assert_eq!(cmd[0], "dsh");
        assert!(cmd.contains(&"@deepseek-ai/dsh-acp-app".to_string()), "{cmd:?}");
    }

    /// The dsh version-skew trap: the two packages' `latest` tags are not on the
    /// same line, so the newer app asks the older launcher for hooks it lacks.
    /// A stack trace is useless here; the fix is one command.
    #[test]
    fn a_recognised_failure_becomes_an_instruction() {
        let skew = "failed to apply loader entry acp-app-startup: stdio app: the \
                    launcher must provide ctx.appExit and ctx.appReady";
        let hint = repair_hint("deepseek", skew).expect("skew should be recognised");
        assert!(hint.contains("dsh@next"), "{hint}");

        let missing = repair_hint("deepseek", "profile \"acp\" does not exist").unwrap();
        assert!(missing.contains("dsh plugin"), "{missing}");

        // Nothing invented for failures we do not understand.
        assert_eq!(repair_hint("deepseek", "some unrelated explosion"), None);
        assert_eq!(repair_hint("kiro", skew), None);
    }

    #[test]
    fn an_unknown_command_stays_unknown() {
        assert_eq!(id_of(&v(&["my-agent", "acp"])), None);
        assert_eq!(id_of(&[]), None);
    }
}
