//! Bridge to local agents. For now: kiro-cli in non-interactive mode.
//! Designed so other agent backends can be added behind the same function.

use anyhow::{anyhow, Result};
use std::process::Command;

/// Send `text` to `kiro-cli chat --no-interactive` and stream its output
/// directly to the terminal. `extra_args` lets callers pass e.g.
/// ["--agent", "myagent"] or ["--trust-tools="].
pub fn ask_kiro(text: &str, extra_args: &[String]) -> Result<()> {
    let mut cmd = Command::new("kiro-cli");
    cmd.arg("chat").arg("--no-interactive");
    for arg in extra_args {
        cmd.arg(arg);
    }
    cmd.arg(text);
    // Stream stdout/stderr to the terminal, but detach stdin so the
    // subprocess doesn't swallow keystrokes meant for the voice assistant.
    cmd.stdin(std::process::Stdio::null());
    let status = cmd
        .status()
        .map_err(|e| anyhow!("failed to run kiro-cli (is it on PATH?): {e}"))?;
    if !status.success() {
        eprintln!("[agent] kiro-cli exited with {status}");
    }
    Ok(())
}
