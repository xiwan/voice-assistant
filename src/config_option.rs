//! Runtime model / reasoning-effort selection, as the agent itself advertises it.
//!
//! Both backends expose their choices over ACP at `session/new` time, but with
//! different shapes (verified 2026-09-04 against kiro-cli 2.21.0 and dsh's acp
//! profile — see `scratch/try_kiro_models.py`, `scratch/try_dsh_config.py`):
//!
//! - **kiro** returns the ACP-standard `models` object:
//!   `{ currentModelId, availableModels: [{ modelId, name, description }] }`.
//!   Changed with `session/set_model { sessionId, modelId }`. No reasoning-effort
//!   option (kiro takes `--effort` at launch instead, which is out of scope here).
//! - **dsh** returns a `configOptions` array, each
//!   `{ id, name, category, type:"select", currentValue, options:[…] }`, and it
//!   carries *both* a `model` option (nested provider groups, values that are JSON
//!   strings like `["deepseek-official","deepseek-v4-pro"]`) and a
//!   `reasoning_effort` option (off/low/high/max). Changed with
//!   `session/set_config_option`.
//!
//! This module flattens both into one `ConfigOption` list so the front end is
//! data-driven: whatever the agent advertised, the panel renders. An agent that
//! advertises nothing yields an empty list and no controls, which is the honest
//! result for a backend that does not let you choose.

use serde_json::Value;

/// How the chosen value is sent back to the agent. The two backends use different
/// ACP methods, so each option remembers which one produced it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SetVia {
    /// kiro: `session/set_model { sessionId, modelId }`.
    Model,
    /// dsh: `session/set_config_option { sessionId, optionId, value }`.
    ConfigOption,
}

/// One selectable value within an option.
#[derive(Clone, Debug, PartialEq)]
pub struct Choice {
    /// The exact string the agent expects back. Opaque on purpose: dsh's model
    /// values are JSON arrays serialized to a string, and we must round-trip them
    /// verbatim rather than interpret them.
    pub value: String,
    pub label: String,
    pub note: String,
}

/// One thing the user can choose (a model, a reasoning-effort level, …), as the
/// agent advertised it.
#[derive(Clone, Debug, PartialEq)]
pub struct ConfigOption {
    /// Stable id used when setting the value (`"model"`, `"reasoning_effort"`).
    pub id: String,
    /// Human label for the group ("模型", "Reasoning effort").
    pub label: String,
    /// The value currently in effect.
    pub current: String,
    pub choices: Vec<Choice>,
    set_via: SetVia,
}

impl ConfigOption {
    pub fn set_via(&self) -> SetVia {
        self.set_via
    }

    /// Label of the current value, for a compact display.
    pub fn current_label(&self) -> &str {
        self.choices
            .iter()
            .find(|c| c.value == self.current)
            .map(|c| c.label.as_str())
            .unwrap_or(&self.current)
    }
}

/// Parse whatever `session/new` (or `session/resume`) returned into a flat option
/// list. Accepts both schemas and silently ignores anything it does not
/// recognise — a new backend with neither shape just yields no options.
pub fn parse(session_result: &Value) -> Vec<ConfigOption> {
    let mut out = Vec::new();
    if let Some(models) = session_result.get("models") {
        if let Some(opt) = parse_kiro_models(models) {
            out.push(opt);
        }
    }
    if let Some(arr) = session_result.get("configOptions").and_then(Value::as_array) {
        out.extend(arr.iter().filter_map(parse_dsh_option));
    }
    out
}

/// kiro's ACP-standard `models` object -> one "model" option.
fn parse_kiro_models(models: &Value) -> Option<ConfigOption> {
    let list = models.get("availableModels")?.as_array()?;
    let choices: Vec<Choice> = list
        .iter()
        .filter_map(|m| {
            let value = m.get("modelId")?.as_str()?.to_string();
            let label = m.get("name").and_then(Value::as_str).unwrap_or(&value).to_string();
            let note = m.get("description").and_then(Value::as_str).unwrap_or("").to_string();
            Some(Choice { value, label, note })
        })
        .collect();
    if choices.is_empty() {
        return None;
    }
    let current = models
        .get("currentModelId")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    Some(ConfigOption {
        id: "model".into(),
        label: "模型".into(),
        current,
        choices,
        set_via: SetVia::Model,
    })
}

/// One entry of dsh's `configOptions` array. Only `select` types are handled;
/// the values (including model values, which are serialized JSON arrays) are kept
/// verbatim for round-tripping.
fn parse_dsh_option(opt: &Value) -> Option<ConfigOption> {
    if opt.get("type").and_then(Value::as_str) != Some("select") {
        return None;
    }
    let id = opt.get("id")?.as_str()?.to_string();
    let label = opt.get("name").and_then(Value::as_str).unwrap_or(&id).to_string();
    let current = opt.get("currentValue").and_then(Value::as_str).unwrap_or("").to_string();
    let choices = flatten_choices(opt.get("options")?);
    if choices.is_empty() {
        return None;
    }
    Some(ConfigOption {
        id,
        label,
        current,
        choices,
        set_via: SetVia::ConfigOption,
    })
}

/// dsh options can be flat (`reasoning_effort`) or grouped by provider
/// (`model`). Both collapse to a flat choice list; a group's own `name` is
/// prefixed onto its children so "DeepSeek-V4-Pro" stays attributable.
fn flatten_choices(options: &Value) -> Vec<Choice> {
    let Some(arr) = options.as_array() else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for entry in arr {
        // A leaf has a `value`; a group has nested `options`.
        if let Some(value) = entry.get("value").and_then(Value::as_str) {
            out.push(Choice {
                value: value.to_string(),
                label: entry.get("name").and_then(Value::as_str).unwrap_or(value).to_string(),
                note: entry.get("description").and_then(Value::as_str).unwrap_or("").to_string(),
            });
        } else if let Some(nested) = entry.get("options").and_then(Value::as_array) {
            for leaf in nested {
                let Some(value) = leaf.get("value").and_then(Value::as_str) else { continue };
                out.push(Choice {
                    value: value.to_string(),
                    label: leaf.get("name").and_then(Value::as_str).unwrap_or(value).to_string(),
                    note: leaf.get("description").and_then(Value::as_str).unwrap_or("").to_string(),
                });
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn kiro_models_flatten_to_one_model_option() {
        let res = json!({
            "sessionId": "x",
            "models": {
                "currentModelId": "claude-opus-5",
                "availableModels": [
                    { "modelId": "auto", "name": "auto", "description": "chosen by task" },
                    { "modelId": "claude-opus-5", "name": "claude-opus-5", "description": "Opus 5" }
                ]
            }
        });
        let opts = parse(&res);
        assert_eq!(opts.len(), 1);
        let m = &opts[0];
        assert_eq!(m.id, "model");
        assert_eq!(m.set_via(), SetVia::Model);
        assert_eq!(m.current, "claude-opus-5");
        assert_eq!(m.current_label(), "claude-opus-5");
        assert_eq!(m.choices.len(), 2);
        assert_eq!(m.choices[0].value, "auto");
    }

    #[test]
    fn dsh_config_options_carry_model_and_reasoning_effort() {
        let res = json!({
            "sessionId": "x",
            "configOptions": [
                {
                    "id": "model", "name": "Model", "type": "select",
                    "currentValue": "[\"deepseek-official\",\"deepseek-v4-flash\"]",
                    "options": [{
                        "group": "deepseek-official", "name": "DeepSeek",
                        "options": [
                            { "value": "[\"deepseek-official\",\"deepseek-v4-flash\"]", "name": "Flash" },
                            { "value": "[\"deepseek-official\",\"deepseek-v4-pro\"]", "name": "Pro" }
                        ]
                    }]
                },
                {
                    "id": "reasoning_effort", "name": "Reasoning effort", "type": "select",
                    "currentValue": "high",
                    "options": [
                        { "value": "off", "name": "Off" },
                        { "value": "high", "name": "High" }
                    ]
                }
            ]
        });
        let opts = parse(&res);
        assert_eq!(opts.len(), 2);

        let model = &opts[0];
        assert_eq!(model.id, "model");
        assert_eq!(model.set_via(), SetVia::ConfigOption);
        assert_eq!(model.choices.len(), 2, "nested provider group flattened");
        // The opaque JSON-string value must survive verbatim.
        assert_eq!(model.choices[1].value, "[\"deepseek-official\",\"deepseek-v4-pro\"]");
        assert_eq!(model.choices[1].label, "Pro");
        assert_eq!(model.current_label(), "Flash");

        let effort = &opts[1];
        assert_eq!(effort.id, "reasoning_effort");
        assert_eq!(effort.current, "high");
        assert_eq!(effort.choices.len(), 2);
    }

    #[test]
    fn a_backend_that_advertises_nothing_yields_no_options() {
        assert!(parse(&json!({ "sessionId": "x" })).is_empty());
    }

    #[test]
    fn non_select_and_empty_options_are_ignored() {
        let res = json!({
            "configOptions": [
                { "id": "x", "type": "text", "currentValue": "y", "options": [] },
                { "id": "z", "type": "select", "currentValue": "", "options": [] }
            ]
        });
        assert!(parse(&res).is_empty());
    }
}
