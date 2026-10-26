// The analysis knobs of katago-ws src/config.rs `EngineConfig` @ bdd674bb9431af2bac8e2e5dfd7d2eb00e5a06f3
// (see docs/M6-report.md): the fields the copied annotate/query/review code reads, with katago-ws's defaults
// and doc comments. Dropped: how katago-ws launches and tunes KataGo (binary, config, timeouts, threads,
// batch sizes, auto-tune). New: `engine_name` / `engine_version`, which katago-ws hard-codes (`"katago"`, none).

use serde::Deserialize;

fn default_model() -> String {
    "model.bin.gz".to_owned()
}
fn default_max_visits() -> u32 {
    100
}
/// Long enough for a whole local sequence (a joseki, a fight) rather than its
/// first few moves; KataGo's own default.
fn default_pv_len() -> u32 {
    15
}
/// The candidates are already searched by the time they're reported, so a
/// larger list costs payload size only, not engine time. Ten rather than five
/// because a consumer checking a proposed move against the engine's opinion can
/// only judge moves that appear here — everything else it has to call unknown.
fn default_top_k() -> usize {
    10
}
fn default_komi() -> f32 {
    7.5
}
fn default_board_size() -> u8 {
    19
}
fn default_refine_winrate_delta() -> f32 {
    0.03
}
fn default_refine_score_delta() -> f32 {
    1.0
}
/// KATAGO-ANNOTATE: katago-ws's `EngineInfo::name` literal.
fn default_engine_name() -> String {
    "katago".to_owned()
}

/// How a game is analysed and annotated: katago-ws's `[engine]` settings that
/// shape the queries and the payload. Read from JSON in camelCase
/// (`maxVisits`, `screenVisits`, ...); every field is optional.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, rename_all = "camelCase", deny_unknown_fields)]
pub struct AnnotateOptions {
    /// KataGo model/weights file (`-model`): recorded as the payload's
    /// `engine.model`.
    pub model: String,
    /// Max visits per analyzed position.
    pub max_visits: u32,
    /// Length of principal variation returned per candidate.
    #[serde(alias = "analysisPVLen")]
    pub analysis_pv_len: u32,
    /// Whether to request ownership maps.
    pub include_ownership: bool,
    /// Whether to request the per-point standard deviation of ownership across
    /// the search tree (`includeOwnershipStdev`): how settled each point is.
    /// No extra search, only payload.
    pub include_ownership_stdev: bool,
    /// Whether to request the raw policy (`includePolicy`): the network's
    /// prior over every move before any search, and — with a human model and a
    /// job's `human_profile` — the human policy. No extra search, only payload.
    pub include_policy: bool,
    /// KataGo human SL network (`-human-model`). Empty (the default) means
    /// none is loaded: a `human_profile` is then dropped with a warning.
    pub human_model: String,
    /// Number of candidate moves to report per position.
    pub top_k: usize,
    /// Komi to assume when an SGF omits it.
    pub default_komi: f32,
    /// Board size to assume when an SGF omits it.
    pub default_board_size: u8,

    // Two-pass analysis. Every turn is first searched with `screen_visits`;
    // then only the turns around moves whose screened evaluation swings by at
    // least a refine delta are re-searched at `max_visits`. Quiet moves (most
    // of a game) keep their cheap screening result. `0` (the default), or a
    // value `>= max_visits`, disables it: one pass at `max_visits`.
    /// Visits per position in the screening pass.
    pub screen_visits: u32,
    /// Refine a move whose screened win-rate swing (either direction) is at
    /// least this. Default: the inaccuracy threshold.
    pub refine_winrate_delta: f32,
    /// Refine a move whose screened score swing (points, either direction) is
    /// at least this. Default: the inaccuracy threshold.
    pub refine_score_delta: f32,

    /// KATAGO-ANNOTATE: the payload's `engine.name` (katago-ws: always
    /// `katago`; the browser: `katago-rs-wasm`).
    pub engine_name: String,
    /// KATAGO-ANNOTATE: the payload's `engine.version`, left out when `None`
    /// (katago-ws never sends one; baduk_review's reader has the field).
    pub engine_version: Option<String>,
}

impl AnnotateOptions {
    /// The options of a JSON object of [`AnnotateOptions`] fields (camelCase) plus an optional `humanProfile`
    /// (a string, or null), and that profile: how the hosts read a review's options.
    ///
    /// # Errors
    ///
    /// Not an object, a `humanProfile` that is not a string, or a field [`AnnotateOptions`] rejects.
    pub fn from_json_with_profile(mut value: serde_json::Value) -> Result<(Self, Option<String>), String> {
        let obj = value.as_object_mut().ok_or("options must be an object")?;
        let human_profile = match obj.remove("humanProfile") {
            None | Some(serde_json::Value::Null) => None,
            Some(serde_json::Value::String(s)) => Some(s),
            Some(_) => return Err("humanProfile must be a string".to_owned()),
        };
        let options = serde_json::from_value(value).map_err(|e| format!("bad options: {e}"))?;
        Ok((options, human_profile))
    }

    /// Whether the two-pass (screen, then refine) analysis is enabled.
    #[must_use]
    pub fn two_pass(&self) -> bool {
        self.screen_visits > 0 && self.screen_visits < self.max_visits
    }
}

impl Default for AnnotateOptions {
    fn default() -> Self {
        Self {
            model: default_model(),
            max_visits: default_max_visits(),
            analysis_pv_len: default_pv_len(),
            // On by default: ownership is what lets a consumer say *which*
            // stones a mistake cost, rather than only how many points. It
            // roughly doubles KataGo's memory use and costs a little speed —
            // turn it off on a memory-constrained worker.
            include_ownership: true,
            include_ownership_stdev: true,
            include_policy: true,
            human_model: String::new(),
            top_k: default_top_k(),
            default_komi: default_komi(),
            default_board_size: default_board_size(),
            screen_visits: 0,
            refine_winrate_delta: default_refine_winrate_delta(),
            refine_score_delta: default_refine_score_delta(),
            engine_name: default_engine_name(),
            engine_version: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // KATAGO-ANNOTATE: from katago-ws config.rs's tests (`EngineConfig` -> `AnnotateOptions`).
    #[test]
    fn two_pass_needs_a_screen_shallower_than_the_full_search() {
        let base = AnnotateOptions::default();
        assert!(!base.two_pass());
        assert!(
            AnnotateOptions {
                screen_visits: 16,
                ..base.clone()
            }
            .two_pass()
        );
        assert!(
            !AnnotateOptions {
                screen_visits: base.max_visits,
                ..base
            }
            .two_pass()
        );
    }

    #[test]
    fn options_read_from_camel_case_json_over_the_defaults() {
        let opts: AnnotateOptions =
            serde_json::from_str(r#"{"maxVisits": 400, "screenVisits": 25, "analysisPVLen": 10}"#)
                .unwrap();
        assert_eq!(opts.max_visits, 400);
        assert_eq!(opts.screen_visits, 25);
        assert_eq!(opts.analysis_pv_len, 10);
        assert_eq!(opts.top_k, 10);
        assert!(opts.two_pass());
        assert!(serde_json::from_str::<AnnotateOptions>(r#"{"maxVisit": 1}"#).is_err());
    }
}
