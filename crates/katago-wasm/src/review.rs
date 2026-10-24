//! Full-game review in the browser (M6): katago-annotate's flow (katago-ws's `AnalysisEngine::analyze`: one
//! pass, or screen + refine) with this engine as the [`Analyzer`]. Each query goes into the engine's input
//! like any other line; its response lines are taken out of the output by [`Router`] (they never reach the
//! worker's line callback) and read the way katago-ws reads them (`ResponseLine`). Not a port of upstream code.

use std::cell::Cell;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use futures::StreamExt;
use futures::channel::mpsc;
use js_sys::Function;
use js_sys::Reflect;
use wasm_bindgen::prelude::*;

use katago_annotate::AnalysisQuery;
use katago_annotate::AnnotateError;
use katago_annotate::AnnotateOptions;
use katago_annotate::AnnotateResult;
use katago_annotate::Analyzer;
use katago_annotate::GameInput;
use katago_annotate::Pass;
use katago_annotate::ResponseLine;
use katago_annotate::TurnResponse;

/// Ids of the review queries (and of their terminate actions): `__kgr_<review>_<n>`.
pub(crate) const REVIEW_ID_PREFIX: &str = "__kgr_";

/// Response lines of the review queries in flight, by query id.
#[derive(Clone, Default)]
pub(crate) struct Router {
  routes: Rc<RefCell<HashMap<String, mpsc::UnboundedSender<String>>>>,
}

impl Router {
  /// Takes `line` when it answers a review query (true), else leaves it to the caller (false).
  pub(crate) fn route(&self, line: &str) -> bool {
    if self.routes.borrow().is_empty() || !line.contains(REVIEW_ID_PREFIX) {
      return false;
    }
    let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else { return false };
    let Some(id) = value.get("id").and_then(|v| v.as_str()) else { return false };
    match self.routes.borrow().get(id) {
      Some(tx) => {
        let _ = tx.unbounded_send(line.to_owned());
        true
      }
      None => false,
    }
  }

  fn add(&self, id: &str) -> Route {
    let (tx, rx) = mpsc::unbounded();
    self.routes.borrow_mut().insert(id.to_owned(), tx);
    Route { router: self.clone(), id: id.to_owned(), rx }
  }
}

/// One query's responses; unregistered when dropped.
struct Route {
  router: Router,
  id: String,
  rx: mpsc::UnboundedReceiver<String>,
}

impl Drop for Route {
  fn drop(&mut self) {
    self.router.routes.borrow_mut().remove(&self.id);
  }
}

/// A review in flight: what `cancelReview` needs.
#[derive(Default)]
pub(crate) struct ReviewState {
  pub(crate) cancelled: Cell<bool>,
  /// The id of the query running now.
  active: RefCell<Option<String>>,
}

/// The engine's input, shared with [`crate::web::Engine`].
pub(crate) type Input = mpsc::UnboundedSender<Vec<Vec<u8>>>;

pub(crate) struct WasmAnalyzer {
  pub(crate) input: Input,
  pub(crate) router: Router,
  pub(crate) review_id: String,
  pub(crate) state: Rc<ReviewState>,
  pub(crate) on_progress: Function,
  pub(crate) seq: Cell<u32>,
  /// Turns analysed so far, over every pass.
  pub(crate) analysed: Cell<u32>,
  pub(crate) two_pass: bool,
}

/// The rejection of a cancelled review (`reviewGame`'s promise rejects with exactly this text).
pub(crate) const CANCELLED: &str = "review cancelled";

fn cancelled() -> AnnotateError {
  AnnotateError::Inference(CANCELLED.to_owned())
}

impl WasmAnalyzer {
  fn progress(&self, pass: Pass, turn: &TurnResponse, done: usize, total: usize) {
    let o = js_sys::Object::new();
    let set = |k: &str, v: JsValue| {
      let _ = Reflect::set(&o, &JsValue::from_str(k), &v);
    };
    set("pass", JsValue::from_str(pass.as_str()));
    set("passes", JsValue::from_f64(if self.two_pass { 2.0 } else { 1.0 }));
    set("turnNumber", JsValue::from_f64(turn.turn_number as f64));
    set("done", JsValue::from_f64(done as f64));
    set("total", JsValue::from_f64(total as f64));
    set("analysed", JsValue::from_f64(self.analysed.get() as f64));
    let _ = self.on_progress.call1(&JsValue::NULL, &o);
  }
}

impl Analyzer for WasmAnalyzer {
  async fn analyze(&self, pass: Pass, mut query: AnalysisQuery) -> AnnotateResult<Vec<TurnResponse>> {
    if self.state.cancelled.get() {
      return Err(cancelled());
    }
    let n = self.seq.get();
    self.seq.set(n + 1);
    let id = format!("{REVIEW_ID_PREFIX}{}_{n}", self.review_id);
    query.id = id.clone();
    // katago-ws expects one response per requested turn (KataGo::analyze, handle_line).
    let expected = query.analyze_turns.len();
    let line = serde_json::to_string(&query).map_err(|e| AnnotateError::Inference(format!("serialize query: {e}")))?;
    let mut route = self.router.add(&id);
    self.input.unbounded_send(vec![line.into_bytes()]).map_err(|_| AnnotateError::Inference("the engine has exited".to_owned()))?;
    *self.state.active.borrow_mut() = Some(id.clone());
    let mut got: Vec<TurnResponse> = Vec::with_capacity(expected);
    let result = loop {
      if got.len() >= expected {
        break Ok(());
      }
      let Some(line) = route.rx.next().await else { break Err(AnnotateError::Inference("the engine has exited".to_owned())) };
      match ResponseLine::parse(&line) {
        ResponseLine::Turn { turn, .. } => {
          // No reportDuringSearchEvery in these queries: every turn line is a final.
          self.analysed.set(self.analysed.get() + 1);
          got.push(*turn);
          // After a cancel the terminated turns still report (partial or noResults): not progress.
          if !self.state.cancelled.get() {
            self.progress(pass, got.last().expect("just pushed"), got.len(), expected);
          }
        }
        ResponseLine::Error { error, .. } => break Err(AnnotateError::Inference(format!("katago: {error}"))),
        ResponseLine::Warning { warning, .. } => log::warn!("katago-wasm review: warning: {warning}"),
        ResponseLine::Unparsable(e) => log::warn!("katago-wasm review: unparsable line: {e}"),
        ResponseLine::UnparsableResponse { error, .. } => log::warn!("katago-wasm review: unparsable response: {error}"),
        ResponseLine::NoId => {}
      }
    };
    *self.state.active.borrow_mut() = None;
    drop(route);
    result?;
    if self.state.cancelled.get() {
      return Err(cancelled());
    }
    Ok(got)
  }
}

/// Stops a review: its query in flight is terminated (the engine then reports what it has for every turn,
/// which is discarded) and no further pass starts; the review's promise rejects with "review cancelled".
pub(crate) fn cancel(state: &ReviewState, review_id: &str, input: Option<&Input>) {
  if state.cancelled.replace(true) {
    return;
  }
  if let (Some(active), Some(input)) = (state.active.borrow().as_ref(), input) {
    let action = serde_json::json!({ "id": format!("{REVIEW_ID_PREFIX}{review_id}_cancel"), "action": "terminate", "terminateId": active });
    let _ = input.unbounded_send(vec![action.to_string().into_bytes()]);
  }
}

/// `reviewGame`'s game: an SGF string or a [`GameInput`] object (as JSON text).
pub(crate) enum ReviewInput {
  Sgf(String),
  Moves(GameInput),
}

/// Reads `reviewGame`'s arguments: the game (SGF text, or an object of KataGo query fields) and the options
/// (AnnotateOptions in camelCase, plus `humanProfile`). Returns the options with the engine's identity set.
pub(crate) fn read_args(input: &JsValue, options: &JsValue, network: &str) -> Result<(ReviewInput, AnnotateOptions, Option<String>), JsError> {
  let input = if let Some(sgf) = input.as_string() {
    ReviewInput::Sgf(sgf)
  } else if input.is_object() {
    let text: String = js_sys::JSON::stringify(input).map_err(|_| JsError::new("reviewGame: the game is not JSON-serializable"))?.into();
    ReviewInput::Moves(serde_json::from_str(&text).map_err(|e| JsError::new(&format!("reviewGame: bad game: {e}")))?)
  } else {
    return Err(JsError::new("reviewGame: the game must be SGF text or {moves, rules, komi, ...}"));
  };
  let mut opts_json = if options.is_undefined() || options.is_null() {
    serde_json::Value::Object(Default::default())
  } else {
    let text: String = js_sys::JSON::stringify(options).map_err(|_| JsError::new("reviewGame: options are not JSON-serializable"))?.into();
    serde_json::from_str(&text).map_err(|e| JsError::new(&format!("reviewGame: bad options: {e}")))?
  };
  let obj = opts_json.as_object_mut().ok_or_else(|| JsError::new("reviewGame: options must be an object"))?;
  let human_profile = match obj.remove("humanProfile") {
    None | Some(serde_json::Value::Null) => None,
    Some(serde_json::Value::String(s)) => Some(s),
    Some(_) => return Err(JsError::new("reviewGame: humanProfile must be a string")),
  };
  for key in ["model", "humanModel", "engineName", "engineVersion"] {
    if obj.contains_key(key) {
      return Err(JsError::new(&format!("reviewGame: option {key} is set by the engine")));
    }
  }
  let mut opts: AnnotateOptions = serde_json::from_value(opts_json).map_err(|e| JsError::new(&format!("reviewGame: bad options: {e}")))?;
  // What produced the payload (docs/M6-report.md): this engine, the KataGo version it follows, its network.
  // No human SL model in the browser: a humanProfile is dropped with a warning, as katago-ws does.
  opts.engine_name = "katago-rs-wasm".to_owned();
  opts.engine_version = Some(katago_core::command::analysis::KATAGO_VERSION.to_owned());
  opts.model = network.to_owned();
  opts.human_model = String::new();
  Ok((input, opts, human_profile))
}
