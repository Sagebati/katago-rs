// Full-game review (M6): the game, the options and the payload v3 that `KataGoClient.reviewGame` returns.
// The payload is katago-ws's (crate katago-annotate, copied from katago-ws src/engine/annotate.rs), the same
// JSON that baduk_review stores for a server-side analysis; field names are its snake_case ones.

import type { ColoredMove } from "./protocol.js";

/** A game given as moves rather than SGF text (KataGo query fields). Coordinates are GTP (`Q16`, `pass`). */
export interface GameInput {
  moves: ColoredMove[];
  /** Square boards only: `boardSize`, or `boardXSize` and `boardYSize` equal. Default 19 (`defaultBoardSize`). */
  boardSize?: number;
  boardXSize?: number;
  boardYSize?: number;
  /** Default 7.5 (`defaultKomi`). */
  komi?: number;
  /** "japanese" (default) or "chinese", as katago-ws maps an SGF's RU. */
  rules?: "japanese" | "chinese";
  /** Handicap / setup stones. */
  initialStones?: ColoredMove[];
  /** The game's result (copied into the payload's `result`). */
  result?: string;
}

/** katago-ws's analysis settings (defaults in brackets). */
export interface ReviewOptions {
  /** Visits per analysed position [100]. */
  maxVisits?: number;
  /** Two-pass review: every turn at this many visits first, then only the turns around moves whose evaluation
   * swings by `refineWinrateDelta` / `refineScoreDelta` at `maxVisits` [0: one pass]. */
  screenVisits?: number;
  refineWinrateDelta?: number;
  refineScoreDelta?: number;
  /** Candidates kept per position [10]. */
  topK?: number;
  /** PV length per candidate [15]. */
  analysisPvLen?: number;
  /** Per-turn grids [all true]. */
  includeOwnership?: boolean;
  includeOwnershipStdev?: boolean;
  includePolicy?: boolean;
  /** For an SGF without SZ / KM [19, 7.5]. */
  defaultBoardSize?: number;
  defaultKomi?: number;
  /** Human SL profile: dropped with a warning (no human SL network in the browser yet). */
  humanProfile?: string;
  /** Cancels the review when aborted (like `ReviewRun.cancel()`). */
  signal?: AbortSignal;
}

/** After each analysed turn. */
export interface ReviewProgress {
  /** "full" (one pass), "screen" or "refine" (two-pass). */
  pass: "full" | "screen" | "refine";
  /** 1 or 2. */
  passes: number;
  turnNumber: number;
  /** Turns done / requested in this pass. */
  done: number;
  total: number;
  /** Turns analysed so far over every pass. */
  analysed: number;
}

export type Classification = "good" | "inaccuracy" | "mistake" | "blunder";

export interface Candidate {
  mv: string;
  /** Mover's perspective. */
  winrate: number;
  score_lead: number;
  prior: number;
  human_prior?: number;
  visits: number;
  pv: string[];
}

export interface MoveAnnotation {
  /** 1-based. */
  move_number: number;
  color: "B" | "W";
  mv: string;
  winrate: number;
  best_winrate: number;
  winrate_loss: number;
  winrate_classification: Classification;
  /** Mover's lead before the move (with best play). */
  score_lead: number;
  score_loss: number;
  score_classification: Classification;
  /** Raw policy of the move played (absent without `includePolicy` when it is not a candidate either). */
  prior?: number;
  human_prior?: number;
  top_moves: Candidate[];
}

export interface ClassCounts {
  good: number;
  inaccuracy: number;
  mistake: number;
  blunder: number;
}

export interface PlayerReport {
  moves: number;
  accuracy: number;
  mean_winrate_loss: number;
  mean_score_loss: number;
  by_winrate: ClassCounts;
  by_score: ClassCounts;
  estimated_rank?: string;
}

export interface EngineInfo {
  /** "katago-rs-wasm" in the browser ("katago" from katago-ws). */
  name: string;
  /** The KataGo version the port follows. */
  version?: string;
  /** The network ("g170e-b10c128", ...). */
  model: string;
  max_visits: number;
  top_k: number;
  report_perspective: string;
  ownership: boolean;
  ownership_stdev: boolean;
  policy: boolean;
  human_model?: string;
  /** Two-pass reviews: the screening depth. */
  screen_visits?: number;
}

/** The analysis payload, version 3 (baduk_review's `katago_sdk::GameAnalysis`). */
export interface GameAnalysis {
  board_size: number;
  komi: number;
  rules: "japanese" | "chinese";
  result: string | null;
  moves: MoveAnnotation[];
  report: { num_moves: number; black: PlayerReport; white: PlayerReport };
  payload_version: number;
  engine: EngineInfo;
  /** Per turn (`moves.length + 1`), row-major from the top-left, Black-positive hundredths. */
  ownership?: number[][];
  /** Per turn, hundredths 0..100. */
  ownership_stdev?: number[][];
  /** Per turn, `round(255 * sqrt(p))`, pass dropped. */
  policy?: number[][];
  human_policy?: number[][];
  human_profile?: string;
}
