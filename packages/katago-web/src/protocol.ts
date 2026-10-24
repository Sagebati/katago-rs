// Types of KataGo's JSON analysis protocol (upstream v1.16.5, docs/Analysis_Engine.md and
// cpp/command/analysis.cpp + cpp/search/searchresults.cpp), which the engine speaks verbatim.
// Hand-written; field names are upstream's. Consumers should tolerate fields added in the future.

export type Player = "B" | "W";

/** A stone or a move: `["B", "Q16"]`, `["W", "pass"]`, or `["B", "(0,13)"]`. */
export type ColoredMove = [Player, string];

/** Full rules object (`kata-get-rules` format); a shorthand string such as "japanese" works too. */
export interface RulesObject {
  ko?: "SIMPLE" | "POSITIONAL" | "SITUATIONAL";
  scoring?: "AREA" | "TERRITORY";
  tax?: "NONE" | "SEKI" | "ALL";
  suicide?: boolean;
  hasButton?: boolean;
  whiteHandicapBonus?: "0" | "N" | "N-1";
  friendlyPassOk?: boolean;
}

export interface MoveRestriction {
  player: Player;
  moves: string[];
  untilDepth: number;
}

/** A query (one line of input). `id` is optional here: KataGoClient fills it in. */
export interface AnalysisQuery {
  id?: string;
  moves: ColoredMove[];
  initialStones?: ColoredMove[];
  initialPlayer?: Player;
  rules: string | RulesObject;
  komi?: number;
  whiteHandicapBonus?: "0" | "N" | "N-1";
  boardXSize: number;
  boardYSize: number;
  /** Turns to analyze (0 = before the first move). Default: only the last turn. */
  analyzeTurns?: number[];
  maxVisits?: number;
  rootPolicyTemperature?: number;
  rootFpuReductionMax?: number;
  analysisPVLen?: number;
  includeOwnership?: boolean;
  includeOwnershipStdev?: boolean;
  includeMovesOwnership?: boolean;
  includeMovesOwnershipStdev?: boolean;
  includePolicy?: boolean;
  includePVVisits?: boolean;
  includeNoResultValue?: boolean;
  avoidMoves?: MoveRestriction[];
  allowMoves?: MoveRestriction[];
  /** Search parameters overriding the config for this query (`maxVisits`, `reportAnalysisWinratesAs`...). */
  overrideSettings?: Record<string, string | number | boolean>;
  /** Report partial results (`isDuringSearch: true`) every this many seconds. */
  reportDuringSearchEvery?: number;
  firstReportDuringSearchAfter?: number;
  priority?: number;
  priorities?: number[];
}

/** One candidate move of a TurnResponse. Values are from `reportAnalysisWinratesAs`'s perspective. */
export interface MoveInfo {
  move: string;
  visits: number;
  edgeVisits: number;
  winrate: number;
  scoreMean: number;
  scoreStdev: number;
  scoreLead: number;
  scoreSelfplay: number;
  prior: number;
  utility: number;
  lcb: number;
  utilityLcb: number;
  weight: number;
  edgeWeight: number;
  order: number;
  playSelectionValue: number;
  pv: string[];
  pvVisits?: number[];
  pvEdgeVisits?: number[];
  noResultValue?: number;
  humanPrior?: number;
  isSymmetryOf?: string;
  ownership?: number[];
  ownershipStdev?: number[];
}

export interface RootInfo {
  winrate: number;
  scoreLead: number;
  scoreSelfplay: number;
  scoreStdev: number;
  utility: number;
  visits: number;
  weight: number;
  lcb?: number;
  thisHash: string;
  symHash: string;
  currentPlayer: Player;
  rawWinrate?: number;
  rawLead?: number;
  rawScoreSelfplay?: number;
  rawScoreSelfplayStdev?: number;
  rawNoResultProb?: number;
  rawStWrError?: number;
  rawStScoreError?: number;
  rawVarTimeLeft?: number;
  humanWinrate?: number;
  humanScoreMean?: number;
  humanScoreStdev?: number;
  humanStWrError?: number;
  humanStScoreError?: number;
}

/** The analysis of one turn: partial (`isDuringSearch: true`) or final. */
export interface TurnResponse {
  id: string;
  isDuringSearch: boolean;
  turnNumber: number;
  moveInfos: MoveInfo[];
  rootInfo: RootInfo;
  /** `boardYSize * boardXSize`, row-major from the top-left (A19), in [-1, 1]. */
  ownership?: number[];
  ownershipStdev?: number[];
  /** `boardYSize * boardXSize + 1` (pass last); -1 = illegal. */
  policy?: number[];
  humanPolicy?: number[];
  /** Only on a terminated query that never got analyzed (then moveInfos / rootInfo are absent). */
  noResults?: true;
}

/** `{"error": ...}`, with `field` and `id` when it is about a field of a query. */
export interface ErrorResponse {
  error: string;
  field?: string;
  id?: string;
}

/** A query field the engine ignored or corrected; the query still runs. */
export interface WarningResponse {
  warning: string;
  field: string;
  id: string;
}

export interface VersionResponse {
  id: string;
  action: "query_version";
  version: string;
  git_hash: string;
}

export interface ModelsResponse {
  id: string;
  action: "query_models";
  models: {
    name: string;
    internalName: string;
    maxBatchSize: number;
    usesHumanSLProfile: boolean;
    version: number;
    usingFP16: string;
  }[];
}

/** Any line the engine writes. */
export type ResponseLine = TurnResponse | ErrorResponse | WarningResponse | VersionResponse | ModelsResponse | Record<string, unknown>;
