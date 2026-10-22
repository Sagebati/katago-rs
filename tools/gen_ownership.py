import sys
A = "Board::MAX_ARR_SIZE as usize"
P = "NNPos::MAX_NN_POLICY_SIZE as usize"
rows = []
def r(file, cls, field, cpp, own, rust, notes=""):
    for v in (file, cls, field, cpp, own, rust, notes):
        assert "\t" not in v and "\n" not in v, v
    assert own in {"OWNED","BORROWED","SHARED","ARENA","NULLABLE_BORROWED","STATIC"}, own
    rows.append((file, cls, field, cpp, own, rust, notes))

# ---------------- game/board.h ----------------
f = "cpp/game/board.h"
r(f,"Board","colors","Color[MAX_ARR_SIZE]","OWNED",f"[Color; {A}]","inline array; index as colors[loc as usize]")
r(f,"Board","chain_data","ChainData[MAX_ARR_SIZE]","OWNED",f"[ChainData; {A}]","ChainData is #[derive(Clone, Copy, Default)]; only valid at chain heads")
r(f,"Board","chain_head","Loc[MAX_ARR_SIZE]","OWNED",f"[Loc; {A}]","Loc indices, not pointers")
r(f,"Board","next_in_chain","Loc[MAX_ARR_SIZE]","OWNED",f"[Loc; {A}]","circular linked list stored as Loc indices")
r(f,"Board","adj_offsets","short[8]","OWNED","[i16; 8]","widen to i32 before adding to a Loc: (loc as i32 + self.adj_offsets[i] as i32) as Loc")
r(f,"Board","IS_ZOBRIST_INITALIZED","static bool","STATIC","(removed)","PORT NOTE: replaced by BOARD_STATICS.get().is_some(); Board::init_hash() forces the OnceLock")
for name, ty, n in [
    ("ZOBRIST_SIZE_X_HASH","static Hash128[MAX_LEN+1]","[Hash128; (Board::MAX_LEN + 1) as usize]"),
    ("ZOBRIST_SIZE_Y_HASH","static Hash128[MAX_LEN+1]","[Hash128; (Board::MAX_LEN + 1) as usize]"),
    ("ZOBRIST_BOARD_HASH","static Hash128[MAX_ARR_SIZE][4]",f"[[Hash128; 4]; {A}]"),
    ("ZOBRIST_BOARD_HASH2","static Hash128[MAX_ARR_SIZE][4]",f"[[Hash128; 4]; {A}]"),
    ("ZOBRIST_PLAYER_HASH","static Hash128[4]","[Hash128; 4]"),
    ("ZOBRIST_KO_LOC_HASH","static Hash128[MAX_ARR_SIZE]",f"[Hash128; {A}]"),
    ("ZOBRIST_KO_MARK_HASH","static Hash128[MAX_ARR_SIZE][4]",f"[[Hash128; 4]; {A}]"),
    ("ZOBRIST_ENCORE_HASH","static Hash128[3]","[Hash128; 3]"),
    ("ZOBRIST_SECOND_ENCORE_START_HASH","static Hash128[MAX_ARR_SIZE][4]",f"[[Hash128; 4]; {A}]"),
]:
    r(f,"Board",name,ty,"STATIC",n,"field of BoardStatics in static BOARD_STATICS: OnceLock<BoardStatics>; read via Board::statics().%s; filled by the ported Board::initHash body in the same RNG call order" % name)
r(f,"Board","ZOBRIST_PASS_ENDS_PHASE","static const Hash128","STATIC","Hash128","pub const on Board: Hash128::new(0x853E097C279EBF4E, 0xE3153DEF9E14A62C)")
r(f,"Board","ZOBRIST_GAME_IS_OVER","static const Hash128","STATIC","Hash128","pub const on Board: Hash128::new(0xb6f9e465597a77ee, 0xf1d583d960a4ce7f)")
r(f,"Board","calculateArea(result)","Color*","BORROWED","&mut [Color]","param; buffer of MAX_ARR_SIZE")
r(f,"Board","calculateIndependentLifeArea(result)","Color*","BORROWED","&mut [Color]","param")
r(f,"Board","calculateIndependentLifeArea(whiteMinusBlackIndependentLifeRegionCount)","int&","BORROWED","&mut i32","out-param")
r(f,"Board","isNonPassAliveSelfConnection(passAliveArea)","const Color*","BORROWED","&[Color]","param")
r(f,"Board","searchIsLadderCaptured(buf)","std::vector<Loc>&","BORROWED","&mut Vec<Loc>","param; method itself is &mut self (mutates chains during search)")
r(f,"Board","searchIsLadderCapturedAttackerFirst2Libs(workingMoves)","std::vector<Loc>&","BORROWED","&mut Vec<Loc>","param")
r(f,"Board","setStonesFailIfNoLibs(placements)","const std::vector<Move>&","BORROWED","&[Move]","param")
r(f,"Board","printBoard(out)","std::ostream&","BORROWED","&mut dyn std::fmt::Write","param")
r(f,"Board","printBoard(hist)","const std::vector<Move>*","NULLABLE_BORROWED","Option<&[Move]>","param")
r(f,"Board","countEmptyHelper(emptyCounted)","bool*","BORROWED","&mut [bool]","param")
r(f,"Location","getAdjacentOffsets(adj_offsets)","short[8]","BORROWED","&mut [i16; 8]","param")

# ---------------- game/boardhistory.h ----------------
f = "cpp/game/boardhistory.h"
r(f,"BoardHistory","initialBoard","Board","OWNED","Board","Board derives Clone (not Copy); every C++ copy is an explicit .clone()")
r(f,"BoardHistory","recentBoards","Board[NUM_RECENT_BOARDS]","OWNED","[Board; BoardHistory::NUM_RECENT_BOARDS as usize]","inline")
r(f,"BoardHistory","wasEverOccupiedOrPlayed","bool[Board::MAX_ARR_SIZE]","OWNED",f"[bool; {A}]","inline")
r(f,"BoardHistory","superKoBanned","bool[Board::MAX_ARR_SIZE]","OWNED",f"[bool; {A}]","inline")
r(f,"BoardHistory","koRecapBlocked","bool[Board::MAX_ARR_SIZE]","OWNED",f"[bool; {A}]","inline")
r(f,"BoardHistory","secondEncoreStartColors","Color[Board::MAX_ARR_SIZE]","OWNED",f"[Color; {A}]","inline")
r(f,"BoardHistory","preventEncoreHistory","std::vector<bool>","OWNED","Vec<bool>","")
r(f,"BoardHistory","koCapturesInEncore","std::vector<EncoreKoCapture>","OWNED","Vec<EncoreKoCapture>","EncoreKoCapture (STRUCT_NAMED_TRIPLE) is #[derive(Clone, Copy, Default)] with new()")
r(f,"BoardHistory","makeBoardMoveAssumeLegal(rootKoHashTable)","const KoHashTable*","NULLABLE_BORROWED","Option<&KoHashTable>","param; Search passes Some(&self.root_ko_hash_table)")
r(f,"BoardHistory","numberOfKoHashOccurrencesInHistory(rootKoHashTable)","const KoHashTable*","NULLABLE_BORROWED","Option<&KoHashTable>","param")
r(f,"BoardHistory","koHashOccursInHistory(rootKoHashTable)","const KoHashTable*","NULLABLE_BORROWED","Option<&KoHashTable>","param")
r(f,"BoardHistory","endAndScoreGameNow(area)","Color[Board::MAX_ARR_SIZE]","BORROWED",f"&mut [Color; {A}]","param; same for getAreaNow, countAreaScoreWhiteMinusBlack, countTerritoryAreaScoreWhiteMinusBlack")
r(f,"BoardHistory","getRecentBoard(return)","const Board&","BORROWED","&Board","return value borrows self")
r(f,"BoardHistory","printBasicInfo(out)","std::ostream&","BORROWED","&mut dyn std::fmt::Write","param; same for printDebugInfo")
r(f,"KoHashTable","idxTable","uint32_t*","OWNED","Box<[u32; KoHashTable::TABLE_SIZE as usize]>","new uint32_t[TABLE_SIZE] in ctor, delete[] in dtor; copy ctor deleted, so no Clone")
r(f,"KoHashTable","koHashHistorySortedByLowBits","std::vector<Hash128>","OWNED","Vec<Hash128>","recompute() sorts with sort_unstable_by (ties are identical hashes)")

# ---------------- game/rules.h ----------------
f = "cpp/game/rules.h"
for name, ty, n in [
    ("ZOBRIST_KO_RULE_HASH","static const Hash128[4]","[Hash128; 4]"),
    ("ZOBRIST_SCORING_RULE_HASH","static const Hash128[2]","[Hash128; 2]"),
    ("ZOBRIST_TAX_RULE_HASH","static const Hash128[3]","[Hash128; 3]"),
    ("ZOBRIST_MULTI_STONE_SUICIDE_HASH","static const Hash128","Hash128"),
    ("ZOBRIST_BUTTON_HASH","static const Hash128","Hash128"),
    ("ZOBRIST_FRIENDLY_PASS_OK_HASH","static const Hash128","Hash128"),
]:
    r(f,"Rules",name,ty,"STATIC",n,"pub const on Rules; literal values copied from rules.cpp")

# ---------------- neuralnet/nninputs.h ----------------
f = "cpp/neuralnet/nninputs.h"
for name in ["ZOBRIST_CONSERVATIVE_PASS","ZOBRIST_FRIENDLY_PASS","ZOBRIST_PASSING_HACKS","ZOBRIST_PLAYOUT_DOUBLINGS","ZOBRIST_NN_POLICY_TEMP","ZOBRIST_AVOID_MYTDAGGER_HACK","ZOBRIST_POLICY_OPTIMISM","ZOBRIST_ZERO_HISTORY"]:
    r(f,"MiscNNInputParams",name,"static const Hash128","STATIC","Hash128","pub const on MiscNNInputParams; literal from nninputs.cpp")
r(f,"NNOutput","policyProbs","float[NNPos::MAX_NN_POLICY_SIZE]","OWNED",f"[f32; {P}]","inline")
r(f,"NNOutput","whiteOwnerMap","float*","OWNED","Option<Vec<f32>>","new float[nnXLen*nnYLen] or NULL; deep-copied by Clone (= copy ctor); freed by Drop")
r(f,"NNOutput","noisedPolicyProbs","float*","OWNED",f"Option<Box<[f32; {P}]>>","new float[MAX_NN_POLICY_SIZE] or NULL; getPolicyProbsMaybeNoised returns &[f32; ..] of this or policy_probs")
r(f,"NNOutput","NNOutput(others)","const std::vector<std::shared_ptr<NNOutput>>&","BORROWED","&[Rc<NNOutput>]","ctor param -> NNOutput::new_others; NNOutput() -> new() zero-initializes (PORT NOTE: C++ leaves fields uninitialized)")
r(f,"NNInputs","fillRowV7(rowBin)","float*","BORROWED","&mut [f32]","param; same shape for fillRowV3..V6 if ever ported")
r(f,"NNInputs","fillRowV7(rowGlobal)","float*","BORROWED","&mut [f32]","param")
r(f,"NNInputs","fillScoring(area)","const Color*","BORROWED","&[Color]","param")
r(f,"NNInputs","fillScoring(scoring)","float*","BORROWED","&mut [f32]","param")
r(f,"SymmetryHelpers","copyInputsWithSymmetry(src)","const float*","BORROWED","&[f32]","param; dst is &mut [f32]; same for copyOutputsWithSymmetry")
r(f,"SymmetryHelpers","markDuplicateMoveLocs(onlySymmetries)","const std::vector<int>*","NULLABLE_BORROWED","Option<&[i32]>","param")
r(f,"SymmetryHelpers","markDuplicateMoveLocs(isSymDupLoc)","bool*","BORROWED","&mut [bool]","param; Search passes &mut self.root_sym_dup_loc")
r(f,"SymmetryHelpers","markDuplicateMoveLocs(validSymmetries)","std::vector<int>&","BORROWED","&mut Vec<i32>","out-param")
r(f,"SymmetryHelpers","getSymmetryDifferences(symmetryDifferences)","double[NUM_SYMMETRIES]","BORROWED","&mut [f64; SymmetryHelpers::NUM_SYMMETRIES as usize]","out-param")
f2 = "cpp/neuralnet/nninputs.cpp"
r(f2,"ScoreValue","expectedSVTable","static double*","STATIC","Vec<f64>","field of ScoreValueStatics in static SCORE_VALUE_STATICS: OnceLock<ScoreValueStatics>; built by the initTables body; PORT NOTE: initTables idempotent, freeTables no-op")
r(f2,"ScoreValue","scoreValueTablesInitialized","static bool","STATIC","(removed)","OnceLock state replaces it")

# ---------------- neuralnet/nneval.h ----------------
f = "cpp/neuralnet/nneval.h"
r(f,"NNCacheTable","entries","Entry*","OWNED","RefCell<Box<[Entry]>>","new Entry[tableSize]; get/set take &self (called during search)")
r(f,"NNCacheTable::Entry","ptr","std::shared_ptr<NNOutput>","SHARED","Option<Rc<NNOutput>>","")
r(f,"NNCacheTable","mutexPool","MutexPool*","OWNED","(removed)","PORT NOTE: lock removed; nothing derived from it")
r(f,"NNEvaluator","numThreads","int","OWNED","Cell<i32>","written by setNumThreads while the evaluator is shared (Rc)")
r(f,"NNEvaluator","gpuIdxByServerThread","std::vector<int>","OWNED","RefCell<Vec<i32>>","rewritten by spawnServerThreads")
r(f,"NNEvaluator","numServerThreadsEverSpawned","int","OWNED","Cell<i32>","incremented by spawned serve tasks")
r(f,"NNCacheTable","mutexPoolMask","uint32_t","OWNED","(removed)","only selected a lock")
r(f,"NNResultBuf","clientWaitingForResult","std::condition_variable","OWNED","WakerList","port::executor::WakerList; client: client_waiting_for_result.wait_until(|| buf.has_result.get()).await; server: set has_result then notify_all() (section 11)")
r(f,"NNResultBuf","resultMutex","std::mutex","OWNED","(removed)","")
r(f,"NNResultBuf","hasResult","bool","OWNED","Cell<bool>","every NNResultBuf field is Cell/RefCell because the buf is shared as Rc<NNResultBuf> with the eval queue")
r(f,"NNResultBuf","includeOwnerMap","bool","OWNED","Cell<bool>","")
r(f,"NNResultBuf","boardXSizeForServer","int","OWNED","Cell<i32>","")
r(f,"NNResultBuf","boardYSizeForServer","int","OWNED","Cell<i32>","")
r(f,"NNResultBuf","rowSpatialBuf","std::vector<float>","OWNED","RefCell<Vec<f32>>","borrow_mut only while filling (evaluate) or reading (backend getOutput); never across await")
r(f,"NNResultBuf","rowGlobalBuf","std::vector<float>","OWNED","RefCell<Vec<f32>>","")
r(f,"NNResultBuf","rowMetaBuf","std::vector<float>","OWNED","RefCell<Vec<f32>>","")
r(f,"NNResultBuf","hasRowMeta","bool","OWNED","Cell<bool>","")
r(f,"NNResultBuf","result","std::shared_ptr<NNOutput>","SHARED","RefCell<Option<Rc<NNOutput>>>","postprocessing in evaluate mutates the fresh, unique result via Rc::get_mut(..).expect(\"unique\")")
r(f,"NNResultBuf","errorLogLockout","bool","OWNED","Cell<bool>","")
r(f,"NNResultBuf","symmetry","int","OWNED","Cell<i32>","")
r(f,"NNResultBuf","policyOptimism","double","OWNED","Cell<f64>","")
r(f,"NNServerBuf","inputBuffers","InputBuffers*","OWNED","Option<Box<dyn InputBuffers>>","created by backend create_input_buffers, NULL when no model; owned by the server task")
r(f,"NNEvaluator","computeContext","ComputeContext*","OWNED","Option<Box<dyn ComputeContext>>","NULL when debugSkipNeuralNet")
r(f,"NNEvaluator","loadedModel","LoadedModel*","OWNED","Option<Box<dyn LoadedModel>>","NULL when debugSkipNeuralNet")
r(f,"NNEvaluator","nnCacheTable","NNCacheTable*","OWNED","Option<Box<NNCacheTable>>","NULL when nnCacheSizePowerOfTwo < 0")
r(f,"NNEvaluator","logger","Logger*","NULLABLE_BORROWED","(removed)","log crate (manifest); ctor param dropped (PORT NOTE)")
r(f,"NNEvaluator","(new) spawner","-","SHARED","Rc<dyn Spawner>","PORT NOTE: port::executor::Spawner injected by the host (new ctor param); used by spawnServerThreads to spawn serve tasks")
r(f,"NNEvaluator","(new) backend","-","SHARED","Rc<dyn NeuralNet>","PORT NOTE: trait in nninterface.rs standing in for the NeuralNet:: free functions (new ctor param); implemented by katago-nn-burn")
r(f,"NNEvaluator","serverThreads","std::vector<std::thread*>","OWNED","(removed)","server threads become tasks spawned via spawner, counted by server_threads: Cell<i32> and num_server_threads_alive: Cell<i32>; killServerThreads sets is_killed + setReadOnly and does not join: the join moves to the next spawnServerThreads (D8, PORT NOTE)")
r(f,"NNEvaluator","(new) serverThreadsExited","-","OWNED","WakerList","PORT NOTE (D8): port::executor::WakerList; each serve task notifies it as it ends (num_server_threads_alive -= 1); spawnServerThreads awaits num_server_threads_alive == 0 on it (the deferred join) before isKilled = false, the post-join testAsserts and unsetReadOnly")
r(f,"NNEvaluator","bufferMutex","mutable std::mutex","OWNED","(removed)","the counters it guarded become Cell")
r(f,"NNEvaluator","isKilled","bool","OWNED","Cell<bool>","")
r(f,"NNEvaluator","numServerThreadsStartingUp","int","OWNED","Cell<i32>","")
r(f,"NNEvaluator","mainThreadWaitingForSpawn","std::condition_variable","OWNED","WakerList","port::executor::WakerList; spawnServerThreads awaits it")
r(f,"NNEvaluator","serverThreadsIsUsingFP16","std::vector<int>","OWNED","RefCell<Vec<i32>>","written by server tasks at startup")
r(f,"NNEvaluator","numOngoingEvals","int","OWNED","Cell<i32>","")
r(f,"NNEvaluator","numWaitingEvals","int","OWNED","Cell<i32>","")
r(f,"NNEvaluator","numEvalsToAwaken","int","OWNED","Cell<i32>","")
r(f,"NNEvaluator","waitingForFinish","std::condition_variable","OWNED","WakerList","port::executor::WakerList; waitForNextNNEvalIfAny awaits it")
r(f,"NNEvaluator","m_numRowsProcessed","std::atomic<uint64_t>","OWNED","Cell<u64>","")
r(f,"NNEvaluator","m_numBatchesProcessed","std::atomic<uint64_t>","OWNED","Cell<u64>","")
r(f,"NNEvaluator","currentDoRandomize","std::atomic<bool>","OWNED","Cell<bool>","")
r(f,"NNEvaluator","currentDefaultSymmetry","std::atomic<int>","OWNED","Cell<i32>","")
r(f,"NNEvaluator","currentBatchSize","std::atomic<int>","OWNED","Cell<i32>","")
r(f,"NNEvaluator","queryQueue","ThreadSafeQueue<NNResultBuf*>","SHARED","AsyncQueue<Rc<NNResultBuf>>","port::executor::AsyncQueue; elements are the SearchThread-owned bufs, shared while an eval is pending")
r(f,"NNEvaluator","evaluate(buf)","NNResultBuf&","SHARED","&Rc<NNResultBuf>","param; evaluate is async and pushes buf.clone() onto query_queue")
r(f,"NNEvaluator","evaluate(sgfMeta)","const SGFMetadata*","NULLABLE_BORROWED","Option<&SGFMetadata>","param")
r(f,"NNEvaluator","averageMultipleSymmetries(return)","std::shared_ptr<NNOutput>*","OWNED","Rc<NNOutput>","caller-owned heap shared_ptr becomes a plain Rc")
r(f,"NNEvaluator","averageMultipleSymmetries(rand)","Rand&","BORROWED","&mut Rand","param")
r(f,"NNEvaluator","serve(buf)","NNServerBuf&","OWNED","NNServerBuf","moved into the spawned serve task, together with its Rand and Box<dyn ComputeHandle>; serve takes self: Rc<Self>")
r(f,"NNEvaluator","serve local outputBuf","std::vector<NNOutput*>","OWNED","Vec<NNOutput>","each element moved into Rc::new(..) when delivered to its NNResultBuf")

# ---------------- neuralnet/nninterface.h ----------------
f = "cpp/neuralnet/nninterface.h"
r(f,"NeuralNet","loadModelFile(return)","LoadedModel*","OWNED","Box<dyn LoadedModel>","TODO(port): human decision: core has no file IO, so the backend loads from bytes (plan §5)")
r(f,"NeuralNet","createComputeContext(return)","ComputeContext*","OWNED","Box<dyn ComputeContext>","freeComputeContext becomes Drop")
r(f,"NeuralNet","createComputeHandle(return)","ComputeHandle*","OWNED","Box<dyn ComputeHandle>","owned by the serve task; freeComputeHandle becomes Drop")
r(f,"NeuralNet","createInputBuffers(return)","InputBuffers*","OWNED","Box<dyn InputBuffers>","freeInputBuffers becomes Drop")
r(f,"NeuralNet","getModelDesc(return)","const ModelDesc&","BORROWED","&ModelDesc","")
r(f,"NeuralNet","getOutput(computeHandle)","ComputeHandle*","BORROWED","&mut dyn ComputeHandle","param; getOutput is async")
r(f,"NeuralNet","getOutput(buffers)","InputBuffers*","BORROWED","&mut dyn InputBuffers","param")
r(f,"NeuralNet","getOutput(inputBufs)","NNResultBuf**","BORROWED","&[Rc<NNResultBuf>]","param")
r(f,"NeuralNet","getOutput(outputs)","std::vector<NNOutput*>&","BORROWED","&mut [NNOutput]","param; the caller pre-builds each NNOutput (nn_x_len, white_owner_map) as upstream does")

# ---------------- neuralnet/sgfmetadata.h ----------------
f = "cpp/neuralnet/sgfmetadata.h"
r(f,"SGFMetadata","fillMetadataRow(sgfMeta)","const SGFMetadata*","NULLABLE_BORROWED","Option<&SGFMetadata>","param; rowMetadata float* -> &mut [f32]")

# ---------------- search/searchparams.h ----------------
f = "cpp/search/searchparams.h"
r(f,"SearchParams","humanSLProfile","SGFMetadata","OWNED","SGFMetadata","passed as &searchParams.humanSLProfile -> Some(&self.search_params.human_sl_profile)")

# ---------------- search/searchnode.h ----------------
f = "cpp/search/searchnode.h"
r(f,"NodeStatsAtomic","visits","std::atomic<int64_t>","OWNED","Cell<i64>","keep memory orders as comments")
for name in ["winLossValueAvg","noResultValueAvg","scoreMeanAvg","scoreMeanSqAvg","leadAvg","utilityAvg","utilitySqAvg","weightSum","weightSqSum"]:
    r(f,"NodeStatsAtomic",name,"std::atomic<double>","OWNED","Cell<f64>","")
r(f,"SearchChildPointer","data","std::atomic<SearchNode*>","ARENA","Cell<Option<NodeIdx>>","get_if_allocated() -> Option<NodeIdx>; store(node.idx); store_if_null uses port::cpp::cell_cas")
r(f,"SearchChildPointer","edgeVisits","std::atomic<int64_t>","OWNED","Cell<i64>","compexweakEdgeVisits -> port::cpp::cell_cas")
r(f,"SearchChildPointer","moveLoc","std::atomic<Loc>","OWNED","Cell<Loc>","")
r(f,"SearchNodeChildrenReference","node","SearchNode*","BORROWED","&'a SearchNode","short-lived view; operator[] -> fn index(&self, i: i32) -> &'a SearchChildPointer")
r(f,"ConstSearchNodeChildrenReference","node","const SearchNode*","BORROWED","&'a SearchNode","PORT NOTE: pub type ConstSearchNodeChildrenReference<'a> = SearchNodeChildrenReference<'a>; its unused capacity field is dropped")
r(f,"SearchNode","(new) idx","-","ARENA","NodeIdx","PORT NOTE: first field, set by NodeArena::alloc; SearchNode::new takes it as first param; node identity (== rootNode, graphPath) uses it")
r(f,"SearchNode","statsLock","mutable std::atomic_flag","OWNED","(removed)","PORT NOTE: spin lock removed, body kept")
r(f,"SearchNode","nextPla","const Player","OWNED","Player","")
r(f,"SearchNode","forceNonTerminal","const bool","OWNED","bool","")
r(f,"SearchNode","patternBonusHash","Hash128","OWNED","Cell<Hash128>","written after construction (allocateOrFindNode, beginSearch)")
r(f,"SearchNode","mutexIdx","const uint32_t","OWNED","u32","still computed by createMutexIdxForNode, which consumes thread RNG")
r(f,"SearchNode","state","std::atomic<SearchNodeState>","OWNED","Cell<SearchNodeState>","compare_exchange_strong -> port::cpp::cell_cas")
r(f,"SearchNode","nnOutput","std::atomic<std::shared_ptr<NNOutput>*>","SHARED","RefCell<Option<Rc<NNOutput>>>","getNNOutput() -> Option<Rc<NNOutput>> (clone); storeNNOutput: replace() and push the old Rc onto thread.old_nn_outputs_to_clean_up; storeNNOutputIfNull: set only if None")
r(f,"SearchNode","humanOutput","std::atomic<std::shared_ptr<NNOutput>*>","SHARED","RefCell<Option<Rc<NNOutput>>>","same as nnOutput")
r(f,"SearchNode","nodeAge","std::atomic<uint32_t>","OWNED","Cell<u32>","")
r(f,"SearchNode","children0","SearchChildPointer*","OWNED","OnceCell<Box<[SearchChildPointer]>>","len SearchChildrenSizes::SIZE0OVERFLOW; set by initializeChildren; the child NODES are ARENA and are not freed with the array")
r(f,"SearchNode","children1","SearchChildPointer*","OWNED","OnceCell<Box<[SearchChildPointer]>>","len SIZE1OVERFLOW; set by tryExpandingChildrenCapacityAssumeFull; PORT NOTE: collapseChildrenCapacity (&self) lowers the state and keeps the emptied array (not freed: no &mut through a shared node), which tryExpandingChildrenCapacityAssumeFull reuses, reset to fresh slots")
r(f,"SearchNode","children2","SearchChildPointer*","OWNED","OnceCell<Box<[SearchChildPointer]>>","len SIZE2OVERFLOW; as children1")
r(f,"SearchNode","stats","NodeStatsAtomic","OWNED","NodeStatsAtomic","inline; NodeStatsAtomic(const&) -> new_copy")
r(f,"SearchNode","virtualLosses","std::atomic<int32_t>","OWNED","Cell<i32>","fetch_add -> port::cpp::cell_fetch_add")
r(f,"SearchNode","lastSubtreeValueBiasDeltaSum","double","OWNED","Cell<f64>","guarded by SubtreeValueBiasEntry::entryLock upstream")
r(f,"SearchNode","lastSubtreeValueBiasWeight","double","OWNED","Cell<f64>","guarded by entryLock upstream")
r(f,"SearchNode","subtreeValueBiasTableEntry","std::shared_ptr<SubtreeValueBiasEntry>","SHARED","RefCell<Option<Rc<SubtreeValueBiasEntry>>>","set in allocateOrFindNode, cleared by removeSubtreeValueBias; strong_count matters (clearUnusedSynchronous)")
r(f,"SearchNode","graphHash","Hash128","OWNED","Hash128","constant after construction")
r(f,"SearchNode","evalCacheEntry","std::shared_ptr<EvalCacheEntry>","SHARED","RefCell<Option<Rc<EvalCacheEntry>>>","evalcache deferred: EvalCacheEntry is an uninhabited placeholder enum until ported, so always None; TODO(port): deferred (evalcache.cpp)")
r(f,"SearchNode","dirtyCounter","std::atomic<int32_t>","OWNED","Cell<i32>","")
r(f,"SearchNode","storeNNOutput(newNNOutput)","std::shared_ptr<NNOutput>*","OWNED","Rc<NNOutput>","param, ownership transferred; same for storeHumanOutput / *IfNull")
r(f,"SearchNode","storeNNOutput(thread)","SearchThread&","BORROWED","&mut SearchThread","param")
r(f,"SearchNode","maybeExpandChildrenCapacityForNewChild(stateValue)","SearchNodeState&","BORROWED","&mut SearchNodeState","in/out param")

# ---------------- search/searchnodetable.h ----------------
f = "cpp/search/searchnodetable.h"
r(f,"SearchNodeTable","entries","std::vector<std::map<Hash128,SearchNode*>>","ARENA","RefCell<Vec<BTreeMap<Hash128, NodeIdx>>>","the arena owns the nodes; upstream `delete node` -> search.node_arena.free(idx) + map erase")
r(f,"SearchNodeTable","mutexPool","MutexPool*","OWNED","u32","PORT NOTE: renamed mutex_pool_num_mutexes (= numShards); Search's ctor reads it as nodeTable->mutexPool->getNumMutexes()")

# ---------------- search/search.h ----------------
f = "cpp/search/search.h"
r(f,"Search","searchNodeAge","uint32_t","OWNED","Cell<u32>","tree walks run through &self while their callbacks borrow the Search")
r(f,"Search","nonSearchRand","Rand","OWNED","RefCell<Rand>","applyRecursively* run through &self; never borrowed across an await")
r(f,"SearchThread","board","Board","OWNED","Board","")
r(f,"SearchThread","history","BoardHistory","OWNED","BoardHistory","")
r(f,"SearchThread","graphPath","std::unordered_set<SearchNode*>","ARENA","HashSet<NodeIdx>","membership only, never iterated")
r(f,"SearchThread","rand","Rand","OWNED","Rand","Rand is not Clone (copy ctor deleted)")
r(f,"SearchThread","nnResultBuf","NNResultBuf","SHARED","Rc<NNResultBuf>","PORT NOTE: inline upstream; Rc so the NN query queue can hold it while this virtual thread awaits")
r(f,"SearchThread","statsBuf","std::vector<MoreNodeStats>","OWNED","Vec<MoreNodeStats>","")
r(f,"SearchThread","oldNNOutputsToCleanUp","std::vector<std::shared_ptr<NNOutput>*>","OWNED","Vec<Rc<NNOutput>>","kept so old outputs live as long as upstream's")
r(f,"SearchThread","illegalMoveHashes","std::set<Hash128>","OWNED","BTreeSet<Hash128>","")
r(f,"SearchThread","SearchThread(search)","const Search&","BORROWED","&Search","ctor param")
r(f,"Search","rootSymDupLoc","bool[Board::MAX_ARR_SIZE]","OWNED",f"[bool; {A}]","inline")
r(f,"Search","rootSafeArea","Color*","OWNED",f"Box<[Color; {A}]>","new Color[MAX_ARR_SIZE] in ctor, delete[] in dtor")
r(f,"Search","rootKoHashTable","KoHashTable*","OWNED","Box<KoHashTable>","new in ctor, never NULL")
r(f,"Search","valueWeightDistribution","DistributionTable*","OWNED","Box<DistributionTable>","new in ctor, never NULL")
r(f,"Search","patternBonusTable","PatternBonusTable*","OWNED","Option<Box<PatternBonusTable>>","patternbonustable deferred: placeholder uninhabited type, so always None; TODO(port): deferred (patternbonustable.cpp)")
r(f,"Search","externalPatternBonusTable","std::unique_ptr<PatternBonusTable>","OWNED","Option<Box<PatternBonusTable>>","deferred, as patternBonusTable")
r(f,"Search","evalCache","std::shared_ptr<EvalCacheTable>","SHARED","Option<Rc<EvalCacheTable>>","shared across bots (analysis.cpp); deferred, so always None; TODO(port): deferred (evalcache.cpp)")
r(f,"Search","nonSearchRand","Rand","OWNED","Rand","")
r(f,"Search","logger","Logger*","BORROWED","(removed)","log crate; ctor param dropped (PORT NOTE)")
r(f,"Search","nnEvaluator","NNEvaluator*","BORROWED","Rc<NNEvaluator>","owned by the host (setup/analysis), outlives Search; setNNEval replaces it")
r(f,"Search","humanEvaluator","NNEvaluator*","NULLABLE_BORROWED","Option<Rc<NNEvaluator>>","")
r(f,"Search","rootNode","SearchNode*","ARENA","Option<NodeIdx>","allocated directly (not in nodeTable); `delete rootNode` -> self.node_arena.free(idx)")
r(f,"Search","(new) nodeArena","-","OWNED","NodeArena","PORT NOTE: port::node_arena::NodeArena owns every SearchNode (root and table nodes)")
r(f,"Search","nodeTable","SearchNodeTable*","OWNED","Box<SearchNodeTable>","new in ctor, never NULL")
r(f,"Search","mutexPool","MutexPool*","OWNED","u32","PORT NOTE: renamed mutex_pool_num_mutexes; createMutexIdxForNode = thread.rand.next_uint() & (self.mutex_pool_num_mutexes - 1)")
r(f,"Search","subtreeValueBiasTable","SubtreeValueBiasTable*","OWNED","Option<Box<SubtreeValueBiasTable>>","created/deleted in beginSearch and computeRootValues (outside the playout loop)")
r(f,"Search","numThreadsSpawned","int","OWNED","(removed)","no task threads (section 11)")
r(f,"Search","threads","std::thread*","OWNED","(removed)","")
r(f,"Search","threadTasks","ThreadSafeQueue<std::function<void(int)>*>*","OWNED","(removed)","")
r(f,"Search","threadTasksRemaining","ThreadSafeCounter*","OWNED","(removed)","")
r(f,"Search","oldNNOutputsToCleanUpMutex","std::mutex","OWNED","(removed)","")
r(f,"Search","oldNNOutputsToCleanUp","std::vector<std::shared_ptr<NNOutput>*>","OWNED","RefCell<Vec<Rc<NNOutput>>>","transferOldNNOutputs runs at the end of each virtual thread, i.e. during search through &self")
r(f,"Search","Search(nnEval)","NNEvaluator*","BORROWED","Rc<NNEvaluator>","ctor param; humanEval -> Option<Rc<NNEvaluator>>")
r(f,"Search","runWholeSearch(searchBegun)","const std::function<void()>*","NULLABLE_BORROWED","Option<&dyn Fn()>","param")
r(f,"Search","runWholeSearch(shouldStopEarly)","std::function<bool()>*","NULLABLE_BORROWED","Option<&dyn Fn() -> bool>","param; called by every virtual thread, so captured state lives in Cell")
r(f,"Search","getRootNode(return)","const SearchNode*","ARENA","Option<Rc<SearchNode>>","same for getChildForMove")
r(f,"Search","allocateOrFindNode(return)","SearchNode*","ARENA","Rc<SearchNode>","store into a child pointer with .idx")
r(f,"Search","enumerateTreePostOrder(return)","std::vector<SearchNode*>","ARENA","Vec<NodeIdx>","")
r(f,"Search","applyRecursivelyPostOrderMulithreaded(nodes)","const std::vector<SearchNode*>&","ARENA","&[NodeIdx]","param; same for applyRecursivelyAnyOrderMulithreaded")
r(f,"Search","applyRecursivelyPostOrderMulithreaded(f)","std::function<void(SearchNode*,int)>*","NULLABLE_BORROWED","Option<&mut dyn FnMut(&SearchNode, i32)>","param; NULL = mark only")
r(f,"Search","applyRecursivelyPostOrderMulithreadedHelper(rand)","PCG32*","BORROWED","&mut PCG32","param; same for the AnyOrder helper")
r(f,"Search","applyRecursivelyPostOrderMulithreadedHelper(nodeBuf)","std::unordered_set<SearchNode*>&","ARENA","&mut HashSet<NodeIdx>","param")
r(f,"Search","getSharpScoreHelper(graphPath)","std::unordered_set<const SearchNode*>&","ARENA","&mut HashSet<NodeIdx>","param; same for traverseTreeForOwnership* and getShallowAverageShorttermWLAndScoreErrorHelper")
r(f,"Search","traverseTreeForOwnership(averaging)","Func&","BORROWED","&mut F (F: FnMut)","template param")
r(f,"Search","performTaskWithThreads(task)","std::function<void(int)>*","BORROWED","&mut dyn FnMut(i32)","param; non-search tasks only (search loop uses join_all)")
r(f,"Search","maybeAddPolicyNoiseAndTemp(return)","std::shared_ptr<NNOutput>*","OWNED","Option<Rc<NNOutput>>","NULL when no noise or temperature was applied")
r(f,"Search","getPlaySelectionValues(retVisitCounts)","std::vector<double>*","NULLABLE_BORROWED","Option<&mut Vec<f64>>","param")
r(f,"Search","getPlaySelectionValues(lcbBuf)","double[NNPos::MAX_NN_POLICY_SIZE]","BORROWED",f"&mut [f64; {P}]","param; same for radiusBuf")
r(f,"Search","getPolicy(policyProbs)","float[NNPos::MAX_NN_POLICY_SIZE]","BORROWED",f"&mut [f32; {P}]","param")
r(f,"Search","getAnalysisJson(ret)","nlohmann::json&","BORROWED","&mut serde_json::Value","param; serialize with port::nlohmann::dump")

# ---------------- search/analysisdata.h ----------------
f = "cpp/search/analysisdata.h"
r(f,"AnalysisData","node","const SearchNode*","NULLABLE_BORROWED","Option<NodeIdx>","\"ONLY valid so long as search is not cleared\"; resolve with search.node_arena.get(idx)")
r(f,"AnalysisData","pv","std::vector<Loc>","OWNED","Vec<Loc>","")
r(f,"AnalysisData","pvVisits","std::vector<int64_t>","OWNED","Vec<i64>","")
r(f,"AnalysisData","pvEdgeVisits","std::vector<int64_t>","OWNED","Vec<i64>","")
r(f,"AnalysisData","writePV(out)","std::ostream&","BORROWED","&mut dyn std::fmt::Write","param; same for the other write* methods")

# ---------------- search/reportedsearchvalues.h ----------------
f = "cpp/search/reportedsearchvalues.h"
r(f,"ReportedSearchValues","ReportedSearchValues(search)","const Search&","BORROWED","&Search","ctor param (new_search_...)")

# ---------------- search/subtreevaluebiastable.h ----------------
f = "cpp/search/subtreevaluebiastable.h"
r(f,"SubtreeValueBiasEntry","deltaUtilitySum","double","OWNED","Cell<f64>","guarded by entryLock upstream; the entry is shared via Rc")
r(f,"SubtreeValueBiasEntry","weightSum","double","OWNED","Cell<f64>","guarded by entryLock upstream")
r(f,"SubtreeValueBiasEntry","entryLock","mutable std::atomic_flag","OWNED","(removed)","PORT NOTE: spin lock removed")
r(f,"SubtreeValueBiasTable","entries","std::vector<std::map<Hash128,std::shared_ptr<SubtreeValueBiasEntry>>>","SHARED","RefCell<Vec<BTreeMap<Hash128, Rc<SubtreeValueBiasEntry>>>>","get() inserts with entry().or_insert_with(|| Rc::new(..)) (net effect of operator[] + null check); clearUnusedSynchronous: use_count() <= 1 -> Rc::strong_count(v) <= 1")
r(f,"SubtreeValueBiasTable","mutexPool","MutexPool*","OWNED","(removed)","PORT NOTE: shard index still computed by hash0 % entries.len()")
r(f,"SubtreeValueBiasTable","get(return)","std::shared_ptr<SubtreeValueBiasEntry>","SHARED","Rc<SubtreeValueBiasEntry>","")
f2 = "cpp/search/subtreevaluebiastable.cpp"
r(f2,"(file static)","patternHasher","static LocalPatternHasher","STATIC","LocalPatternHasher","field of SubtreeValueBiasTableStatics in static SUBTREEVALUEBIASTABLE_STATICS: OnceLock<SubtreeValueBiasTableStatics>; initIfNeeded() -> &'static SubtreeValueBiasTableStatics")
r(f2,"(file static)","ZOBRIST_MOVE_LOCS","static Hash128[Board::MAX_ARR_SIZE][2]","STATIC",f"[[Hash128; 2]; {A}]","field of SubtreeValueBiasTableStatics; same RNG seeds and call order")
r(f2,"(file static)","ZOBRIST_KO_BAN","static Hash128[Board::MAX_ARR_SIZE]","STATIC",f"[Hash128; {A}]","field of SubtreeValueBiasTableStatics")
r(f2,"(file static)","isInited / initMutex","static std::atomic<bool> / static std::mutex","STATIC","(removed)","OnceLock state replaces both")

# ---------------- search/evalcache.h (deferred) ----------------
f = "cpp/search/evalcache.h"
r(f,"EvalCacheEntry","firstExploreEvals","std::map<Loc,FirstExploreEval>","OWNED","BTreeMap<Loc, FirstExploreEval>","deferred module (manifest [[ignored]] deferred); type used once ported")
r(f,"EvalCacheTable","entries","std::vector<std::map<Hash128,std::shared_ptr<EvalCacheEntry>>>","SHARED","RefCell<Vec<BTreeMap<Hash128, Option<Rc<EvalCacheEntry>>>>>","deferred; Option because update() reads via operator[], which inserts a null")
r(f,"EvalCacheTable","mutexPool","MutexPool*","OWNED","(removed)","deferred")
r(f,"EvalCacheTable","update(node)","const SearchNode*","BORROWED","&SearchNode","deferred; PORT NOTE: gains a nodes: &NodeArena param to dereference children")
r(f,"EvalCacheTable","find(return)","std::shared_ptr<EvalCacheEntry>","SHARED","Option<Rc<EvalCacheEntry>>","deferred")

# ---------------- search/patternbonustable.h (deferred) ----------------
f = "cpp/search/patternbonustable.h"
r(f,"PatternBonusTable","entries","std::vector<std::map<Hash128,PatternBonusEntry>>","OWNED","Vec<BTreeMap<Hash128, PatternBonusEntry>>","deferred; read-only during search")
r(f,"PatternBonusTable","addBonus(hashesThisGame)","std::set<Hash128>&","BORROWED","&mut BTreeSet<Hash128>","deferred; param")
f2 = "cpp/search/patternbonustable.cpp"
r(f2,"(file static)","patternHasher / ZOBRIST_MOVE_LOCS","static LocalPatternHasher / static Hash128[Board::MAX_ARR_SIZE]","STATIC",f"PatternBonusTableStatics {{ patternHasher: LocalPatternHasher, ZOBRIST_MOVE_LOCS: [Hash128; {A}] }}","deferred; static PATTERNBONUSTABLE_STATICS: OnceLock<PatternBonusTableStatics>")

# ---------------- search/distributiontable.h ----------------
f = "cpp/search/distributiontable.h"
r(f,"DistributionTable","pdfTable","double*","OWNED","Box<[f64]>","new double[size] in ctor, delete[] in dtor")
r(f,"DistributionTable","cdfTable","double*","OWNED","Box<[f64]>","new double[size] in ctor, delete[] in dtor")
r(f,"DistributionTable","DistributionTable(pdf)","const std::function<double(double)>&","BORROWED","&dyn Fn(f64) -> f64","ctor param; same for cdf")

# ---------------- search/mutexpool.h ----------------
f = "cpp/search/mutexpool.h"
r(f,"MutexPool","mutexes","std::mutex*","OWNED","(removed)","class not ported (manifest: replaced); holders keep only the count where it feeds a computation (SearchNodeTable, Search)")

# ---------------- search/asyncbot.h ----------------
f = "cpp/search/asyncbot.h"
r(f,"AsyncBot","search","Search*","OWNED","Box<Search>","new in ctor, delete in dtor")
r(f,"AsyncBot","controlMutex","std::mutex","OWNED","(removed)","no bot thread: an async driver over Search (manifest)")
r(f,"AsyncBot","threadWaitingToSearch","std::condition_variable","OWNED","(removed)","")
r(f,"AsyncBot","userWaitingForStop","std::condition_variable","OWNED","(removed)","the caller awaits the search future")
r(f,"AsyncBot","searchThread","std::thread","OWNED","(removed)","")
r(f,"AsyncBot","shouldStopNow","std::atomic<bool>","SHARED","Rc<Cell<bool>>","set by stopWithoutWait from the host (terminate) while the search future runs; read by the shouldStopEarly closure")
r(f,"AsyncBot","isKilled","bool","SHARED","Rc<Cell<bool>>","set by setKilled from another virtual thread (AsyncBotStopHandle, analysis shutdown)")
r(f,"AsyncBot","queuedOnMove","std::function<void(Loc,int,Search*)>","OWNED","Option<Box<dyn FnMut(Loc, i32, &mut Search)>>","set by the private genMove*Async queueing fns")
r(f,"AsyncBot","analyzeCallback","std::function<void(const Search*)>","SHARED","Option<Rc<dyn Fn(&Search) -> Result<(), KataError>>>","Rc: internalSearchThreadLoop copies it; Result: an exception in upstream's callback thread terminates the process")
r(f,"AsyncBot","searchBegunCallback","std::function<void()>","SHARED","Option<Rc<dyn Fn(&Search)>>","gets the Search (upstream callbacks reach it through the bot, which the search borrows mutably)")
r(f,"AsyncBot","AsyncBot(nnEval)","NNEvaluator*","BORROWED","Rc<NNEvaluator>","ctor param; humanEval -> Option<Rc<NNEvaluator>>; Logger* param dropped")
r(f,"AsyncBot","getSearch(return)","const Search*","BORROWED","&Search","")

# ---------------- command/analysis.cpp ----------------
f = "cpp/command/analysis.cpp"
r(f,"AnalyzeRequest","status","std::atomic<int>","SHARED","Cell<i32>","the request is shared between the request handler and the search driver as Rc<AnalyzeRequest>")
r(f,"(local)","openRequests","std::map<int64_t,AnalyzeRequest*>","SHARED","RefCell<BTreeMap<i64, Rc<AnalyzeRequest>>>","field of AnalysisContext; openRequestsMutex removed")

out = sys.argv[1]
with open(out, "w") as fh:
    fh.write("\t".join(["file","class","field","cpp_type","ownership","rust_type","notes"]) + "\n")
    for row in rows:
        fh.write("\t".join(row) + "\n")
print(len(rows))
