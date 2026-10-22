# PORTING.md: the binding guide for porting KataGo to Rust

Applies to upstream **v1.16.5** (`ba938676d7f42d70950b3a535af2466fb642008c`). Every agent that
writes or reviews ported code follows this file. Precedence: PORTING.md > OWNERSHIP.tsv >
CLAUDE.md > your own judgement. When this guide is silent or seems wrong, write
`// TODO(port): <what is unclear>` at the spot and say so in your report: we fix the guide,
not the diff. Status: **revised after the pilot** (hash, rand, rules, board; 2026-10-22).

## 1. Goal and principles

Fidelity over idiom. The port is a mechanical translation whose shape matches upstream so
that an upstream diff maps to a Rust diff, function by function (plan §2). Idiomatic Rust
is used only at the API edges (`katago-wasm`, `katago-cli`), never inside ported files.

| Rule | Example |
|---|---|
| One `.rs` per `.cpp`/`.h` pair, at the mirrored path. The manifest has the exact map. | `cpp/game/board.cpp` + `board.h` → `crates/katago-core/src/game/board.rs` |
| A class's struct goes in its header's `.rs`. Each `.cpp` that defines methods gets its own `impl` block in its own `.rs`. | `search/searchexplorehelpers.rs` contains `impl Search { fn get_explore_scaling … }` |
| Header inline functions live in the header's `.rs`. File-`static` helpers become private `fn`s in the same file. | `static double inverse_atan(double)` → `fn inverse_atan(x: f64) -> f64` |
| Type, struct, enum and enum value names stay verbatim. | `Board`, `NNOutput`, `SearchNodeChildrenReference`, `enabled_t::Auto` |
| **Nested types** (`Board::ChainData`, `Board::MoveRecord`) become module-level types in the outer class's file, named without the prefix, tagged `/// upstream: Board::ChainData (board.h)`. Other modules import them by path (`crate::game::board::MoveRecord`). | `struct ChainData` in `game/board.rs` |
| **`crate::core` shadows Rust's `core`.** Inside katago-core, `use core::…` means *our* module; write `::core::…` for the language crate. | `::core::cmp::Ordering` |
| **Reserved words.** Edition 2024 reserves `gen`; upstream identifiers that collide with Rust keywords get a trailing `_`. | `gen` → `gen_`, `move` → `move_` |
| Function, method, field and local names use snake_case: `heck` 0.5 `to_snake_case`, **except** that `UInt` is one word (`nextUInt` → `next_uint`, `hexStringToUInt64` → `hex_string_to_uint64`, not heck's `next_u_int`). When heck and this exception disagree, the exception wins. | `searchIsLadderCaptured` → `search_is_ladder_captured`, `getNNOutput` → `get_nn_output`, `nnXLen` → `nn_x_len` |
| Constants, statics and macros-as-constants stay verbatim, including SCREAMING_CASE. Add `#[allow(non_upper_case_globals)]` where needed. | `NNInputs::NUM_FEATURES_SPATIAL_V7`, `Board::MAX_ARR_SIZE`, `SearchNode::STATE_EXPANDED0` |
| A namespace becomes a unit struct with associated fns and consts, so call sites read the same. | `namespace Location {…}` → `pub struct Location; impl Location { pub fn get_loc(…) }` → `Location::get_loc(x, y, x_size)` |
| C++ `public` becomes `pub`. `private`/`protected` becomes `pub(crate)`, because sibling files implement the same class. | |
| Keep every upstream comment, translated `//` → `//`, at the same place. Don't add commentary. | |
| Keep statement order, branch order, early returns, loop shape and evaluation order. Never merge, split, reorder or "simplify" functions. | |
| Every ported item carries a doc tag, and every file starts with a header line. | see below |

```rust
// Ported from KataGo cpp/game/board.cpp, cpp/game/board.h @ ba938676d7f42d70950b3a535af2466fb642008c

/// upstream: Board::searchIsLadderCaptured (board.cpp)
pub fn search_is_ladder_captured(&mut self, loc: Loc, defender_first: bool, buf: &mut Vec<Loc>) -> bool {
```

Tag format: `/// upstream: <Qualified::name> (<file>)`, one per fn, method, struct, const group
or static. Overloads get the same qualified name. For file-scope statics use
`(file.cpp) static`. `xtask upstream-diff` greps these tags, so a missing tag is a bug.

**Overloads.** (1) A `const char*`/`const std::string&` pair collapses into one `&str` fn.
(2) A const/non-const pair (`getNNOutput()`, `getNNOutput() const`) becomes one `&self` fn.
(3) Of the remaining overloads, the one with the fewest parameters keeps the base name (ties: first
declared in the header). Each other overload appends `_` + the snake_case names of its
parameters that are not in the base overload, in declaration order. (4) If that still
collides, the overload declared later appends the snake_case of its first differing parameter's C++ type (`std::ostringstream&` → `_ostringstream`, `const std::string&` → `_string`, `uint64_t` → `_u64`). Parameter names come from the **header** declaration when header and `.cpp` disagree. (4c) When a header declaration and the `.cpp` definition name or order parameters differ, the **definition** wins (it is what the body uses); note it in a PORT NOTE. (4b) For a constructor whose parameters all have defaults, the base `new` is the all-defaulted call (e.g. `ConfigParser(bool=false, bool=true)` → `new(ko, kfi)` with callers passing the values). (5) **Appendix B overrides rules 3–4**; add an entry there whenever a generated name is longer than ~40 characters or reads badly. Constructors use the same
rule with base `new`. A copy constructor becomes `impl Clone`, or `new_copy` when it takes
extra arguments. Examples: `Rand::nextUInt()` → `next_uint`, `nextUInt(n)` → `next_uint_n`,
`nextUInt(freq,n)` → `next_uint_freq_n`; `Rand(seed: string)` → `new_seed`, `Rand(uint64_t seed)`
→ `new_seed_u64`; `Location::getCenterLoc(const Board&)` → `get_center_loc`,
`getCenterLoc(x_size,y_size)` → `get_center_loc_x_size_y_size`; `SearchNode(const SearchNode&,
bool fnt, bool copySVB)` → `new_copy`.

## 2. Workflow

| Phase | Goal | Done when |
|---|---|---|
| **A: draft** | Faithful translation of every fn in the assigned files. **Need not compile.** | Every upstream fn is ported, listed as skipped in the manifest `notes`, or marked `TODO(port)`. |
| **B: compile** | `cargo check -p katago-core` (and `--target wasm32-unknown-unknown`). Errors are grouped by file and fixed by the owner of that file. | Clean check, no `TODO(port)` left without an owner or a reason. |
| **C: parity** | Run the oracles (§13) to green, a batch of failing fixtures per agent. | Module tests pass, then set manifest `status = "done"`. |

**Agent loop per unit.** One implementer, then **two adversarial reviewers** in separate contexts.
Each reviewer is told: "find every way this Rust differs from the C++". They check types,
signedness, float width, evaluation order, asserts, missing fns and wrong ownership class.
A fixer then applies the review. Reviewers diff the code side by side with the upstream file.
They don't read the implementer's summary.

**Markers (grep-able, mandatory).**

| Marker | Use |
|---|---|
| `// TODO(port): <reason>` | Anything you could not port faithfully, or would otherwise guess. A flag beats a guess. Deferred features use `// TODO(port): deferred (<file.cpp>)`. |
| `// PORT NOTE: <what and why>` | A deliberate divergence that this guide prescribes, such as a removed mutex, Cell for an atomic, an added parameter, or a changed signature. Copy the reason from this guide. |
| `// PERF(port): <note>` | A slow shape kept on purpose, or a hot spot noticed. Don't optimise. |
| `// SAFETY: <invariant>` | On every `unsafe` block. katago-core should need none outside `port/`. |

## 3. Type map

| C++ | Rust | Notes |
|---|---|---|
| `Loc` (`typedef short`) | `pub type Loc = i16;` | Signed, as upstream. **The plan text said u16: wrong, use i16.** Index with `loc as usize`. Do arithmetic in `i32` (C++ promotes): `(loc as i32 + adj) as Loc`. |
| `Player`, `Color` (`typedef int8_t`) | `pub type Player = i8;` `pub type Color = i8;` | `P_BLACK`, `C_WALL`… are `pub const X: Player`. `getOpp(c)` → `c ^ 3`. |
| `Spot` | none | Doesn't exist in v1.16.5. Don't invent it. |
| `SearchNodeState` (`typedef int`) | `pub type SearchNodeState = i32;` | Every `typedef`/`using` becomes `pub type`. |
| `bool` `char` | `bool`, and `u8` or `char` | x86-64 `char` is **signed**: `(uint64_t)c` → `c as i8 as u64` (sign-extends, see `Hash::simpleHash`). Character literals used for text → `char`. |
| `int8_t` `uint8_t` `short` `int` `unsigned` `int64_t` `uint64_t` | `i8` `u8` `i16` `i32` `u32` `i64` `u64` | Width and signedness exact. `long` is 64-bit on the fixture platform, so `i64`. |
| `size_t` | `usize` for lengths and indices | wasm32 `usize` is **32-bit**, native `size_t` is 64. If a `size_t` value is hashed, fed to RNG, serialized, or can exceed 2³², use `u64` and cast at the indexing site. |
| `float` / `double` | `f32` / `f64` | Never widen or narrow on your own. See §6. |
| `Hash128` | `#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)] pub struct Hash128 { pub hash0: u64, pub hash1: u64 }` | **`Ord`/`PartialOrd` by hand**, comparing `hash1` first, then `hash0` (`hash.h`). Derived `Ord` would be wrong, and it decides `std::map` order. `^ \| &` → `impl BitXor/BitOr/BitAnd` + `*Assign`. `operator<<` → `impl Display` (= `to_string()`, uppercase hex, hash1 then hash0). Constructors are `const fn new(h0, h1)`. |
| `T x[N]` (field or local) | `[T; N as usize]` | Keep the constant's declared type (`MAX_ARR_SIZE` is `i32`). Write `as usize` in array types and indices. Uninitialized C++ locals are zero-initialized in Rust (no note needed). |
| `std::vector<T>` | `Vec<T>` | Param `const vector<T>&` → `&[T]`. Param `vector<T>&` → `&mut Vec<T>`. `vector<bool>` → `Vec<bool>`. |
| `std::string` | `String` | Param `const string&`/`const char*` → `&str`. A `const char*` ends at its first NUL: when the `&str` can contain one, read it through `port::cpp::c_str` (PORT NOTE). When rule 1 of §1 collapses a `const char*`/`std::string` overload pair, follow the overload upstream callers actually use, and say which in the PORT NOTE. Out-param `string&` → `&mut String`. `std::string` is **bytes**: indexing, `size()`, `substr`, `find` and `+ char` are byte operations. Use `s.as_bytes()[i]`, `s.len()`, and slice only at ASCII boundaries (all KataGo text is ASCII); a byte that may be non-ASCII → `TODO(port)`. |
| `std::map<K,V>` / `std::set<K>` | `BTreeMap<K,V>` / `BTreeSet<K>` | Always, even when order looks irrelevant. `K` must use the C++ `operator<` order (see Hash128). `std::string` keys compare as unsigned bytes in both languages. |
| `std::unordered_map/set` | `HashMap/HashSet` | Only for lookups. **Never iterate one.** If upstream iterates an unordered container, write `TODO(port)`. |
| `std::pair<A,B>` | `(A, B)` | `STRUCT_NAMED_PAIR(Loc,loc,Player,pla,Move)` → `#[derive(Clone, Copy, Debug, Default, PartialEq)] pub struct Move { pub loc: Loc, pub pla: Player }` + `Move::new`. |
| `std::unique_ptr<T>` | `Box<T>`, or `Option<Box<T>>` if it can be null | |
| `std::shared_ptr<T>` | `Rc<T>`, or `Option<Rc<T>>` | `use_count()` → `Rc::strong_count`. `make_shared` → `Rc::new`. A mutation through a shared_ptr that is provably unique → `Rc::get_mut(..).expect("unique: <why>")`. Otherwise `TODO(port)`. |
| `std::shared_ptr<T>*` (heap shared_ptr, only for atomic swaps) | `Rc<T>` | The extra indirection disappears: `new shared_ptr(..)`/`delete p` vanish. `// PORT NOTE: shared_ptr* indirection only existed for atomics`. |
| raw `T*` / `T&` **field** | per **OWNERSHIP.tsv** | Copy its `rust_type` verbatim. A pointer field missing from the TSV is a `TODO(port)`. Don't decide ownership yourself. |
| raw `T*` / `T&` **parameter** | `&T`/`&mut T`; `Option<&T>` if any caller passes NULL; `&[T]`/`&mut [T]` if it points at a buffer | `Color* result` → `&mut [Color]`. `Color area[Board::MAX_ARR_SIZE]` → `&mut [Color; Board::MAX_ARR_SIZE as usize]`. `const` decides `&` vs `&mut`. |
| `SearchNode*`/`SearchNode&` | param: `&SearchNode`/`Option<&SearchNode>`; return: `Rc<SearchNode>`/`Option<Rc<SearchNode>>`; field or collection: `NodeIdx` | See §4. Node identity is `node.idx`. |
| `nullptr` / `NULL` | `None` | |
| `std::atomic<T>` | `Cell<T>` | Single-threaded. Keep the memory order as a comment: `node.state.get() /*acquire*/`. Ops are listed in §5. |
| `std::atomic_flag` spin lock, `std::mutex`, `lock_guard`, `MutexPool` | removed | `// PORT NOTE: lock removed (single-threaded)`. **Keep every computation that fed the lock**: `createMutexIdxForNode` still draws `thread.rand.next_uint()`, or the RNG stream shifts. Data a mutex guarded and that is written during search → `RefCell` (borrow scope = lock scope). |
| `std::condition_variable` | `port::executor::WakerList` | §11. |
| `std::thread` | removed | §11. |
| `std::function<R(A)>` field | `Box<dyn Fn(A) -> R>`, or `Option<…>` when it can be empty | |
| `std::function` param | `&dyn Fn(..)` if upstream may call it from several search threads at once (`shouldStopEarly`, `searchBegun`, analyze callbacks); otherwise `&mut dyn FnMut(..)` (`performTaskWithThreads`, `applyRecursively*`) | `std::function*` that can be NULL → `Option<…>`. Mutable state captured by an `Fn` lives in `Cell`/`RefCell`. |
| templates | generics | `template<typename Func> … Func& f` → `f: &mut F` where `F: FnMut(..)`. |
| `const` member fn | `&self` | |
| non-const member fn | `&mut self`, **except** methods reachable from `Search::runSinglePlayout` or `NNEvaluator::evaluate`, which take `&self` and mutate only through the `Cell`/`RefCell` fields in OWNERSHIP.tsv | §11 explains why. |
| static data member / file-scope static, constant initializer | `pub const` (associated const on the struct) | `Rules::ZOBRIST_KO_RULE_HASH`, `Board::ZOBRIST_PASS_ENDS_PHASE`, `FORCE_NON_TERMINAL_HASH`. |
| static data, initialized at runtime | one `OnceLock<XStatics>` per class or file | §8. |
| `enum` / `enum class` | `#[repr(i32)] enum` with verbatim variant names, or `pub const` ints when upstream uses plain `static const int` (`Rules::KO_SIMPLE`) | Don't convert int constants into an enum. |
| `StringError`, `IOError`, `ValueError`, `ConfigParsingError`, `CommandError` | `KataError` variants of the same names (`core/global.rs`) | §7. |
| `std::ostream&` param | `&mut dyn std::fmt::Write` | `out << x` → `write!(out, …).unwrap()`, and `out << x << endl` → `writeln!(out, …).unwrap()` (not `write!` with `\n`) (writing to a `String` can't fail; this is the only allowed bare unwrap). `cout`/`cerr` → `log` macros. |
| `std::cout << …` written directly (not via a passed `ostream&`) | `log::info!`, and tests that compare console text must know which lines bypassed the logger prefix (`SearchParams::printParams` writes two) | Keep a list of such lines next to the test that captures them. |
| `Logger*` / `logger->write(s)` | removed / `log::info!("{}", s)` | Manifest: logger replaced by the `log` crate. `if(logger != NULL) logger->write(..)` → just log. Keep the message text. |
| iterators | indices or slice iterators | `it = m.erase(it)` loops → `BTreeMap::retain`, only if visit order and side effects stay identical; otherwise collect the keys first. |

## 4. Ownership, the node arena, and interior mutability

OWNERSHIP.tsv lists every pointer, smart-pointer, atomic and container-of-pointers field in
the ported classes. The classes are: **OWNED** (this object frees it), **BORROWED** (non-owning,
outlives us), **NULLABLE_BORROWED**, **SHARED** (refcounted or co-owned), **ARENA** (points
into the search node arena) and **STATIC** (process-wide). Rules behind the column:

- **Long-lived BORROWED collaborators** (the `NNEvaluator*` held by `Search`, `Search*` in
  `AsyncBot`) become `Rc<T>`. This keeps lifetime parameters off long-lived structs.
  **Short-lived views** (`SearchNodeChildrenReference`) borrow: `&'a SearchNode`.
- **Search nodes live in `port::node_arena::NodeArena`**, a field of `Search`. A `SearchNode*`
  stored anywhere (child pointers, node table, `rootNode`, `graphPath`, `AnalysisData::node`)
  is a `NodeIdx(u32)`. `arena.get(idx) -> Rc<SearchNode>`, `arena.alloc(|idx| SearchNode::new(idx, …)) -> NodeIdx`,
  `arena.free(idx)` (= `delete node`), `arena.with_mut(idx, |n| …)` (panics if the node is shared:
  only for `collapseChildrenCapacity`, which upstream calls "not thread-safe").
  All of them take `&self`. `SearchNode` gains a first field `idx: NodeIdx` (PORT NOTE), and
  pointer comparisons such as `&node == rootNode` become `node.idx == root_idx`.
  Code outside `Search` that dereferences children (`EvalCacheTable::update`) gets an extra
  `nodes: &NodeArena` parameter (PORT NOTE).
- **`SearchNode` and everything reachable during search is mutated through `&`.** Every field
  that upstream writes during search (atomics, fields under `statsLock`/`entryLock`/a mutex,
  fields set after construction) is `Cell`/`RefCell`/`OnceCell` as the TSV says. Children
  arrays are `OnceCell<Box<[SearchChildPointer]>>`: set once while searching, taken only via
  `with_mut`. `ConstSearchNodeChildrenReference` becomes `pub type … = SearchNodeChildrenReference<'a>`.
- **`NNOutput` is immutable once shared** (`Rc<NNOutput>`). Upstream only mutates copies
  (noise, antimirror) or still-unique results (postprocess in `evaluate`).

## 5. Idiom map

| C++ idiom | Rust | Upstream example |
|---|---|---|
| `FOREACHADJ(BLOCK)` | `for adjoffset in [-(self.x_size+1), -1, 1, self.x_size+1] { BLOCK }` with `ADJOFFSET` → `adjoffset: i32`. `return` inside the block still returns from the fn. If `BLOCK` contains `break`/`continue` (none today), write `TODO(port)`: in C++ they target the *enclosing* loop. | `Board::isSuicide` |
| `ADJ0..ADJ3`, `adj_offsets[i]` | `-(self.x_size+1)` … ; `self.adj_offsets[i] as i32` | `Loc adj = loc + adj_offsets[i];` → `let adj: Loc = (loc as i32 + self.adj_offsets[i as usize] as i32) as Loc;` |
| `for(int i=0;i<n;i++)` | `for i in 0..n` **only** if the body never assigns `i` and `n` can't change during the loop. Otherwise use `while`. If `i` is used after the loop (`int i = 0; for(; i<cap; i++) {…break;}`), use `let mut i = 0; while i < cap { …; i += 1; }`. | `recursivelyRecomputeStats` |
| `do { … } while(c);` | `loop { …; if !(c) { break; } }`. A `continue` inside it jumps to the condition in C++: restructure carefully and note it. | `Rand::nextUInt(n)` |
| `x++` / assignment inside expressions | Split into statements in C++ sequencing order: `libs[numLibs++] = adj;` → `libs[num_libs as usize] = adj; num_libs += 1;`. `a[a_idx = (a_idx+1) & MASK]` → assign first, then index. If the C++ order is unspecified, `TODO(port)`. | `getNumLibertiesAfterPlay`, `XorShift1024Mult::nextUInt` |
| out-params `Loc& x`, `int& lo` | `x: &mut Loc`, `lo: &mut i32`; keep the return value too | `getBoundNumLibertiesAfterPlay` |
| default arguments | No default: every caller passes the value explicitly, with no builder or `Option`. **Exception:** a default that is a long literal repeated at many call sites becomes a `pub const` next to the fn, named `<FN>_DEFAULT_<PARAM>`, and callers pass the const. | `isEqualForTesting(other, true, true)`; `Global::trim(s, Global::TRIM_DEFAULT_DELIM)` |
| `[&]` lambda that captures state the enclosing fn also mutates | Rust forbids the aliasing. Make the lambda a closure or local `fn` that takes the captured state as explicit `&mut` parameters, in capture order, and pass them at every call. `// PORT NOTE: captures passed explicitly (aliasing)`. Keep the lambda's body and name. | `calculateAreaForPla` `buildRegion`; `initHash` `nextHash` |
| compound assignment on narrow types (`short += int`) | promote, compute, truncate, as C++ does: `x = (x as i32 + d) as i16;` | `num_liberties += delta` |
| unspecified argument evaluation order with side effects (`f(rand.next(), rand.next())`) | Match **GCC** (the fixture compiler), which evaluates arguments right to left: hoist into `let`s in that order + `// PORT NOTE: GCC arg order`. | `Board b(5+rand.nextUInt(14), 5+rand.nextUInt(14))` |
| `switch` with grouped/fallthrough cases | `match` with `A \| B \| C =>`. Real fallthrough (code then no `break`) → `TODO(port)`. Matching on `static const int` works via paths: `Rules::KO_SIMPLE \| Rules::KO_POSITIONAL => …`. `default: ASSERT_UNREACHABLE` → `_ => unreachable!()`. | `BoardHistory::newConsecutiveEndingPassesAfterPass` |
| `goto` | None in the ported files. If one appears, `TODO(port)`. | |
| constructor init list + body | `fn new(..) -> Self`: build the struct literal in init-list field order, then run the body statements on `let mut s`. | |
| destructor | Omit it if it only frees memory. Otherwise `impl Drop` with an `/// upstream: X::~X` tag. | `NNEvaluator::~NNEvaluator` (stops servers) |
| `operator==` with plain field semantics | `impl PartialEq` | `Rules` |
| free `bool operator<(a,b)` | `pub fn operator_lt(a: &T, b: &T) -> bool`, used via `port::cpp::cmp_from_lt` | `AnalysisData` |
| `std::stable_sort(v)` | `v.sort_by(port::cpp::cmp_from_lt(operator_lt))`: stable, same result | `getAnalysisData` |
| `std::sort(v, cmp)` | `v.sort_unstable_by(port::cpp::cmp_from_lt(cmp))` **only** if elements that compare equal are indistinguishable (true for `KoHashTable::recompute`: ties are identical hashes). Otherwise `TODO(port)` + a tie-order test. Never `partial_cmp().unwrap()` on floats. Since Rust 1.81 `sort_by` panics on a non-total order; a NaN in a key means `TODO(port)`. | |
| `std::map::operator[]` | **Inserts a default when the key is missing.** → `.entry(k).or_default()`, so a `shared_ptr` value type becomes `Option<Rc<T>>` to allow the inserted null. | `EvalCacheTable::update`, `SubtreeValueBiasTable::get` |
| nlohmann `json[key]` on a non-const object | Also inserts `null`. Mirror it with `entry(key).or_insert(Value::Null)` when the object is later dumped. | `analysis.cpp` |
| `std::min/max` on **floats** | `port::cpp::cpp_min/cpp_max`: `max(a,b) = if a < b {b} else {a}`, `min(a,b) = if b < a {b} else {a}`, NaN-propagating like C++. **Never `f64::max`**. On integers `std::cmp::min/max` is fine. `fmin/fmax` → `f64::min/max`. | `nneval.cpp` pass-policy cap |
| atomics | `.load(o)` → `.get() /*o*/`; `.store(v,o)` → `.set(v) /*o*/`; `fetch_add(d)` → `port::cpp::cell_fetch_add(&c, d)` (returns the old value); `exchange(v)` → `c.replace(v)`; `compare_exchange_strong/weak(exp, des)` → `port::cpp::cell_cas(&c, &mut exp, des) -> bool`, which writes the current value into `exp` on failure like C++ (weak never spuriously fails). | `SearchNode::maybeExpandChildrenCapacityForNewChild` |
| spin lock `while(x.test_and_set());` … `x.clear();` | Delete both lines and keep the body. | `addLeafValue` |
| `assert(e)` | `debug_assert!(e)`, **only if `e` has no side effects** (no calls that mutate, no `++`, no `=`). Otherwise hoist the side effect: `let r = e; debug_assert!(r);`. | |
| `std::fill`, `memset`, `memcpy`, `std::copy` | `slice.fill(v)`, `dst[..n].copy_from_slice(&src[..n])` | |

## 6. Numeric fidelity

Fixtures compare features **bit-exactly** and values to 1e-4. Most regressions in Bun's
port were in this category.

1. **Usual arithmetic conversions.** Put the casts exactly where C++ converts, at the operand,
   not on the result. `double winLogits = whiteWinProb * outputScaleMultiplier;` (float*float,
   computed in **float**, then widened) → `let win_logits: f64 = (a * b) as f64;`, **not**
   `a as f64 * b as f64`. int op float → float. int op double → double. float op double → double.
2. **Math functions follow the argument type** (C++ `<cmath>` overloads, `using namespace std`).
   `exp(policy[i] - maxPolicy)` with floats → `math::expf(p - m)` (**f32**). `sqrt(int)` →
   `(x as f64).sqrt()`. `round`/`floor`/`ceil` → same-named methods (`round` = half away from
   zero in both). `rint`/`nearbyint` → `round_ties_even`. `pow(10.0, n)` → `math::powf(10.0, n as f64)`,
   **never `powi`**. `isfinite`/`isnan` → `is_finite`/`is_nan`.
3. **All transcendental calls go through `port::math`; never call `f64::exp/ln/powf` (or any
   other std transcendental) directly in katago-core.** On wasm32 the std methods come from
   compiler-builtins' libm, which differs from glibc in the last ulp for 3–10% of inputs (that
   desyncs `Rand`'s accept/reject loops and every later draw). `port::math` reproduces the glibc
   2.44 x86-64/FMA results bit for bit on every target (Appendix). C `exp`/`log`/`pow`/`expf` →
   `math::exp`/`ln`/`powf`/`expf`; `atan` → `math::atan` (CORE-MATH's correctly rounded atan,
   plus glibc's 12 misroundings on ScoreValue's table arguments: identical on every target and to
   glibc on those tables, DECISIONS.md D13); `tan` → `math::tan` (platform libm, ≤ 1 ulp on wasm,
   D13; only model-version-3 nets reach it). `expm1` → `math::expm1`, `lgamma` → `math::lgamma` (`lgamma_r` for the sign), `tanhf` → `math::tanhf`, `log10` → `math::log10`. A C function it lacks (`tanh`, `atanf`, `logf`…) is a `TODO(port): needs port::math::<fn>`, not a std call.
   `crates/katago-core/clippy.toml` lists the std methods under `disallowed_methods`, which
   `lib.rs` denies. `sqrt`, `abs`, `floor`, `round`… are exact everywhere and stay std methods.
4. **No FMA.** Rust never contracts and upstream is built without `-march`/FMA. Don't use
   `mul_add`, except inside `port/math.rs`, which mirrors the fused operations of glibc's FMA build.
5. **Float→int casts.** C++ `(int)d` truncates and is UB out of range or for NaN. Rust `as`
   truncates and saturates (NaN→0). Use `as` when the value is known to be in range. If it can
   be out of range or NaN (anything derived from user input), use
   `port::cpp::static_cast_i32_f64`/`static_cast_i32_f32`, which give `i32::MIN` like x86-64
   `cvttsd2si`/`cvttss2si`, and add `// PORT NOTE: C++ UB; x86-64 gives INT_MIN`. `int64_t` targets work the same way with
   `port::cpp::static_cast_i64_f64`/`_f32` (`i64::MIN`); plain `as i64` only when provably in
   range, with `// PORT NOTE: in range because …`.
5b. **Printers too.** Every `(int)` of a float, including in debug printers, goes through `port::cpp::static_cast_*`, never `as i32`; arithmetic on the cast result (`cast + const`) uses `wrapping_add`.
6. **Integer overflow.** Signed overflow is UB in C++: use plain `+ - *` (debug builds panic,
   which catches bugs). Unsigned wraparound that upstream relies on uses `wrapping_*`.
   **Always** in hash, RNG and zobrist code (`Rand::nextUInt(n)`'s `bits - val + (n-1)`,
   PCG32's multiply, `Hash::simpleHash`). Shift amounts ≥ width are UB in C++ and panic in
   Rust: keep the upstream guards (`rot == 0 ? x : …`).
7. **`/` and `%`** truncate toward zero in both languages, and the remainder takes the
   dividend's sign. **Never `rem_euclid`/`div_euclid`.** `>>` on signed is arithmetic in both.
   Promote `short`/`int8_t` operands to `i32` before arithmetic, as C++ does.
8. **Signed/unsigned comparisons.** C++ converts the signed side to unsigned
   (`int x = -1; x < v.size()` is false). Reproduce it: `(x as usize) < v.len()`.
9. **Literal types.** `0.20f` → `0.20f32`, `1e-20f` → `1e-20f32`, `2.0` → f64. `1ULL << 53` → `1u64 << 53`.

## 6b. Undefined behaviour that upstream actually reaches

Some upstream code or tests hit UB (e.g. `isAdjacentToPla(PASS_LOC)` reads `colors[-9]` from `testboardbasic`'s replay test via `chooseRandomLegalMove`). Rule: if the UB is a **cast or arithmetic overflow** with deterministic x86-64/GCC behaviour, emulate it (`port::cpp::static_cast_*`, `wrapping_*`) with a PORT NOTE. If it is an **out-of-bounds or uninitialised read**, don't emulate memory: make the case explicit at the call site with the behaviour that keeps every *asserted* upstream result unchanged, add `// PORT NOTE: upstream UB (<what>)`, and list it in the manifest notes of the module. Evidence: an oracle build that runs the real UB call next to the substitute and counts disagreements (as `oracle/boardreplay_rng.cpp` does). That evidence holds for the fixture compiler and flags only.

## 7. Errors and assertions

| Upstream | Rust | Rule |
|---|---|---|
| `throw StringError(m)` and subclasses (`IOError`, `ValueError`, `ConfigParsingError`, `CommandError`) | `return Err(KataError::IOError(m))` etc. | A fn that throws, or calls a fallible fn without catching, returns `Result<T, KataError>`. This propagates transitively, including through `async` fns. Use `?`. A **throwing constructor** is `new_* -> Result<Self, KataError>`. |
| `try*` fn that catches everything its callees can throw (`Rules::tryParseRules`) | plain `bool`/value return, no `Result` | Only when every error the body can raise is caught there; otherwise `Result`. |
| `catch(const StringError&)` | `match … { Err(e) if e.is_string_error() => … }` | Mirror the C++ hierarchy: `is_string_error()` is true for all five variants, `is_io_error()` for `IOError` and `ConfigParsingError`. |
| `catch(nlohmann::detail::exception&)` | `Err(KataError::NlohmannJson(_))` | `NlohmannJson` is **not** a StringError subclass, so `catch(const StringError&)` must not catch it. `catch(std::exception&)`/`catch(...)` catches everything. Wrap the `try` body in a closure that returns `Result`. |
| a call to a throwing fn with constant, known-valid input and no `try` | `.expect("upstream: cannot throw for constant input")` + `// PORT NOTE` | `Rules::toStringNoKomiMaybeNice` → `parseRulesHelper("TrompTaylor", false)` |
| `Global::fatalError(s)` | `Global::fatal_error(s) -> !`, which panics with the upstream text | Fatal invariant. In wasm, a panic kills the worker and the client reports it. |
| `testAssert(e)` | `testAssert!(e)` (crate-root macro from `core/test.rs`) | Always on, like upstream, with upstream's failure message via `testAssertFailed`. Side effects are allowed. `#[macro_export]` macros (`testAssert!`, `strprintf!`) live at the crate root: `crate::testAssert!`. |
| `assert(e)` | `debug_assert!(e)` | Side-effect free only (§5). Upstream's default build (`RelWithDebInfo`) keeps asserts on, so run tests in debug. |
| `ASSERT_UNREACHABLE` | `unreachable!()` | |
| `unwrap()` on a value upstream treats as fallible | **forbidden** | Use `?`, or `expect` with a `TODO(port)` explaining why it can't fail. |

## 8. Statics and global init

- Runtime-initialized statics (`Board::ZOBRIST_*` tables filled by `initHash`, `ScoreValue`
  tables, the file statics of `subtreevaluebiastable.cpp`/`patternbonustable.cpp`) go into
  **one struct per class or file**, `<Class>Statics`, whose fields keep the upstream names
  (`#[allow(non_snake_case)]`). It lives in a `static <CLASS>_STATICS: OnceLock<<Class>Statics>`.
  The upstream init fn body becomes `fn compute_<…>() -> <Class>Statics`, with the RNG call
  order unchanged. The upstream init fn name forces the `OnceLock`. Readers call
  `Board::statics().ZOBRIST_BOARD_HASH[loc as usize][pla as usize]`. Concrete names are in
  OWNERSHIP.tsv.
- `IS_ZOBRIST_INITALIZED` / `isInited` / `initMutex` are replaced by the `OnceLock` state
  (PORT NOTE). `ScoreValue::freeTables` becomes a no-op and `initTables` is idempotent (PORT NOTE).
- Function-local `static`/`static constexpr` → a `const` inside the fn, or a fn-local `OnceLock`.

## 9. Strings, formatting, JSON

- **`Global::strprintf(fmt, …)`** → `strprintf!("%4d ", x)` (crate-root macro defined in `core/global.rs`, backed
  by `port::fmt::printf`). It implements C printf for `%d %i %u %ld %lld %zu %x %X %f %e %g %s %c %%`
  with flags `- + 0 space`, width and precision. **Keep the C format string verbatim.** `float`
  varargs are promoted to `double`. Ties (`%.2f` of 0.125) must follow glibc's exact-binary
  round-half-even: `port::fmt` has a test table against glibc output.
- **`ostream << double`** (default precision 6, `%g`) → `port::fmt::cpp_ostream_f64(x)`. For
  `float` it's the same after widening. `Global::doubleToString`/`floatToString` use it.
  `ss.precision(17)` → `port::fmt::cpp_ostream_f64_prec(x, 17)`. `std::to_string(double)` → `%f`.
  Integers and `Hash128` → `Display`. `endl` → `"\n"` (`writeln!` when it ends a `write!`).
- **`%d`/`%i` with an `int64_t` argument** prints the low 32 bits on x86-64 (varargs read an `int`): reproduce with `as i32`, PORT NOTE.
- **JSON output must equal nlohmann's `dump()` byte for byte.** The goldens prove it
  (`tests/results/analysis/*.stdout`): keys **sorted** (`"edgeVisits","edgeWeight","lcb","move",…`),
  compact separators, doubles printed nlohmann-style (`9.0`, `0.65662659`, `4.12549165e-05`,
  NaN/Inf → `null`).
  - Use `serde_json::Value` **without the `preserve_order` feature**. Its `Map` is then a `BTreeMap`,
    which sorts exactly like nlohmann's `std::map<std::string, …>`. Cargo unifies features
    workspace-wide, so CI runs `cargo tree -e features -i serde_json` and fails if
    `preserve_order` appears.
  - **Never `serde_json::to_string`** for protocol output. Use `port::nlohmann::dump(&Value)`,
    a port of nlohmann's serializer and its grisu2 `dtoa_impl` (MIT). It reproduces the exponent
    format (`e-05`), the trailing `.0`, and grisu2's digit choice, which `ryu` doesn't always match.
  - Store `f32` values as `x as f64` (nlohmann widens floats: `0.1f` dumps as `0.10000000149011612`).
  - Input: **always `port::nlohmann::parse`**, never `serde_json::from_str`. It is a port of
    nlohmann 3.8.0's lexer and iterative parser that builds a `serde_json::Value`: number classes
    as upstream (no `-` and fits u64 → u64; `-` and fits i64 → i64, so `-0` is integer 0 and dumps
    `0`; otherwise f64, correctly rounded like strtod), UTF-8 BOM skipped, a NUL byte outside a string
    ends the input, last duplicate key wins, no nesting limit, and `e.what()` texts byte-identical
    (`parse_error.101` with line/column/last read/expected, `out_of_range.406` for `1e400`). Oracle:
    `tests/nlohmann_parse.rs`. Known gap: `KataError` holds a `String`, so a `what()` that ends
    inside a multi-byte character (invalid non-ASCII input) gets U+FFFD there.
  - nlohmann `is_number_integer()` ≈ `Value::is_i64() || is_u64()`; `is_number()` ≈ `is_number()`.
    serde_json cannot tell nlohmann's integer `-0` from unsigned `0` (`is_number_unsigned()` differs
    only there). `get<int64_t>()` on a u64 > i64::MAX wraps in C++: `TODO(port)` if it matters.
    `get<int>()` of a float out of range or NaN is `INT_MIN` (x86-64 `cvttsd2si`), not Rust's saturation.
  - serde_json is built with `float_roundtrip` and without `preserve_order`; `just check-features` fails otherwise.

## 10. Crate map

| Upstream | Rust | Notes |
|---|---|---|
| zlib (`.bin.gz` model files) | `flate2` (pure-Rust `miniz_oxide` backend) | wasm-safe. |
| nlohmann json | `serde_json::Value` (no `preserve_order`, `float_roundtrip` on) + `port::nlohmann::{parse, dump}` | §9. |
| libm (`<cmath>` transcendentals) | `port::math` | §6 item 3. Never the std float methods. |
| `core/sha2.cpp` | `sha2` | `get256(char[65])` = **lowercase** hex. `get256(uint64_t[4])` = big-endian words: `u64::from_be_bytes(d[8i..8i+8])`. `uint32_t[8]` the same, with `from_be_bytes`. |
| `core/md5.cpp` | `md-5` | `MD5::get(.., uint32_t hash[4])` = the state words: `u32::from_le_bytes(d[4i..4i+4])`. |
| `core/rand.cpp` (`Rand` = PCG32 + XorShift1024Mult, MD5/SHA-256 string seeding) | **ported bit-exactly** in `core/rand.rs` + `core/rand_helpers.rs` | **Never** the `rand` crate. Unseeded `Rand::init()` (time/pid/hostname/stack address) → entropy string from `getrandom` (PORT NOTE). It is never used where outputs are compared. |
| Logger | `log` | §3. |
| `ClockTimer` | `core/timer.rs` with an injected clock (`set_clock(fn() -> f64)` once at startup) | No `Instant` in core. `ClockTimer::new()` keeps its signature. |
| tclap, `external/filesystem` (ghc), httplib, half, clblast | not ported | |
| threads, ThreadSafeQueue, MutexPool | `port::executor` (§11) | |
| async plumbing | `futures` (`default-features = false`, features `std`, `executor`, **no** `thread-pool`) | `LocalBoxFuture`, `join_all`, `poll_fn`, `LocalPool` for native tests/CLI. |

## 11. Concurrency → async virtual threads (plan §6)

Upstream runs `numSearchThreads` OS threads per search, plus NN server threads that batch
queued evals. The port runs **one OS thread**: each search thread becomes a future (a *virtual
thread*), all joined inside `runWholeSearch`, and each NN server thread becomes a task spawned
on the host's local executor. Every blocking wait becomes an `.await`. Everything else stays
synchronous and keeps its upstream code.

**Becomes `async` (the call chain from the playout loop down to the NN), and nothing else:**

| File | Functions |
|---|---|
| `neuralnet/nninterface.rs` | `NeuralNet::getOutput` (trait method; GPU readback is async) |
| `neuralnet/nneval.rs` | `NNEvaluator::evaluate` (both overloads), `averageMultipleSymmetries`, `waitForNextNNEvalIfAny`, `spawnServerThreads` (awaits the previous serve tasks' end, D8, then server start-up, `mainThreadWaitingForSpawn`), `serve` (the spawned server task) |
| `search/searchnnhelpers.rs` | `computeRootNNEvaluation`, `initNodeNNOutput`, `maybeRecomputeExistingNNOutput` |
| `search/search.rs` | `runSinglePlayout`, `playoutDescend`, `beginSearch`, `computeRootValues`, every `runWholeSearch*` / `runWholeSearchAndGetMove*` overload, the `searchLoop` lambda |
| `search/asyncbot.rs` | `genMoveSynchronous*`, `genMoveSynchronousAnalyze*`, `waitForSearchToEnd` (runs the queued job: there is no bot thread) and the analyze `callbackLoop` (a task `runWholeSearch` polls next to its search threads, `Search::run_whole_search_full`'s `concurrent_task`) |
| `program/setup.rs`, `program/playutils.rs`, `command/analysis.rs` | `initializeNNEvaluator(s)` (spawnServerThreads); `computeOwnership`, `maybeFriendlyPass`, `getFullSymmetryNNOutput`, ...; `MainCmds::analysis` with its request / analysis / write loops as joined virtual threads |

**Stays sync:** board/history/rules, `fillRowV7`, `selectBestChildToDescend` and all explore,
update and results helpers, `allocateOrFindNode`, `recursivelyRecomputeStats`, tree traversals,
`performTaskWithThreads` for non-search tasks, JSON output.

| Upstream construct | Port |
|---|---|
| `performTaskWithThreads(&searchLoop, capThreads)` | `futures::future::join_all` over `min(capThreads, searchParams.numThreads)` futures `search_loop(thread_idx)`. Each owns its `SearchThread`. Locals the threads share (`numPlayoutsShared`, `shouldStopNow`, `tcMaxTime`…) become `Cell`s borrowed by the futures (PORT NOTE). |
| `performTaskWithThreads(&g, …)` for tree tasks, `numAdditionalThreadsToUseForTasks()` | **`numAdditionalThreadsToUseForTasks()` returns 0** (PORT NOTE) so sharded tasks (`deleteAllTableNodesMulithreaded`) cover every shard on thread 0. `spawnThreadsIfNeeded`/`killThreads`/`respawnThreads` stay as empty-bodied fns with a PORT NOTE. That's a prescribed no-op, not a stub. |
| `std::this_thread::yield()` | `port::executor::yield_now().await` (returns `Pending` once and wakes itself) |
| `ThreadSafeQueue<NNResultBuf*> queryQueue` | `port::executor::AsyncQueue<Rc<NNResultBuf>>` with the same API names: `force_push`, `wait_pop_up_to_n(&mut buf, n).await -> bool`, `set_read_only`, `unset_read_only`, `close`. |
| `NNResultBuf` handshake (`resultMutex` + `clientWaitingForResult` + `hasResult`) | `SearchThread::nnResultBuf` is an `Rc<NNResultBuf>` whose fields are `Cell`/`RefCell` (TSV). `client_waiting_for_result: WakerList`. The client does `buf.client_waiting_for_result.wait_until(\|\| buf.has_result.get()).await` (= `while(!buf.hasResult) wait`). The server sets `has_result` then calls `notify_all()`. |
| `std::condition_variable cv` + `while(!cond) cv.wait(lock)` | `cv: WakerList`; `cv.wait_until(\|\| cond).await`; `notify_all()` wakes all of them. The mutex disappears. Counters it guarded become `Cell`. |
| server thread `serve(...)` | spawned via the injected `Rc<dyn Spawner>` (new NNEvaluator ctor param, PORT NOTE). The loop body is unchanged. `waitPopUpToN` returns whatever is queued when the server is polled. With FIFO scheduling that is every virtual thread blocked on the NN, which gives the plan's "flush when all blocked or batch full". `isKilled` + `setReadOnly()` end the loop. `killServerThreads` can't join: the join moves to the next `spawnServerThreads`, which awaits `num_server_threads_alive == 0` (WakerList `server_threads_exited`, notified by each ending task) before `isKilled = false`, the post-join `testAssert`s and `unsetReadOnly` (D8). |
| timing (`ClockTimer`) | the injected clock. `maxTime` and `reportDuringSearchEvery` work because the server awaits a real GPU readback (a macrotask). A CPU backend must yield to the event loop in its `get_output`: on wasm32 a macrotask (`katago_nn_burn::host::yield_to_event_loop`, a Promise resolved by `setTimeout(_, 0)`; test `katago-nn-burn/tests/wasm_yield.rs` under `just test-wasm`), never only `yield_now()` (a microtask under `wasm_bindgen_futures`: the event loop would not turn until the search ends). Natively `yield_now()`. |

Rules:
- **Never hold a `Ref`/`RefMut` (or a `RefCell` borrow) across `.await`.** katago-core sets
  `#![deny(clippy::await_holding_refcell_ref)]`. Holding an `Rc<SearchNode>`/`Rc<NNOutput>` is fine.
- **Re-read after await exactly where upstream re-loads an atomic.** Other virtual threads run
  at every await. Never cache a `Cell` value across an await unless upstream used a local copy too.
- **Virtual losses, the `SearchNode` state machine (`STATE_UNEVALUATED → EVALUATING →
  EXPANDED0…`), the "someone else got there first" CAS failures and the edge-visit
  catch-up are ported as is.** They now interleave only at awaits, which makes runs
  deterministic for a given thread count.
- **Lifecycle.** Spawned serve tasks hold an `Rc<NNEvaluator>`, so the evaluator can't be dropped while they live and `Drop` never runs `killServerThreads` by itself: the host calls `kill_server_threads()` explicitly before dropping. Upstream's post-join `testAssert`s run only once no serve task is alive: in `killServerThreads` if none is, else in the next `spawnServerThreads` (D8); `isKilled` stays true until then, as upstream's waiters see it during the join.
- **Futures on the eval path must not be cancelled.** An `evaluate` future dropped after its `force_push` leaves the `Rc<NNResultBuf>` in the queue (and the server later writes into it); the next `evaluate` with that buf would queue it twice (debug builds assert). Hosts that need to abort a search stop it through upstream's own flags (`shouldStopNow`, `killServerThreads`) and let the futures finish, never by dropping them.
- **A backend's `get_output` reads its inputs before its first await.** Everything it takes from the `input_bufs` (features, `symmetry`, `policyOptimism`) is copied or snapshotted before awaiting, as upstream reads them under the batch; after the await it only writes `outputs`.
- Recursive async fns return a boxed future: `fn playout_descend<'a>(&'a self, thread: &'a mut
  SearchThread, node: &'a SearchNode, is_root: bool) -> LocalBoxFuture<'a, Result<bool, KataError>>
  { Box::pin(async move { … }) }`, marked `// PERF(port): one allocation per depth`.
- `Search` methods on the playout path take `&self`. `runWholeSearch(&mut self)` mutates before
  the join (`beginSearch`), reborrows `&*self` for the joined futures, then mutates after.
- Parity fixtures use `numSearchThreads = 1`, `nnRandomize = false` and `forDeterministicTesting`
  (see `tests/results/analysis/*.stderr`). With one virtual thread the port must match upstream
  playout for playout.

## 11b. Lints

The fidelity rules trigger lints that are not bugs here. `crates/katago-core/src/lib.rs` carries one crate-level allow list so real warnings stand out; add to it, never sprinkle per-item allows for these:

```rust
#![allow(non_snake_case, non_upper_case_globals, non_camel_case_types)]
#![allow(clippy::collapsible_if, clippy::collapsible_else_if, clippy::needless_range_loop,
         clippy::manual_range_contains, clippy::new_without_default, clippy::too_many_arguments,
         clippy::inherent_to_string_shadow_display, clippy::manual_swap, clippy::nonminimal_bool,
         clippy::needless_return, clippy::comparison_chain, clippy::if_same_then_else,
         clippy::field_reassign_with_default)]
// upstream literals (twoOverPi) trip the deny-by-default approx_constant; the rest are
// upstream statement shapes kept verbatim
#![allow(clippy::approx_constant, clippy::excessive_precision, clippy::assign_op_pattern,
         clippy::bool_comparison, clippy::needless_late_init, clippy::int_plus_one,
         clippy::manual_clamp, clippy::neg_cmp_op_on_partial_ord, clippy::inherent_to_string,
         clippy::wrong_self_convention)]
// M3: upstream literals (0x3fffFFFF), the mirrored path search::search, and kept statement shapes
#![allow(clippy::mixed_case_hex_literals, clippy::module_inception, clippy::unnecessary_unwrap,
         clippy::should_implement_trait, clippy::ptr_arg, clippy::identity_op, clippy::borrow_deref_ref,
         clippy::diverging_sub_expression)]
#![deny(clippy::await_holding_refcell_ref)]
```

Test crates don't inherit `lib.rs` attributes: each file in `tests/` that replays upstream code carries its own `#![allow(…)]` with the subset it needs (`bool_comparison`, `needless_late_init`, `manual_memcpy`… are common in transcribed tests).

## 12. Forbidden

- In katago-core: `std::thread`, `std::sync::{Mutex, RwLock, Condvar, atomic::*}` (only
  `OnceLock` is allowed), `tokio`, `rayon`, `async-std`, `std::time::{Instant, SystemTime}`,
  file or network IO.
- `anyhow`. The `rand` crate. Changing any RNG, hash, zobrist seed string, constant or table
  size that affects outputs.
- The ndarray crate and burn's ndarray backend (CPU is `flex`, CLAUDE.md).
- Refactoring for idiom: iterator chains replacing loops, merging, splitting or reordering
  functions, early-return rewrites, `Option` combinators replacing upstream `if`s, newtype
  wrappers beyond those listed here.
- **Stubbing a fn body to make it compile** (`todo!()`, `unimplemented!()`, `Default::default()`,
  empty body). Port what you can and put `TODO(port)` at the gap. The only allowed empty bodies
  are the PORT NOTE no-ops prescribed in §8/§11.
- `unwrap()`/`expect()` on upstream-fallible paths without `TODO(port)` (§7). `f64::max/min`
  where C++ uses `std::max/min` (§5). `serde_json::to_string` for protocol output (§9).
- `#[ignore]` on a test without a `TODO(port)` and a manifest note. Editing fixtures by hand.
- Touching `upstream/`.

## 13. Testing (Phase C)

Oracles, from strictest to loosest. Fixtures live in `fixtures/<tag>/` and are never hand-edited.

| Tier | Oracle | Comparison |
|---|---|---|
| Unit (M1) | `R"%%(…)%%"` expected blocks in `cpp/tests/test*.cpp`, extracted by `xtask extract-tests`. The Rust `#[test]` replays the same calls in the same order and prints with the ported printers (`Board::printBoard`, `BoardHistory::printDebugInfo`, `NNOutput::debugPrint`). That's why printers are ported exactly. | string-equal after `TestCommon::expect`-style trimming |
| Features (M2) | `build/eigen/katago` + the dump patch: `rowSpatial`/`rowGlobal`/hashes for ~2k positions | bit-exact |
| NN (M2) | the upstream NNEvaluator on the Eigen build (`nn/oracle/nneval_eigen.cpp`, agrees with `kata-raw-nn`) | after post-processing: flex ≤1e-5 policy/winrate, ≤2e-5 ownership (tanh of final-conv accumulation-order noise, measured: no bias, mean 2e-7), ≤5e-4 score lead/mean; wgpu ≤2e-4 logits / ≤1e-4 winrate, ownership |
| Search (M3) | `cpp/tests/analysis/*.txt` queries through the ported analysis path. Primary oracle: our **Eigen** rerun under `runcmdtests.sh` overrides. Secondary: upstream `tests/results/analysis/*.stdout`, which were produced on CUDA with b6c96 (see `.stderr`), so allow float tolerance. | identical moves, order and visits; values ±1e-4; JSON key order and format exact |
| Corpus (M3) | deterministic Eigen `katago analysis`, b10c128, 100 visits | ≥99% same top move, ≥95% identical visit vectors |

**Comparisons.** Unit blocks use `TestCommon::expect` (upstream's per-line trim). Oracle-program outputs are compared **byte-exact** by `tests/common::compare_lines` (line endings and trailing newline included). `tests/fixture_coverage.rs` fails if an extracted expected block is consumed by no test and isn't on its allowlist with a reason.

**Async tests** drive pools through `tests/common::run_or_stall_panic`, which panics when the pool stalls while the future is pending and unwoken, so a lost wakeup fails fast instead of hanging.

**Search oracles (M3).** `tests/testsearchnonn.rs` replays upstream `runNNLessSearchTests` (debugSkipNeuralNet: backend-independent, byte-exact). Upstream's `Logger` lines (`: ` prefix) are part of upstream test stdout; tests that compare such output install a `log::Log` that writes `": " + msg` into the same thread-local buffer as the test's own `cout` text, and run the pool until stalled after `kill_server_threads()` where upstream's `delete nnEval` joins (the serve task logs `GPU 0 finishing...` as it ends). `tests/search_oracle.rs` is a differential oracle against `oracle/search_nonn.cpp` (op-stream cases, floats as hex bits, 1 search thread) plus a Rust-only multi-virtual-thread determinism/invariant test. Real-net searches are compared with tolerances by `katago-nn-burn/tests/search_smoke.rs` (`just search-smoke`, release only). Edition 2024 reserves `gen`: don't name test helpers `gen`.

**Random-game generators** for oracles include out-of-turn (board-legal) moves, odd×even board sizes, finished games where pass dominates, and hand-built encore ko fights: plain legal play never reaches those states.

**Feature sets.** katago-nn-burn's default features include all three nets (b6c96, b10c128, b18c384nbt); wasm test builds use `--no-default-features -F flex,net-b6c96`.

**wasm stack.** Debug wasm has a 1 MiB stack, and in debug every temporary of a `BoardHistory` (~34 KB) or `Board` (~4.7 KB) gets its own slot, so boxing at the call site isn't enough: build boxed values in a small helper fn (`boxed_history` in `tests/timecontrols.rs`), and init statics from a small frame. Test macros from `tests/common` are invoked as `crate::report!`.

**Oracle link lines.** Oracle programs link upstream's `core`, `game`, `neuralnet`, `search`, `dataio` and `program` object files from `build/eigen` only, never `command/*.o` or `main.cpp.o` (they need `Version::getGitRevision`). Generators must avoid inputs that trip upstream asserts (e.g. a button ruleset with a nonzero starting `encorePhase`: `BoardHistory::clear` reads the old phase for `hasButton`).

**Fixture layout.** `fixtures/<tag>/unit/<test-file-stem>/<test-slug>.<input|expected><k>.txt` plus an `INDEX.tsv`, produced by `tools/extract_raw_blocks.py` (stand-in for `xtask extract-tests`). C++ oracle programs that print values from upstream code live next to their outputs, in `fixtures/<tag>/unit/oracle/`, with the build command in their header.

**Audit that tests actually ran:**
- Every fixture-driven test asserts it loaded at least 1 case and prints how many it ran
  (`eprintln!("board_tests: 37 cases")`). An empty glob fails.
- Reports quote the `cargo test` summary line (passed / failed / ignored / filtered). "Tests
  pass" without counts doesn't count. A drop in case count is a regression.
- Tests run in the debug profile (debug assertions on, as in upstream's default build).
  Parity runs also run once with `--release`, to catch overflow-wrap differences.
- Oracle tests whose results depend on float math also run on wasm32 (`just test-wasm`, node +
  wasm-bindgen-test-runner): give them `#[cfg_attr(not(target_arch = "wasm32"), test)]` and
  `#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]` and add them to
  the recipe. `tests/common::fixture` reads fixtures through node's `fs` there.
- Reviewers spot-check one fixture by hand: change one expected byte locally and confirm the
  test fails.

**Injected host services.** Core does no IO and reads no clock. File access that upstream does through `FileUtils` (config `@include`, `initialize(fname)`, `overrideKeys(fname)`) goes through a reader callback the host installs (`ConfigParser::set_include_reader`), which receives the path exactly as `FileUtils::open` would and fails with `FileUtils::open`'s error text when absent (D15). Time comes from `core::timer::set_clock(fn() -> f64)`: seconds as `f64`, first call wins (returns `false` after), and a `ClockTimer` with no clock installed is a fatal error, so hosts and search tests install one first. Timed waits (AsyncBot's `reportDuringSearchEvery` callback loop, upstream `condition_variable::wait_for`) go through `port::executor::set_sleep`, installed the same way (katago-cli: a timer thread; wasm: a `setTimeout` Promise). NN backends, model bytes and the NN server executor reach `Setup::initializeNNEvaluator(s)` through the `NNEvaluatorHost` trait (program/setup.rs); `MainCmds::analysis` gets its argv, input lines, stdout/stderr sinks and that host through `AnalysisHost` (command/analysis.rs).

**Test-local copies.** A test that needs an upstream helper from a later milestone (e.g. `PlayUtils::chooseRandomLegalMove`, M3) may carry a faithful copy in its test file, tagged `/// upstream: PlayUtils::chooseRandomLegalMove (playutils.cpp) — test-local copy until M3`. Delete it when the module lands. A function skipped because it does IO or reads stdin (`Global::pauseForKey`) gets no Rust fn; the manifest `notes` say so.

## 14. Re-sync after an upstream tag bump

1. Bump the submodule to the new tag in a branch. Run `python3 -I tools/manifest_check.py`:
   it fails on new, removed or unmapped upstream files. Map them in `port-manifest.toml`.
2. Run `cargo xtask upstream-diff --to <tag>`. It produces a per-module `git diff --stat <synced>..<tag>`
   and the changed C++ functions (tree-sitter), resolved to Rust fns through the `/// upstream:` tags.
3. Re-port **only the listed functions**, with the same agent loop (§2), following this
   guide. New functions get tags. Deleted functions are deleted. A renamed upstream fn is
   renamed in Rust.
4. Regenerate params and fixtures: `xtask gen-params` (fails on new `SearchParams` keys),
   `xtask fixtures --upstream <tag>`, `xtask extract-tests`. Commit the fixtures in their own commit.
5. Run Phase C. For each module that is green again, `xtask bump-synced <module>`: set
   `synced` to the new commit, and update `status`/`notes` (new divergences, skipped fns).
6. If a rule in this guide changed (a new upstream idiom), update PORTING.md and
   OWNERSHIP.tsv **first**, then re-port.

## Appendix: `crates/katago-core/src/port/` (port-only helpers, binding names)

Implementers don't edit `port/`: a missing helper is `TODO(port): needs port::<name>` in the report. Fixers whose task explicitly asks for a helper add it, with a unit test and a row in this table.

| Item | Signature / meaning |
|---|---|
| `port::cpp::cpp_min`, `cpp_max` | `fn<T: PartialOrd + Copy>(a: T, b: T) -> T`, with C++ `std::min/max` semantics |
| `port::cpp::cmp_from_lt` | `fn(lt: impl Fn(&T,&T)->bool) -> impl Fn(&T,&T)->Ordering` |
| `port::cpp::cell_cas`, `cell_fetch_add` | `fn<T: Copy+PartialEq>(c: &Cell<T>, expected: &mut T, desired: T) -> bool`; `fn<T: Copy+Add<Output=T>>(c: &Cell<T>, d: T) -> T` |
| `port::fmt` | `printf`, `printf_bytes`, `cpp_ostream_f64`, `cpp_ostream_f64_prec`, `cpp_to_string_f64`, `PrintfArg` (keeps C width and signedness so `%u`/`%x` of an int is right; `char`/`short` promote to `int`). `printf_bytes` has C byte semantics (`%c` one raw byte, `%.Ns` may cut a UTF-8 character, `%s` stops at NUL); `printf` is its `String` form, with U+FFFD where the bytes are not UTF-8 |
| `port::math::{exp, ln, powf, expf}` | C `exp`, `log`, `pow` (double) and `expf`, bit-identical to glibc 2.44 on x86-64 with FMA (`__exp_fma`…), on every target. Ported from ARM optimized-routines (MIT) with glibc's FMA contractions as explicit `mul_add`. Oracle: `tests/math.rs` vs `oracle/math.cpp` (also under `just test-wasm`). `port::math::atan`: C `atan`, port of CORE-MATH's correctly rounded `cr_atan` (MIT, math/atan.rs; also exported as `port::math::cr_atan`) plus `GLIBC_ATAN_OVERRIDES`, glibc 2.44's 12 misroundings (`__atan_fma`) among the 50,531 ScoreValue table arguments: identical on every target, bit-identical to glibc on those tables, 1 ulp from glibc elsewhere wherever glibc is not correctly rounded (D13); oracle `tests/math_atan.rs` vs `oracle/math_atan.cpp` (MPFR + glibc; also under `just test-wasm`). `port::math::tan` wraps the platform libm (≤ 1 ulp from glibc on wasm, D13; only model-version-3 nets reach it; oracle `oracle/math_ulp.txt`). `port::math::tanhf`: C `tanhf`, bit-identical to glibc 2.44 (CORE-MATH, correctly rounded) on every target; our double-double evaluation + one rounding, checked against glibc on all 2^32 inputs; oracle `tests/math_tanhf.rs` vs `oracle/math_tanhf.cpp` (also under `just test-wasm`). Not provided yet (TODO(port) list in the file): `tanh`, `atanf`, float `logf`/`powf` |
| `port::math::log10` | C `log10`: port of glibc 2.44's fdlibm `__ieee754_log10` (no FMA variant on x86-64) over `port::math::ln`, bit-identical on every target. Oracle: `tests/math_log10.rs` vs `oracle/math_log10.txt` (2^20 inputs + every power of ten and its neighbours; also under `just test-wasm`). Used by `Global::roundDynamic` |
| `port::math::{lgamma, lgamma_r, expm1}` | C `lgamma` (ln\|Γ(x)\|), `lgamma_r(x, &mut signgam)` (also the sign of Γ(x), what glibc stores in `signgam`), `expm1`: bit-identical to glibc 2.44 (x86-64, FMA) on every target. `lgamma`: port of glibc's CORE-MATH code (MIT, correctly rounded) in `port/math/lgamma.rs`. `expm1`: fdlibm `s_expm1.c` as built into `__expm1_fma`, GCC's FMA contractions as explicit `mul_add`. Oracle: `tests/math_lgamma.rs` vs `oracle/math_lgamma.cpp` (also under `just test-wasm`); 1.2·10^9 inputs each matched the host glibc |
| `port::cpp::static_cast_i32_f64`, `static_cast_i32_f32` | `(int)x` with x86-64 `cvtts*2si` results: truncation in range, `i32::MIN` for NaN/out of range (§6 item 5) |
| `port::cpp::static_cast_i64_f64`, `static_cast_i64_f32` | same for `int64_t` targets: `i64::MIN` for NaN/out of range |
| `port::cpp::c_str` | `fn(&str) -> &str`: cut at the first NUL, the `const char*` view of a string |
| `port::cpp::istream_parse_{i32,i64,u64,hex_u64,f32,f64}_whole` | `istream >>` semantics for `Global::tryStringTo*` |
| `port::digest` | sha2/md5 glue: C-string NUL cut-off, big-endian u64 words, hex, `get512` |
| `port::nlohmann::{parse, sax_parse, Sax, Exception}` | `json::parse(s)` / `json::sax_parse(s, &sax)`: port of nlohmann 3.8.0's lexer + parser (§9 Input); errors are `KataError::NlohmannJson(e.what())` |
| `port::nlohmann::{get_string, get_bool, get_f32, get_i32, get_i64, get_f64, get_vector_i32, get_vector_i64, is_number_integer, static_cast_i32_f64}` | nlohmann `get<T>` semantics (e.g. `get<float>`/`get<int>` accept booleans, `get<int64_t>`/`get<double>` do not; `get<int>` of an out-of-range/NaN float is `INT_MIN`, `get<int64_t>` of a u64 above i64::MAX wraps; `get<std::vector<T>>` needs an array); errors are `KataError::NlohmannJson` |
| `port::nlohmann::dump` | `fn(&serde_json::Value) -> String`, equal to `nlohmann::json::dump()` |
| `port::node_arena::{NodeIdx, NodeArena<T>}` | §4. Generic over the node type (`Search::node_arena: NodeArena<SearchNode>`): `alloc(\|idx\| T) -> NodeIdx`, `get(idx) -> Rc<T>` (panics on a freed index), `free(idx)` (the index may be reused, like freed memory; held `Rc`s stay valid), `with_mut(idx, f)` (panics if shared), plus `is_allocated` / `num_allocated` for tests. Unit tests in the module |
| `port::executor` | `yield_now`, `WakerList { wait_until, notify_all }`, `AsyncQueue<T>` (`ThreadSafeQueue` names: `force_push`, `try_pop`, async `wait_pop` (`waitPop`, `None` = false), async `wait_pop_up_to_n`, `set_read_only`, `unset_read_only`, `close`, `reserve`, `size`; no `maxSize`/`waitPush`, unused on ported paths; plus `any(f)`, for debug assertions only), `AsyncPriorityQueue<K, V>` (`ThreadSafePriorityQueue`: max-heap on the key, equal keys in insertion order; `force_push(key, value)`, `try_pop`, async `wait_pop`, `set_read_only`, `close`, `size`), `set_sleep(fn(f64) -> LocalBoxFuture<'static, ()>)` / `sleep(secs)` (the host's timed wait, first install wins, fatal if missing; AsyncBot's analyze-callback loop), `trait Spawner { fn spawn_local(&self, f: LocalBoxFuture<'static, ()>) }` (implemented for `futures::executor::LocalSpawner`), re-export `LocalBoxFuture`. Unit tests in the module |

## Appendix B: name overrides (win over the overload rules)

| Upstream | Rust |
|---|---|
| `Rules::Rules(int kRule, int sRule, int tRule, bool suic, bool button, int whbRule, bool pOk, float km)` | `Rules::new_full` |
| `Location::toStringMach(Loc, const Board&)` | `Location::to_string_mach_b` (rule 3 result, kept) |
| `TestCommon::expect(const char*, std::ostringstream&, …)` | `TestCommon::expect_string` |
| `BoardHistory::BoardHistory(const Board&, Player, const Rules&, int encorePhase)` | `BoardHistory::new_full` |
| `Global::mapGet(const std::map<std::string,V>&, const char*)` (specialisation with its own message) | `Global::map_get_str` |
| `Global::contains(const std::vector<A>&, const A&)` / `(const std::set<A>&, …)` / `(const std::map<A,B>&, …)` | `contains_vector` / `contains_set` / `contains_map` |
| `Global::concat(const char*, int len)` / `concat(const std::set<std::string>&, …)` | `concat_len` / `concat_set` |
| `ReportedSearchValues::ReportedSearchValues(const Search&, double winLossValue, …)` | `ReportedSearchValues::new_search` |
| `AnalysisData::move` (field; `move` is a Rust keyword) | `AnalysisData::move_` (also used for `Loc move` parameters in searchresults.rs) |
| `PrintTreeOptions` fields `maxDepth_` … `alsoBranch_` / builder methods `maxDepth(int)` … | fields keep the trailing underscore (`max_depth_`, `branch_`, `also_branch_`), methods take the plain names (`max_depth(..)`) |
| `Search::getPlaySelectionValues(vector<Loc>&, vector<double>&, double)` | `get_play_selection_values` |
| `Search::getPlaySelectionValues(vector<Loc>&, vector<double>&, vector<double>*, double)` | `get_play_selection_values_ret_visit_counts` |
| `Search::getPlaySelectionValues(const SearchNode&, …, bool allowDirectPolicyMoves)` | `get_play_selection_values_node` |
| `Search::getPlaySelectionValues(const SearchNode&, …, bool alwaysComputeLcb, bool neverUseLcb, double lcbBuf[], double radiusBuf[])` | `get_play_selection_values_node_lcb` |
| `Search::getPolicy(const SearchNode*, float[])` | `get_policy_node` |
| `Search::getPolicySurpriseAndEntropy(double&, double&, double&, const SearchNode*)` | `get_policy_surprise_and_entropy_node` |
| `Search::printPV(ostream&, const vector<Loc>&)` | `print_pv_buf` |
| `Search::getAnalysisData(const SearchNode&, vector<AnalysisData>&, …)` | `get_analysis_data_node` |
| `Search::getAverageTreeOwnership(const Player, const SearchNode*, int)` | `get_average_tree_ownership_perspective` |
| `Search::getAverageAndStandardDeviationTreeOwnership(const Player, const SearchNode*, int)` | `get_average_and_standard_deviation_tree_ownership_perspective` |
| `Search::Search(params, nnEval, logger, randSeed)` / `(params, nnEval, humanEval, logger, randSeed)` | `Search::new` / `Search::new_human_eval` (Result) |
| `Search::runWholeSearch(Player)` / `(Player, std::function<bool()>*)` / `(Player, bool pondering)` / `(Player, bool, std::function<bool()>*)` | `run_whole_search` / `run_whole_search_should_stop_early` / `run_whole_search_pondering` / `run_whole_search_pondering_should_stop_early` |
| `Search::runWholeSearch(const std::function<void()>* searchBegun, std::function<bool()>* shouldStopEarly, bool pondering, const TimeControls&, double searchFactor)` | `run_whole_search_full` |
| `Search::runWholeSearchAndGetMove(Player, bool pondering)` | `run_whole_search_and_get_move_pondering` |
| `Search::makeMove(Loc, Player, bool preventEncore)` | `make_move_prevent_encore` |
| `SearchNode::getChildren(SearchNodeState)` | `get_children_state` |
| `NodeStats::NodeStats(const NodeStatsAtomic&)` / `NodeStatsAtomic::NodeStatsAtomic(const NodeStatsAtomic&)` | `NodeStats::new_other` / `NodeStatsAtomic::new_copy` |
| `NodeStatsAtomic::getChildWeight(int64_t, int64_t)` / `getChildWeightSq(int64_t, int64_t)` | `get_child_weight_child_visits` / `get_child_weight_sq_child_visits` |
| `Setup::loadParams(ConfigParser&, setup_for_t, bool, bool)` | `Setup::load_params_full` |
| `AsyncBot::genMoveSynchronous(Player, const TimeControls&, double, const std::function<void()>&)` | `gen_move_synchronous_full` |
| `AsyncBot::genMoveSynchronousAnalyze(..., callback, const std::function<void()>& onSearchBegun)` | `gen_move_synchronous_analyze_full` |
| `AsyncBot::genMoveAsync(Player, int, const TimeControls&, double, onMove, onSearchBegun)` / `genMoveAsyncAnalyze(..., callback, onSearchBegun)` | `gen_move_async_full` / `gen_move_async_analyze_full` (private) |
| the `searchLoop` lambda of `Search::runWholeSearch` | private `async fn search_loop(&self, thread_idx, &SearchLoopCaptures)` (its by-reference captures as one struct of `&Cell`s) |
