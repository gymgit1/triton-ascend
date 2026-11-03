// === #2686 Rule A (SelectedBlockScanRule, bit 65536) — POSITIVE case ===
//
// STATUS: post-implementation form; every RUN line below is green.
//
// Two things are checked here, against two different artifacts:
//
//  A. The DESIGN-TARGET assertions (FileCheck on Inputs/nsa_fwd_ruleA_design_target.mlir):
//     that fixture is the compiled gold standard (the Rule-A-only reference
//     implementation run through the early TTIR pipeline), re-printed once by
//     `triton-opt <file> -o <new>` without --mlir-print-debuginfo.  It is
//     therefore the SAME IR WITH THE LOCATIONS DROPPED -- not a byte-for-byte
//     copy of the reference artifact: every #loc alias is gone, and with it the
//     SSA value names the name locs used to supply (%q, %k_5, ... are now
//     %arg0, %6, ...).  The op stream, the types and the module attribute are
//     unchanged.  Keeping an implementation-independent artifact in the tree is
//     what makes the CHECK lines below verifiable against something other than
//     the rule itself; otherwise every assertion about the rewritten form is
//     circular.
//
//  B. The IMPLEMENTATION assertions (`-graph-optimize='rule-mask=65536
//     selected-block-scan-force-apply=true'`): Rule A must reproduce form A on
//     the REAL kernel IR.  Landed as (1) bit + phase + registry, (2) escape
//     hatch in Passes.td, (3) factory/matcher, (4) rewrite body.
//
// What the fixture is: the REAL NSA forward kernel IR (nsa_fwd.py) after the early
// TTIR pipeline, i.e. exactly what graph-optimize sees in production
// (third_party/ascend/backend/compiler.py:309-327).  Shapes G=16, K=V=128, S=16,
// BS=64, bf16; constant trip count (NS = S = 16); online-softmax (Form 1:
// iter_args o/m/l).  See Inputs/nsa_fwd_const_ns.mlir for provenance.
//
// Why rule-mask=65536 and nothing else: attribution isolation, the same contract
// that convert_modulo_to_mask.mlir:8-9 states in prose.
// nsa_selected_block_scan_attribution.mlir proves that today's 3071 rules do not
// touch this fixture.
//
// Escape hatch: `selected-block-scan-force-apply` is the test-only Passes.td bool
// that short-circuits the joint Rule A + Rule B gate INSIDE the plan factory
// (never inside apply(): GraphOptimizationPass.cpp:359-365 turns an apply()
// failure into signalPassFailure).  Without it Rule A legitimately produces no
// plan while Rule B's matcher is a stub, so the rewrite form would be untestable.
//
// --- D3/D4 settled (2026-10-31): both ACCEPTED ------------------------------
// D3 (trip count from a scalar tt.load(block_counts)): ACCEPTED as a
//   non-constant upper bound -> Inputs/nsa_fwd_dyn_ns.mlir is a positive case,
//   group D below, check prefix DYNREW.
// D4 (i64 induction variable for an otherwise identical constant-trip-count
//   loop): ACCEPTED -> Inputs/nsa_fwd_const_ns_i64_iv.mlir is a positive case,
//   group E below, reusing the REWRITTEN fingerprint verbatim.
// Rule A additive (settled 2026-11-02): the bwd dq kernel's pure-additive
//   carried-state form is ACCEPTED -> Inputs/nsa_bwd_dq_const_ns.mlir, group G
//   below, prefix BWDREW.  That fixture used to be a negative case (section (g)
//   of nsa_selected_block_scan_negative.mlir), which is now deleted.
// The D3/D4 segments used to live in nsa_selected_block_scan_decisions.mlir;
// with both decisions "accept" that file was dissolved (its rejection-side
// diff-empty assertions would contradict the implementation) and the cases
// moved here, so the decisions file is not part of this directory.

// --- A. design target: the CHECK lines below, checked without the rule -----
// RUN: triton-opt %S/Inputs/nsa_fwd_ruleA_design_target.mlir -o %t.design
// RUN: FileCheck %s --check-prefix=REWRITTEN --implicit-check-not='scf.if' < %t.design
// RUN: FileCheck %s --check-prefix=NOSCALAR < %t.design

// --- B. the implementation, with the escape hatch --------------------------
// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns.mlir -graph-optimize='rule-mask=65536 selected-block-scan-force-apply=true' -canonicalize -cse -o %t
// RUN: FileCheck %s --check-prefix=REWRITTEN --implicit-check-not='scf.if' < %t
// RUN: FileCheck %s --check-prefix=NOSCALAR < %t
// (meta) the rewrite must actually change the IR: a green FileCheck on an
// unchanged module would be a false positive.  `diff` exits 1 when the files
// differ, so `not diff` asserts "they differ" (repo precedent for `not`:
// DiscreteMaskAccess/loop_mask_unsupported.mlir:1-2; repo precedent for the
// diff-empty idiom: row-coalescing-dot-fallback.mlir:1-7).
// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns.mlir -graph-optimize='rule-mask=0' -canonicalize -cse -o %t.control
// RUN: not diff %t.control %t
// (meta) idempotency: applying the rule to its own output must be a no-op
// (row-coalescing-dot-basic.mlir:4-5 does the same).
// RUN: triton-opt %t -graph-optimize='rule-mask=65536 selected-block-scan-force-apply=true' -canonicalize -cse -o %t.twice
// RUN: diff %t %t.twice

// --- Rule A has no separate Rule B gate ------------------------------------
// Rule A always runs Rule B's real implementation (the QueryRowBatching.h
// materialize entry) inside its own sandbox and lets Rule B decide, so bit
// 65536 alone must produce exactly the same IR as 65536|131072: the 131072 bit
// only decides whether Rule B is ALSO scheduled as a standalone rule, where it
// finds nothing left to lift.  (Rule A is default-on now: bits 16/17 join the
// default mask, 199679 = 3071|65536|131072, see
// nsa_selected_block_scan_attribution.mlir for the baseline-vs-default split.)
// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns.mlir -graph-optimize='rule-mask=196608' -canonicalize -cse -o %t.joint
// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns.mlir -graph-optimize='rule-mask=65536' -canonicalize -cse -o %t.gated
// RUN: diff %t.joint %t.gated
// ... and the joint rewrite must still differ from the rule-off run, so that
// equality cannot pass vacuously:
// RUN: not diff %t.control %t.gated

// --- D. D3 accepted: runtime trip count from block_counts is REWRITTEN -------
// Decision D3 (settled 2026-10-31): v0 accepts a trip count that comes from a
// scalar tt.load (block_counts) outside the loop, so Inputs/nsa_fwd_dyn_ns.mlir
// moved from the dissolved decisions file into the positive set.  The
// block_counts load itself survives in the prologue (the weight mask still
// consumes it) BEFORE the table vector read, so this case cannot reuse the
// REWRITTEN prefix verbatim: REWRITTEN-NOT for `tt.load ... : !tt.ptr<i64>`
// spans [func entry, table vector read] and would match the legitimate NS
// load.  DYNREW below is the same fingerprint with that one NOT moved below
// the vector read.
// RUN: triton-opt %S/Inputs/nsa_fwd_dyn_ns.mlir -graph-optimize='rule-mask=0' -canonicalize -cse -o %t.dyns.ctl
// RUN: triton-opt %S/Inputs/nsa_fwd_dyn_ns.mlir -graph-optimize='rule-mask=65536 selected-block-scan-force-apply=true' -canonicalize -cse -o %t.dyns
// RUN: FileCheck %s --check-prefix=DYNREW --implicit-check-not='scf.if' < %t.dyns
// RUN: FileCheck %s --check-prefix=NOSCALAR < %t.dyns
// RUN: not diff %t.dyns.ctl %t.dyns
// RUN: triton-opt %t.dyns -graph-optimize='rule-mask=65536 selected-block-scan-force-apply=true' -canonicalize -cse -o %t.dyns.twice
// RUN: diff %t.dyns %t.dyns.twice

// --- E. D4 accepted: i64 induction variable is REWRITTEN ---------------------
// Decision D4 (settled 2026-10-31): the matcher accepts an i64 induction
// variable for an otherwise identical constant-trip-count loop (the frontend
// emits i64 as soon as the bound is i64; "i32 only" would be a silent
// narrowing).  No extra scalar load survives the rewrite, so the case reuses
// the REWRITTEN fingerprint verbatim.
// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns_i64_iv.mlir -graph-optimize='rule-mask=0' -canonicalize -cse -o %t.i64iv.ctl
// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns_i64_iv.mlir -graph-optimize='rule-mask=65536 selected-block-scan-force-apply=true' -canonicalize -cse -o %t.i64iv
// RUN: FileCheck %s --check-prefix=REWRITTEN < %t.i64iv
// RUN: FileCheck %s --check-prefix=NOSCALAR < %t.i64iv
// RUN: not diff %t.i64iv.ctl %t.i64iv
// RUN: triton-opt %t.i64iv -graph-optimize='rule-mask=65536 selected-block-scan-force-apply=true' -canonicalize -cse -o %t.i64iv.twice
// RUN: diff %t.i64iv %t.i64iv.twice

// --- F. R = BN/BS > 1: BS = 32 => BN = 64, R = 2 (production main path) -----
// Every other fixture in this directory has BS = 64, so BN = max(64, BS) = 64
// and the rewrite takes the R = BN/BS = 1 shape.  With BS < 64 one index-table
// entry covers R > 1 BN-wide tiles, so the per-token weight count has to be
// evaluated per sub-tile (R x S) and reshaped back to BN tokens -- a distinct
// code path, and the one the online deployment actually runs.  Fixture:
// Inputs/nsa_fwd_const_ns_bs32.mlir, the same REAL kernel IR as
// nsa_fwd_const_ns.mlir with the single documented mutation BS 64 -> 32.
// The R2REW fingerprints below are exactly what R = 1 cannot produce: count
// reduce tensor<2x16xi32> -> tensor<2xi32>, broadcast to tensor<2x32xf32>,
// reshape tensor<2x32xf32> -> tensor<64xf32>, slot index 2*i and the BS tensor
// dense<32> : tensor<2xi32>.  Everything else must stay shared with R = 1:
// BN is still 64, so the K/V tiles are still tensor<128x64xbf16> /
// tensor<64x128xbf16> and both dots keep M = 16.
// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns_bs32.mlir -graph-optimize='rule-mask=0' -canonicalize -cse -o %t.bs32.ctl
// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns_bs32.mlir -graph-optimize='rule-mask=65536 selected-block-scan-force-apply=true' -canonicalize -cse -o %t.bs32
// RUN: FileCheck %s --check-prefix=R2REW --implicit-check-not='scf.if' < %t.bs32
// RUN: FileCheck %s --check-prefix=NOSCALAR < %t.bs32
// RUN: not diff %t.bs32.ctl %t.bs32
// RUN: triton-opt %t.bs32 -graph-optimize='rule-mask=65536 selected-block-scan-force-apply=true' -canonicalize -cse -o %t.bs32.twice
// RUN: diff %t.bs32 %t.bs32.twice

// --- G. bwd dq additive form: ACCEPTED and REWRITTEN (settled 2026-11-02) ----
// The bwd dq kernel (Inputs/nsa_bwd_dq_const_ns.mlir) carries its state as ONE
// tensor<16x128xf32> accumulator with neutral 0 instead of the o/m/l triple,
// loads lse/delta once outside the loop, keeps token arithmetic in i64 and
// addresses K/V with tt.addptr plus an explicit mask (never tt.make_tensor_ptr).
// Rule A used to DECLINE that shape -- it was section (g) of
// nsa_selected_block_scan_negative.mlir, i.e. the fixture used to pin a refusal.
// The additive branch (matcher classification by the number of iter_args,
// matchAddPtrTile, and the exp -> where -> *w -> (do v - delta) -> truncf ->
// dot(ds, trans(k), acc) builder) landed and was accepted on 2026-11-02, so the
// fixture moves HERE: (g) is deleted from the negative file and BWDREW below is
// the fingerprint of the rewritten additive form.  BWDREW re-asserts what
// REWRITTEN pins (no scf.if, no scalar per-iteration index read, vector table
// read, divsi scan bound, one loop, weight-count reduce, three dots with
// M = 16) and adds the three things only this branch has: the single carried
// accumulator that the third dot consumes, the surviving tt.trans, and the
// addptr tile whose mask keeps the `tokens < T` predicate.
// RUN: triton-opt %S/Inputs/nsa_bwd_dq_const_ns.mlir -graph-optimize='rule-mask=0' -canonicalize -cse -o %t.bwd.ctl
// RUN: triton-opt %S/Inputs/nsa_bwd_dq_const_ns.mlir -graph-optimize='rule-mask=65536 selected-block-scan-force-apply=true' -canonicalize -cse -o %t.bwd
// RUN: FileCheck %s --check-prefix=BWDREW --implicit-check-not='scf.if' < %t.bwd
// (meta) the OP COUNTS, as a second FileCheck pass over the same prefix with the
// ops whose count is part of the contract as implicit NOTs: `scf.for` (exactly
// ONE, so a nested/duplicated loop cannot hide in a gap between two positive
// matches) and `tt.dot` (exactly the THREE the fingerprint matches, so a fourth
// dot is caught).  A single pass pins the sequence, this pass pins the counts.
// RUN: FileCheck %s --check-prefix=BWDREW --implicit-check-not='scf.for' --implicit-check-not='tt.dot' < %t.bwd
// RUN: not diff %t.bwd.ctl %t.bwd
// RUN: echo -n > %t.empty
// RUN: triton-opt %S/Inputs/nsa_bwd_dq_const_ns.mlir -graph-optimize='rule-mask=65536 selected-block-scan-force-apply=true' -canonicalize -cse -o /dev/null 2> %t.bwd.err
// RUN: diff %t.empty %t.bwd.err
// RUN: triton-opt %t.bwd -graph-optimize='rule-mask=65536 selected-block-scan-force-apply=true' -canonicalize -cse -o %t.bwd.twice
// RUN: diff %t.bwd %t.bwd.twice

// --- target-gate sensitivity (decision settled 2026-10-31: NO arch gate) ----
// Rule A deliberately does not gate on hacc::utils::isAscend950: the real
// frontend writes "Ascend910_95", which that helper rejects, so an arch gate
// would make the rule silently never fire on production kernels.  The rewrite
// must therefore fire under the frontend's real spelling exactly like it does
// under the fixture's 950PR spelling, i.e. the REWRITTEN fingerprint must hold
// for the a595 variant too.
// RUN: sed 's/Ascend950PR_9579/Ascend910_95/' %S/Inputs/nsa_fwd_const_ns.mlir > %t.a595.mlir
// RUN: triton-opt %t.a595.mlir -graph-optimize='rule-mask=65536 selected-block-scan-force-apply=true' -canonicalize -cse -o %t.a595
// RUN: FileCheck %s --check-prefix=REWRITTEN --implicit-check-not='scf.if' < %t.a595

// ---------------------------------------------------------------------------
// REWRITTEN — the shape the rewrite must produce.  The design target
// (Inputs/nsa_fwd_ruleA_design_target.mlir) is the compiled gold standard with
// the locations dropped, so group A checks this fingerprint WITHOUT running the
// rule at all.  The assertions anchor semantic fingerprints only (vector table
// read / scan upper bound / one loop with 3 iter_args / weight reduce / safe_m /
// both dots keeping M = 16), never implementation detail.
// ---------------------------------------------------------------------------
// REWRITTEN-LABEL: tt.func public @parallel_nsa_fwd_kernel
// 负向 1：循环里不许再有分支；负向 2：不许再有"每轮读一个标量索引"；
// 负向 3：块级 i32 谓词必须消失（token 级 tensor<64xi32> 的 sle 允许保留）
// REWRITTEN-NOT: scf.if
// REWRITTEN-NOT: tt.load %{{.*}} : !tt.ptr<i64>
// REWRITTEN-NOT: arith.cmpi sle, %{{.*}}, %{{.*}} : i32
// 正向 1：整张索引表一次读成向量
// REWRITTEN: tt.load %{{.*}} : tensor<16x!tt.ptr<i64>>
// 正向 2：扫描上界来自 cdiv(i_t + 1, BN)
// REWRITTEN: arith.divsi %{{.*}}, %{{.*}} : i32
// 正向 3：仍是 1 个 scf.for、仍是 3 个 iter_args (o, m, l)
// REWRITTEN: %{{.*}}:3 = scf.for {{.*}} iter_args(
// 正向 4：权重 reduce（沿 S = 16 那条轴）
// REWRITTEN: "tt.reduce"(%{{.*}}) <{axis = 1 : i32}>
// 正向 5：第一个 dot 的 M 仍是 16（没被行合并摊大）
// REWRITTEN: tt.dot %{{.*}} : tensor<16x128xbf16> * tensor<128x64xbf16> -> tensor<16x64xf32>
// 正向 6：safe_m 守卫在位
// REWRITTEN: arith.cmpf oeq, %{{.*}}, %{{.*}} : tensor<16xf32>
// 正向 7：第二个 dot 的 M 仍是 16
// REWRITTEN: tt.dot %{{.*}} : tensor<16x64xbf16> * tensor<64x128xbf16> -> tensor<16x128xf32>

// ---------------------------------------------------------------------------
// NOSCALAR — the three properties Rule B's matcher needs, expressed as "nothing
// forbidden from the loop onwards".  The NOT region starts at the `scf.for` match
// and runs to end of file (the loop is the last construct: in the after-IR and in
// any sane rewrite the only thing after it is the epilogue + tt.return).
// This also holds for the D3 case (group D): its surviving block_counts load
// sits BEFORE the first scf.for, outside the forbidden region.
// ---------------------------------------------------------------------------
// NOSCALAR-NOT: hacc.program_grid_transforms
// NOSCALAR-NOT: hacc.coalesce_factor
// NOSCALAR-LABEL: tt.func public @parallel_nsa_fwd_kernel(
// NOSCALAR: scf.for
// NOSCALAR-NOT: : !tt.ptr<i64>
// NOSCALAR-NOT: scf.if
// NOSCALAR-NOT: scf.for

// ---------------------------------------------------------------------------
// DYNREW — the REWRITTEN fingerprint for the D3 case (Inputs/nsa_fwd_dyn_ns.mlir,
// group D).  Identical to REWRITTEN except for ONE placement: the block_counts
// load (`%NS = tt.load ... : !tt.ptr<i64>`) legitimately survives OUTSIDE the
// loop and BEFORE the table vector read (the weight mask still consumes it), so
// a `NOT` for the scalar i64 load anchored at the function label would match it
// and fail.  That NOT is moved below the vector read instead; the other two
// negatives and all seven positives are byte-identical to REWRITTEN.
// ---------------------------------------------------------------------------
// DYNREW-LABEL: tt.func public @parallel_nsa_fwd_kernel
// DYNREW-NOT: scf.if
// DYNREW-NOT: arith.cmpi sle, %{{.*}}, %{{.*}} : i32
// DYNREW: tt.load %{{.*}} : tensor<16x!tt.ptr<i64>>
// DYNREW-NOT: tt.load %{{.*}} : !tt.ptr<i64>
// DYNREW: arith.divsi %{{.*}}, %{{.*}} : i32
// DYNREW: %{{.*}}:3 = scf.for {{.*}} iter_args(
// DYNREW: "tt.reduce"(%{{.*}}) <{axis = 1 : i32}>
// DYNREW: tt.dot %{{.*}} : tensor<16x128xbf16> * tensor<128x64xbf16> -> tensor<16x64xf32>
// DYNREW: arith.cmpf oeq, %{{.*}}, %{{.*}} : tensor<16xf32>
// DYNREW: tt.dot %{{.*}} : tensor<16x64xbf16> * tensor<64x128xbf16> -> tensor<16x128xf32>

// ---------------------------------------------------------------------------
// R2REW — the fingerprint of the R = BN/BS = 2 path (Inputs/nsa_fwd_const_ns_bs32.mlir,
// group F).  Same contract as REWRITTEN, one extra requirement: the weight count
// and its broadcast must carry the R axis (2 x S = 2x16 -> 2, then 2x32 -> 64).
// The R = 1 fixtures produce tensor<1x16xi32> -> tensor<1xi32> and
// tensor<1x64xf32> -> tensor<64xf32> instead, so these lines fail on them.
// ---------------------------------------------------------------------------
// R2REW-LABEL: tt.func public @parallel_nsa_fwd_kernel(
// 正向 1：BS 本身作为 R 个 slot 的尺度常量出现（R = 1 时没有这个 tensor）。
// 它是函数入口常量，所以是本组第一条正向 CHECK；三条负向 CHECK 放在它之后，让 NOT 区间
// 覆盖 [入口常量, 表向量读) 的整段 prologue。
// R2REW: %{{.*}} = arith.constant dense<32> : tensor<2xi32>
// 负向 1/2/3：与 REWRITTEN 相同的三条禁止
// R2REW-NOT: scf.if
// R2REW-NOT: tt.load %{{.*}} : !tt.ptr<i64>
// R2REW-NOT: arith.cmpi sle, %{{.*}}, %{{.*}} : i32
// 正向 2：整张索引表一次读成向量
// R2REW: tt.load %{{.*}} : tensor<16x!tt.ptr<i64>>
// 正向 3：扫描上界来自 cdiv(i_t + 1, BN)
// R2REW: arith.divsi %{{.*}}, %{{.*}} : i32
// 正向 4：R 轴在进入循环之前就出现（make_range{0,R} 与两条 R 轴广播；R = 1 时是 tensor<1xi32>）
// 注意顺序：这三条都是循环不变量，流水线里的 canonicalize/CSE 会把它们提到 scf.for 之前
// R2REW: tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
// R2REW: tt.broadcast %{{.*}} : tensor<1x16xi32> -> tensor<2x16xi32>
// R2REW: tt.broadcast %{{.*}} : tensor<1x16xi1> -> tensor<2x16xi1>
// 正向 5：仍是 1 个 scf.for、仍是 3 个 iter_args，且 K/V tile 仍是 BN = 64 宽
// R2REW: %{{.*}}:3 = scf.for {{.*}} iter_args(
// R2REW: tt.make_tensor_ptr %{{.*}}, [%c128_i64, %{{.*}}], [%c1_i64, %c512_i64], [%c0_i32, %{{.*}}] {order = array<i32: 0, 1>} : <tensor<128x64xbf16>>
// R2REW: tt.make_tensor_ptr %{{.*}}, [%{{.*}}, %c128_i64], [%c512_i64, %c1_i64], [%{{.*}}, %{{.*}}] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
// 正向 6：循环内的 slot 索引是 2*i（R = 1 时就是 i 本身），splat 成 [R] 后再乘 BS = 32
// R2REW: arith.muli %{{.*}}, %c2_i32 : i32
// R2REW: tt.splat %{{.*}} : i32 -> tensor<2xi32>
// R2REW: arith.muli %{{.*}}, %{{.*}} : tensor<2xi32>
// 正向 7：R x S 的权重计数 reduce（R = 1 时是 1x16 -> 1）
// R2REW: tt.broadcast %{{.*}} : tensor<2x1xi32> -> tensor<2x16xi32>
// R2REW: "tt.reduce"(%{{.*}}) <{axis = 1 : i32}>
// R2REW: }) : (tensor<2x16xi32>) -> tensor<2xi32>
// 正向 8：计数广播回 BN 个 token，再 reshape 成 token 序（R = 1 时是 1x64）
// R2REW: tt.broadcast %{{.*}} : tensor<2x1xf32> -> tensor<2x32xf32>
// R2REW: tt.reshape %{{.*}} : tensor<2x32xf32> -> tensor<64xf32>
// R2REW: arith.cmpf ogt, %{{.*}}, %{{.*}} : tensor<64xf32>
// 正向 9/10：两个 dot 的 M 仍是 16
// R2REW: tt.dot %{{.*}} : tensor<16x128xbf16> * tensor<128x64xbf16> -> tensor<16x64xf32>
// 正向 11：safe_m 守卫在位
// R2REW: arith.cmpf oeq, %{{.*}}, %{{.*}} : tensor<16xf32>
// 正向 12：第二个 dot 的 M 仍是 16
// R2REW: tt.dot %{{.*}} : tensor<16x64xbf16> * tensor<64x128xbf16> -> tensor<16x128xf32>

// ---------------------------------------------------------------------------
// BWDREW — the fingerprint of the bwd dq additive rewrite (group G,
// Inputs/nsa_bwd_dq_const_ns.mlir).  Every line below was written against a
// dumped product of the binary, not against the rule's source.  It re-asserts
// REWRITTEN's contract and adds three additive-specific facts:
//   * carried state: exactly ONE iter_arg (the dq accumulator tensor<16x128xf32>,
//     neutral 0) instead of the o/m/l triple, and the third dot takes that
//     iter_arg as its C operand (`dot(ds, trans(k), acc)` is the latch);
//   * tt.trans survives as the third dot's B operand (the fwd form has none);
//   * K/V are read through tt.addptr + an explicit mask, so the tile load keeps
//     the `tokens < T` predicate (the pre-rewrite file has no tt.make_tensor_ptr
//     for the scan rule to reuse).
// NUMERIC ORDER is asserted, not just described: `where` (arith.select on the
// broadcast valid predicate) is applied to exp(s - lse) BEFORE the weight count
// is multiplied in.  An empty index slot has weight 0 and lse = -inf, so masking
// after the multiply would evaluate inf * 0 = nan; the `arith.select` ->
// `arith.mulf %[[SEL]], %[[WB]]` pair pins both the order and the fact that the
// where result is the value that gets scaled.  Note this order is pinned
// deliberately as "select, then multiply", and the weight multiply's operands
// are pinned as (where-result, weight) in that order.
// ---------------------------------------------------------------------------
// BWDREW-LABEL: tt.func public @parallel_nsa_bwd_kernel_dq(
// 负向 1：循环里不许再有分支
// BWDREW-NOT: scf.if
// 负向 2：不许再有"每轮读一个标量索引"
// BWDREW-NOT: tt.load %{{.*}} : !tt.ptr<i64>
// 正向 1：整张索引表一次读成向量（与 REWRITTEN 相同）
// BWDREW: tt.load %{{.*}} : tensor<16x!tt.ptr<i64>>
// 正向 2：扫描上界来自 cdiv(pid + BN, BN)，并且就是循环的 upper bound
// BWDREW: %[[BOUND:.*]] = arith.divsi %{{.*}}, %c64_i32 : i32
// 正向 3：仍是 1 个 scf.for，iter_args 恰好 1 个（additive accumulator）；括号里
// 用 [^,)] 而不是 .*，两个 iter_args 就匹配不上
// BWDREW: %{{.*}} = scf.for %{{.*}} = %c0_i32 to %[[BOUND]] step %c1_i32 iter_args(%[[ACC:[^,]*]] = %{{.*}}) -> (tensor<16x128xf32>)
// 正向 4：权重计数 reduce 沿 S = 16 归约，类型仍是 1x16 -> 1（R = 1 路径）
// BWDREW: "tt.reduce"(%{{.*}}) <{axis = 1 : i32}>
// BWDREW: }) : (tensor<1x16xi32>) -> tensor<1xi32>
// 正向 5：计数 reshape 回 token 序
// BWDREW: %[[WCNT:.*]] = tt.reshape %{{.*}} : tensor<1x64xf32> -> tensor<64xf32>
// 正向 6：valid = (w > 0) & (token <= i_t)（w > 0 是严格大于，空槽位才被排除）
// BWDREW: %[[POSW:.*]] = arith.cmpf ogt, %{{.*}}, %{{.*}} : tensor<64xf32>
// BWDREW: %[[POST:.*]] = arith.cmpi sle, %{{.*}}, %{{.*}} : tensor<64xi32>
// BWDREW: %[[VALID:.*]] = arith.andi %[[POSW]], %[[POST]] : tensor<64xi1>
// 正向 7：`tokens < T` 的 load mask 仍在（i64 比较 + 广播 + andi 进 load 的 mask）
// BWDREW: arith.cmpi slt, %{{.*}}, %{{.*}} : tensor<64xi64>
// BWDREW: %[[TBC:.*]] = tt.broadcast %{{.*}} : tensor<1x64xi1> -> tensor<128x64xi1>
// BWDREW: %[[TMASK:.*]] = arith.andi %{{.*}}, %[[TBC]] : tensor<128x64xi1>
// BWDREW: tt.load %{{.*}}, %[[TMASK]], %{{.*}} : tensor<128x64x!tt.ptr<bf16>>
// 正向 8：第一个 dot 的 M 仍是 16（q.k^T 仍是 16x128 x 128x64）
// BWDREW: tt.dot %{{.*}} : tensor<16x128xbf16> * tensor<128x64xbf16> -> tensor<16x64xf32>
// 正向 9：valid 广播成 where 的谓词
// BWDREW: %[[VEXP:.*]] = tt.expand_dims %[[VALID]] {axis = 0 : i32} : tensor<64xi1> -> tensor<1x64xi1>
// BWDREW: %[[PBC:.*]] = tt.broadcast %[[VEXP]] : tensor<1x64xi1> -> tensor<16x64xi1>
// 正向 10：先 where（select 把无效槽位置 0）
// BWDREW: %[[SEL:.*]] = arith.select %[[PBC]], %{{.*}}, %{{.*}} : tensor<16x64xi1>, tensor<16x64xf32>
// 正向 11：后乘权重——权重来自 reduce 计数（reshape -> expand_dims -> broadcast），
// 且乘法的被乘数是 where 的结果，不是 exp 的裸输出
// BWDREW: %[[WEXP:.*]] = tt.expand_dims %[[WCNT]] {axis = 0 : i32} : tensor<64xf32> -> tensor<1x64xf32>
// BWDREW: %[[WB:.*]] = tt.broadcast %[[WEXP]] : tensor<1x64xf32> -> tensor<16x64xf32>
// BWDREW: arith.mulf %[[SEL]], %[[WB]] : tensor<16x64xf32>
// 正向 12：第二个 dot 的 M 仍是 16（do.k^T）
// BWDREW: tt.dot %{{.*}} : tensor<16x128xbf16> * tensor<128x64xbf16> -> tensor<16x64xf32>
// 正向 13：tt.trans 仍在（k 的转置仍作为第三个 dot 的 B 操作数）
// BWDREW: %[[KT:.*]] = tt.trans %{{.*}} {order = array<i32: 1, 0>} : tensor<128x64xbf16> -> tensor<64x128xbf16>
// 正向 14：第三个 dot 仍是 16x64 x 64x128 且 C 操作数就是循环的 iter_arg（additive 状态）
// BWDREW: tt.dot %{{.*}}, %[[KT]], %[[ACC]] : tensor<16x64xbf16> * tensor<64x128xbf16> -> tensor<16x128xf32>
// 正向 15：唯一的 dq store 仍是 addptr tile 上的 masked store（16x128）
// BWDREW: tt.store %{{.*}}, %{{.*}}, %{{.*}} : tensor<16x128x!tt.ptr<bf16>>
// BWDREW: tt.return

// ---------- varlen: Rule A recognizes the dispatch (VARLENREW) ---------------
// (2026-11-03) The varlen dispatch keeps the SAME scan skeleton: the block-index
// table is still one row per query, read once; only the row position and the
// K/V base move from "program id" to "loop-invariant scalar load"
// (token_indices_q / cu_seqlens). Rule A accepts that spelling now, so with the
// escape hatch open the rewrite must land on the real varlen IR too.
// The joint gate is what keeps it OFF in production (Rule B declines varlen, and
// A alone would trade O(NS) block reads for an O(TK) token scan); that side is
// asserted in nsa_selected_block_scan_negative.mlir, section (a).
//
// RUN: triton-opt %S/Inputs/nsa_fwd_varlen.mlir -graph-optimize='rule-mask=65536 selected-block-scan-force-apply=true' -canonicalize -cse -o %t.varlenrew
// RUN: FileCheck %s --check-prefix=VARLENREW < %t.varlenrew

// VARLENREW-LABEL: tt.func public @parallel_nsa_fwd_kernel(
// 1. the index table is read ONCE as a vector, with the varlen prefix loads kept
// VARLENREW: tt.load %{{.*}} : tensor<16x!tt.ptr<i64>>
// 2. slot_valid = offs < NS over the padded slot axis (tensor<16xi32>)
// VARLENREW: arith.cmpi slt, %{{.*}}, %{{.*}} : tensor<16xi32>
// 3. the scan bound is cdiv(row_position + BN, BN) with a runtime row position
// VARLENREW: %[[BND:.*]] = arith.divsi %{{.*}}, %c64_i32 : i32
// 4. exactly one loop, i32 induction variable, three carried values (o/m/l)
// VARLENREW: scf.for %{{.*}} = %c0_i32 to %[[BND]] step %c1_i32 iter_args(
// 5. no per-block branch survives (the block guard was the only scf.if here)
// VARLENREW-NOT: scf.if
// 6. K/V tiles: one BN-wide tile each, read unconditionally
// VARLENREW: tt.load %{{.*}} {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<128x64xbf16>>
// VARLENREW: tt.load %{{.*}} {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x128xbf16>>
// 7. the new per-token weight reduce over the slot axis (1x16 -> 1x1)
// VARLENREW: "tt.reduce"(
// 8. valid = (weight > 0) & (token <= row_position) & (token < TK)
// VARLENREW: arith.cmpf ogt, %{{.*}}, %{{.*}} : tensor<64xf32>
// VARLENREW: arith.cmpi sle, %{{.*}}, %{{.*}} : tensor<64xi32>
// VARLENREW: arith.cmpi slt, %{{.*}}, %{{.*}} : tensor<64xi32>
// 9. softmax max and sum are still 2-D over BN
// VARLENREW: "tt.reduce"(
// VARLENREW: "tt.reduce"(
// 10. one masked output store (and the lse store) leave the function
// VARLENREW: tt.store
// VARLENREW: tt.store
// VARLENREW: tt.return

// ---------- joint golden standard: Rule A + Rule B (JOINTREW) ---------------
// (2026-11-03) The joint flavour is the launch-contract one: the rewrite commits
// hacc.program_grid_transforms (the launcher divides the grid by the row factor)
// and the kernel keeps "program id == query row". The in-kernel flavour (flat
// program id + `if active`) that the hand-written reference `nsa_fwd_new.py`
// uses is a *different* plumbing; SelectedBlockScanRule must never emit both
// (see the mutual-exclusion check in the rule and 15-v3-realignment.md 0.1).
//
// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns.mlir -graph-optimize='rule-mask=196608' -o %t.joint
// RUN: FileCheck %s --check-prefix=JOINTREW < %t.joint

// JOINTREW: hacc.program_grid_transforms = {extent_source = "runtime_original_grid"
// the joint function grows the hidden original-grid arguments Rule B appends
// JOINTREW: tt.func public @parallel_nsa_fwd_kernel(
// no per-block branch survives: the scan loop is the only loop
// JOINTREW-NOT: scf.if
// JOINTREW: scf.for
// the per-token weight reduce is present (axis 2, the lifted row axis), so A ran
// JOINTREW: "tt.reduce"(%{{.*}}) <{axis = 2 : i32}>
// and the rewritten body still ends with the masked stores
// JOINTREW: tt.store
// JOINTREW: tt.return

// ---------- S = 17 (non power-of-two table width) (S17REW) ------------------
// The slot axis must be padded to next_pow2(S) = 32 and the slot index clamped
// to S-1 so the table read stays inside its row; `tt.make_range` must never be
// asked for a 17-wide range.
//
// RUN: triton-opt %S/Inputs/nsa_fwd_s17.mlir -graph-optimize='rule-mask=65536 selected-block-scan-force-apply=true' -canonicalize -cse -o %t.s17
// RUN: FileCheck %s --check-prefix=S17REW < %t.s17

// S17REW-LABEL: tt.func public @parallel_nsa_fwd_kernel(
// padded slot range (32, not 17)
// S17REW: tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
// the clamp that keeps the read inside the row
// S17REW: arith.minsi %{{.*}}, %{{.*}} : tensor<32xi32>
// the vectorised table read covers the PADDED width
// S17REW: tt.load %{{.*}} : tensor<32x!tt.ptr<i64>>
// and the per-block branch is gone
// S17REW-NOT: scf.if
// S17REW: scf.for

// ---------- bwd dq dyn NS: folded index address (BWDYNREW / BWDYNJOINT) -----
// (2026-11-03) The dyn-NS spelling of the bwd dq kernel folds the row advance
// into the in-loop index address:
//     tt.addptr %root, arith.addi(%rowOffset, %iv)
// The dense kernels keep it as a separate SSA value instead:
//     tt.addptr %rowBase, %iv      (%rowBase = tt.addptr %root, %rowOffset)
// The address is the same, but matchIndexTable used to insist that the offset be
// exactly the induction variable, so this spelling was rejected -- which left the
// block_counts production path of the bwd dq kernel silently unmatchable.
// Inputs/nsa_bwd_dq_dyn_ns.mlir is exactly that IR.  The rewrite re-creates the
// row base the vectorised read needs (tt.addptr %root, %rowOffset) in the
// prologue, so both spellings land on the same scan form.
// RUN: triton-opt %S/Inputs/nsa_bwd_dq_dyn_ns.mlir -graph-optimize='rule-mask=0' -canonicalize -cse -o %t.bwdyndyn.ctl
// RUN: triton-opt %S/Inputs/nsa_bwd_dq_dyn_ns.mlir -graph-optimize='rule-mask=65536 selected-block-scan-force-apply=true' -canonicalize -cse -o %t.bwdyndyn
// RUN: FileCheck %s --check-prefix=BWDYNREW < %t.bwdyndyn
// RUN: not diff %t.bwdyndyn.ctl %t.bwdyndyn
// RUN: triton-opt %t.bwdyndyn -graph-optimize='rule-mask=65536 selected-block-scan-force-apply=true' -canonicalize -cse -o %t.bwdyndyn.twice
// RUN: diff %t.bwdyndyn %t.bwdyndyn.twice

// the vectorised table read, the single additive accumulator and the one reduce
// BWDYNREW-LABEL: tt.func public @parallel_nsa_bwd_kernel_dq(
// BWDYNREW-NOT: scf.if
// BWDYNREW: tt.load %{{.*}} : tensor<16x!tt.ptr<i64>>
// BWDYNREW: scf.for %{{.*}} = %c0_i32 to %{{.*}} step %c1_i32 iter_args(%{{.*}} = %{{.*}}) -> (tensor<16x128xf32>)
// BWDYNREW: "tt.reduce"(%{{.*}}) <{axis = 1 : i32}>

// the joint gate accepts the same shape and publishes the runtime grid contract
// RUN: triton-opt %S/Inputs/nsa_bwd_dq_dyn_ns.mlir -graph-optimize='rule-mask=196608' -o %t.bwdyndyn.joint
// RUN: FileCheck %s --check-prefix=BWDYNJOINT < %t.bwdyndyn.joint

// BWDYNJOINT: hacc.program_grid_transforms = {extent_source = "runtime_original_grid"
// BWDYNJOINT: tt.func public @parallel_nsa_bwd_kernel_dq(
// BWDYNJOINT-NOT: scf.if
// BWDYNJOINT: scf.for
// BWDYNJOINT: "tt.reduce"(%{{.*}}) <{axis = 2 : i32}>
// BWDYNJOINT: tt.store
// BWDYNJOINT: tt.return

// ---------- varlen + block_counts: 7 loop-invariant scalar loads (VARLENBCREW)
// (2026-11-03) The production varlen dispatch also passes block_counts, so the
// loop-invariant scalar loads before the loop are 2 token_indices_q + 2
// cu_seqlens_q + 2 cu_seqlens_k + 1 block_counts = 7, not the 6 of the pure
// varlen prefix.  The skeleton budget used to be six and rejected the whole
// dispatch before matchIndexTable ran.  Inputs/nsa_fwd_varlen_bc.mlir is that IR.
// RUN: triton-opt %S/Inputs/nsa_fwd_varlen_bc.mlir -graph-optimize='rule-mask=0' -canonicalize -cse -o %t.varlenbc.ctl
// RUN: triton-opt %S/Inputs/nsa_fwd_varlen_bc.mlir -graph-optimize='rule-mask=65536 selected-block-scan-force-apply=true' -canonicalize -cse -o %t.varlenbc
// RUN: FileCheck %s --check-prefix=VARLENBCREW < %t.varlenbc
// RUN: not diff %t.varlenbc.ctl %t.varlenbc

// VARLENBCREW-LABEL: tt.func public @parallel_nsa_fwd_kernel(
// VARLENBCREW-NOT: scf.if
// VARLENBCREW: tt.load %{{.*}} : tensor<16x!tt.ptr<i64>>
// VARLENBCREW: %{{.*}}:3 = scf.for
// VARLENBCREW: "tt.reduce"(%{{.*}}) <{axis = 1 : i32}>

// (meta) the joint gate must still refuse it: Rule B does not lift varlen, so
// offering only bit 65536 must leave the IR untouched.
// RUN: triton-opt %S/Inputs/nsa_fwd_varlen_bc.mlir -graph-optimize='rule-mask=65536' -canonicalize -cse -o %t.varlenbc.gated
// RUN: diff %t.varlenbc.ctl %t.varlenbc.gated

// ---------- H == 1 + dyn NS: table length from the row stride (H1DYNREW) -----
// (2026-11-03) With H == 1 the head split folds away, so the row base loses its
// i_h*S addend and the row stride of block_indices is H*S == S.  The table
// length used to be recoverable only from a CONSTANT trip count, so the dyn-NS
// (block_counts) H == 1 decode shape never matched.  The matcher now reads S off
// the constant multiplier of the pid-carrying row offset, but only when the
// function carries no `i_bh % H` remainder -- H > 1 always leaves one, and with
// H > 1 that same constant would be H*S, i.e. a vector read spilling into the
// next head's row.
// Inputs/nsa_fwd_h1_dyn_ns.mlir is that IR (H=1 HQ=16 G=16 K=V=64 S=16 BS=32).
// RUN: triton-opt %S/Inputs/nsa_fwd_h1_dyn_ns.mlir -graph-optimize='rule-mask=0' -canonicalize -cse -o %t.h1dyn.ctl
// RUN: triton-opt %S/Inputs/nsa_fwd_h1_dyn_ns.mlir -graph-optimize='rule-mask=196608' -o %t.h1dyn.joint
// RUN: FileCheck %s --check-prefix=H1DYNREW < %t.h1dyn.joint
// RUN: not diff %t.h1dyn.ctl %t.h1dyn.joint
// the joint rewrite lands on the runtime-grid flavour with R=8 and the slot
// axis still S=16 wide (the row stride H*S was read as S because H == 1)
// H1DYNREW: hacc.program_grid_transforms = {extent_source = "runtime_original_grid"
// H1DYNREW: tt.func public @parallel_nsa_fwd_kernel(
// H1DYNREW-NOT: scf.if
// H1DYNREW: tensor<8x16x!tt.ptr<i64>>
// H1DYNREW: scf.for
// H1DYNREW: "tt.reduce"(%{{.*}}) <{axis = 2 : i32}>
// H1DYNREW: tt.store
// H1DYNREW: tt.return
