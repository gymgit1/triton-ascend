// === #2686 Rule A (SelectedBlockScanRule, bit 65536) — on-chip budget gate ===
//
// Rule A's rewrite grows the loop body (vector table read, unconditional K/V
// tiles, scan masks), so the rewritten SANDBOX -- not the gather loop it
// replaces -- is measured against the two queries the resource model already
// exposes to rules:
//   * UB : getSafeUBBudget() vs estimatePeakLiveBytes()  (rule B's caliber)
//   * L0 : exceedsCubeBudgets()                          (#2841 cube query)
// Zero capacities keep a query disabled, so every other RUN line in this
// directory (no budget options) is unaffected.  An over-budget form is declined
// with a warning on the live function; the pass never fails.
//
// Measured on Inputs/nsa_fwd_const_ns.mlir: rule A's own form peaks at 40,964 B,
// the joint rule A + rule B form at 164,864 B.  Rule B judges its own candidates
// on a demand figure, not that peak: once the value dimension leaves one vector
// register it multiplies the estimate by the split-tier x loop-carried-buffer
// amplification (2 x 2), so the joint form demands 659,456 B.  The production
// budget is 209,715 B (physical 256 KiB x 80 %), which is why section 5 below
// passes a larger explicit capacity: it exercises the gate's "fits, do not warn"
// path, while section 4 pins the declining path.

// --- 1. UB: budget 2048*80/100 = 1638 B < rule A's 40,964 B ---------------
// force-apply commits rule A alone, so its own overflow declines the rewrite and
// names both numbers.  The IR stays byte-identical to the control.
// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns.mlir -graph-optimize='rule-mask=0' -canonicalize -cse -o %t.ctl
// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns.mlir -graph-optimize='rule-mask=65536 selected-block-scan-force-apply=true ub-capacity-bytes=2048' -canonicalize -cse -o %t.ub 2> %t.ub.err
// RUN: FileCheck %s --check-prefix=BUDGETWARN < %t.ub.err
// RUN: diff %t.ctl %t.ub

// --- 2. L0: 1024*80/100 = 819 B per buffer, below the >=4 KiB dot tiles ----
// UB is left unspecified (disabled), so only the cube query can reject.
// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns.mlir -graph-optimize='rule-mask=65536 selected-block-scan-force-apply=true mapping-l0a-capacity-bytes=1024 mapping-l0b-capacity-bytes=1024 mapping-l0c-capacity-bytes=1024' -canonicalize -cse -o %t.cube 2> %t.cube.err
// RUN: FileCheck %s --check-prefix=CUBEWARN < %t.cube.err
// RUN: diff %t.ctl %t.cube

// --- 3. generous budgets: the rewrite lands, stderr stays empty ------------
// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns.mlir -graph-optimize='rule-mask=65536 selected-block-scan-force-apply=true ub-capacity-bytes=262144 mapping-l0a-capacity-bytes=65536 mapping-l0b-capacity-bytes=65536 mapping-l0c-capacity-bytes=262144' -canonicalize -cse -o %t.fit 2> %t.fit.err
// RUN: echo -n > %t.empty
// RUN: diff %t.empty %t.fit.err
// RUN: FileCheck %s --check-prefix=FIT --implicit-check-not='scf.if' < %t.fit
// RUN: not diff %t.ctl %t.fit

// --- 4. joint path under the same tight UB budget -------------------------
// The own-form gate runs before rule B is consulted, so the production path
// (no force-apply) reports the same verdict and rewrites nothing.
// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns.mlir -graph-optimize='rule-mask=196608 ub-capacity-bytes=2048' -canonicalize -cse -o %t.joint 2> %t.joint.err
// RUN: FileCheck %s --check-prefix=BUDGETWARN < %t.joint.err
// RUN: diff %t.ctl %t.joint

// --- 5. joint path under budgets wide enough for rule B's demand model ----
// 1 MiB x 80 % = 838,861 B >= the joint form's 659,456 B demand, so the second
// (post-rule-B) checkpoint passes and the committed form is the lifted loop.
// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns.mlir -graph-optimize='rule-mask=196608 ub-capacity-bytes=1048576 mapping-l0a-capacity-bytes=65536 mapping-l0b-capacity-bytes=65536 mapping-l0c-capacity-bytes=262144' -canonicalize -cse -o %t.jointfit 2> %t.jointfit.err
// RUN: diff %t.empty %t.jointfit.err
// RUN: FileCheck %s --check-prefix=JOINTFIT --implicit-check-not='scf.if' < %t.jointfit

// BUDGETWARN — the UB verdict: which form, both numbers, the decline.
// BUDGETWARN: warning: selected-block-scan: rule A's rewrite needs 40964 bytes of UB, budget 1638; declining

// CUBEWARN — the cube verdict (no scalar to quote; the budgets are per-arch).
// CUBEWARN: warning: selected-block-scan: rule A's rewrite exceeds the L0A/L0B/L0C cube budgets; declining

// FIT — rule A alone: vector table read, one 3-iter_arg scan loop, two dots.
// FIT-LABEL: tt.func public @parallel_nsa_fwd_kernel
// FIT: tt.load %{{.*}} : tensor<16x!tt.ptr<i64>>
// FIT: %{{.*}}:3 = scf.for {{.*}} iter_args(
// FIT: tt.dot %{{.*}} : tensor<16x128xbf16> * tensor<128x64xbf16> -> tensor<16x64xf32>
// FIT: tt.dot %{{.*}} : tensor<16x64xbf16> * tensor<64x128xbf16> -> tensor<16x128xf32>

// JOINTFIT — committed joint form: 8 lifted rows, 4 carried tensors, 3 dots.
// JOINTFIT-LABEL: hacc.program_grid_transforms
// JOINTFIT-LABEL: tt.func public @parallel_nsa_fwd_kernel
// JOINTFIT: tt.load %{{.*}}, %{{.*}}, %{{.*}} : tensor<8x16x!tt.ptr<i64>>
// JOINTFIT: %{{.*}}:4 = scf.for {{.*}} iter_args(
// JOINTFIT: tt.dot %{{.*}} : tensor<128x128xbf16> * tensor<128x64xbf16> -> tensor<128x64xf32>
// JOINTFIT-COUNT-2: tt.dot %{{.*}} : tensor<128x64xbf16> * tensor<64x64xbf16> -> tensor<128x64xf32>
