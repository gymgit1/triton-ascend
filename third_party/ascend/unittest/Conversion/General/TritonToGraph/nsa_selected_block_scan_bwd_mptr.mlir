// === #2686 Rule A (SelectedBlockScanRule, bit 65536) — bwd dq, BOTH tile spellings ===
//
// The additive (bwd dq) carried-state form reads one K tile and one V tile per
// iteration.  Its two address spellings must both match:
//
//   * tt.make_tensor_ptr + boundaryCheck  — what the production kernel
//     Q3TritonKernel/src/kernels/fla/ops/nsa/nsa_bwd_dq.py emits (it uses
//     tl.make_block_ptr).  Fixture: %S/Inputs/nsa_bwd_dq_mptr.mlir.
//   * two-level tt.addptr + explicit mask — what fla/ops/nsa/parallel.py emits.
//     Fixture: %S/Inputs/nsa_bwd_dq_const_ns.mlir.
//
// matchTiles() used to accept only the second one ("The additive form: no mptr
// at all"), so the rule matched the fixture and silently declined on every real
// bwd dq kernel: `selected-block-scan: stage tiles declined`, function
// byte-identical.  These RUN lines pin both spellings down.
//
// GATED (3rd RUN) locks in the joint contract: with Rule B's bit off, Rule A
// must NOT commit — the gather loop, its scf.if and the per-iteration scalar
// table load all stay in place.

// RUN: triton-opt %S/Inputs/nsa_bwd_dq_mptr.mlir -graph-optimize='rule-mask=196608' -o - | FileCheck %s --check-prefix=MPTR
// RUN: triton-opt %S/Inputs/nsa_bwd_dq_mptr.mlir -graph-optimize='rule-mask=65536' -o - | FileCheck %s --check-prefix=GATED
// RUN: triton-opt %S/Inputs/nsa_bwd_dq_const_ns.mlir -graph-optimize='rule-mask=196608' -o - | FileCheck %s --check-prefix=ADDPTR

// ---- the production (mptr) spelling: the loop is fully rewritten ----
// MPTR-LABEL: tt.func public @parallel_nsa_bwd_kernel_dq
// MPTR-NOT: scf.if
// The scan bound is cdiv(row + 1, BN) in the scalar domain, plus the per-row
// block-count divide of the weight chain.
// MPTR: arith.divsi
// MPTR: arith.divsi {{.*}} : tensor<
// The weight reduce is the new third reduction of the additive form (the
// baseline has none): axis 2 of the row-lifted [R, G, BN] score tile.
// MPTR: "tt.reduce"(%{{.*}}) <{axis = 2 : i32}>
// Three dots, all with the lifted M = 128 row block.  The last one consumes
// k^T, whose order attribute must be replayed verbatim (it is semantic).
// MPTR: tt.dot {{.*}} : tensor<128x64xf16> * tensor<64x64xf16> -> tensor<128x64xf32>
// MPTR: tt.dot {{.*}} : tensor<128x64xf16> * tensor<64x64xf16> -> tensor<128x64xf32>
// MPTR: tt.trans {{.*}} {order = array<i32: 1, 0>} : tensor<64x64xf16> -> tensor<64x64xf16>
// MPTR: tt.dot {{.*}} : tensor<128x64xf16> * tensor<64x64xf16> -> tensor<128x64xf32>
// The single dq store survives; nothing else was added to the function.
// MPTR: tt.store
// MPTR-NOT: scf.if

// ---- joint gate closed: nothing may be committed ----
// GATED-LABEL: tt.func public @parallel_nsa_bwd_kernel_dq
// The per-iteration scalar table read and the block guard survive untouched.
// GATED: tt.load %{{.*}} : !tt.ptr<i64>
// GATED: arith.muli %{{.*}} : i32
// GATED: scf.if
// GATED-NOT: tt.reduce

// ---- the legacy addptr spelling must keep matching (no regression) ----
// ADDPTR-LABEL: tt.func public @parallel_nsa_bwd_kernel_dq
// ADDPTR-NOT: scf.if
// ADDPTR: "tt.reduce"(%{{.*}}) <{axis = 2 : i32}>
// ADDPTR: tt.trans {{.*}} {order = array<i32: 1, 0>}
// ADDPTR: tt.store
// ADDPTR-NOT: scf.if
