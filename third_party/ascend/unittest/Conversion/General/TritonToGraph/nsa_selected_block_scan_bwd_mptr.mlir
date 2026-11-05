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
// parseTiles() used to accept only the second one ("The additive form: no mptr
// at all"), so the rule matched the fixture and silently declined on every real
// bwd dq kernel: `selected-block-scan: stage tiles declined`, function
// byte-identical.  These RUN lines pin both spellings down.
//
// The GATED pair (3rd/4th RUN) locks in that Rule A needs no Rule B bit: Rule A
// always runs Rule B's real implementation (QueryRowBatching.h), so offering
// only bit 65536 must give the same IR as 65536|131072 -- the standalone Rule B
// scheduled by 131072 finds nothing left to lift in the joint result.

// RUN: triton-opt %S/Inputs/nsa_bwd_dq_mptr.mlir -graph-optimize='rule-mask=196608' -o - | FileCheck %s --check-prefix=MPTR
// RUN: triton-opt %S/Inputs/nsa_bwd_dq_mptr.mlir -graph-optimize='rule-mask=65536' -o %t.gated
// RUN: triton-opt %S/Inputs/nsa_bwd_dq_mptr.mlir -graph-optimize='rule-mask=196608' -o %t.joint
// RUN: diff %t.joint %t.gated
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
// Three dots, all with the lifted M = 512 row block (16 rows x 32).  The factor
// comes from the shared row-count policy, which now derives its upper bound from
// the per-value budget instead of the retired hard-coded 8-row cap.  The last
// one consumes k^T, whose order attribute must be replayed verbatim (it is
// semantic).
// MPTR: tt.dot {{.*}} : tensor<512x64xf16> * tensor<64x64xf16> -> tensor<512x64xf32>
// MPTR: tt.dot {{.*}} : tensor<512x64xf16> * tensor<64x64xf16> -> tensor<512x64xf32>
// MPTR: tt.trans {{.*}} {order = array<i32: 1, 0>} : tensor<64x64xf16> -> tensor<64x64xf16>
// MPTR: tt.dot {{.*}} : tensor<512x64xf16> * tensor<64x64xf16> -> tensor<512x64xf32>
// The single dq store survives; nothing else was added to the function.
// MPTR: tt.store
// MPTR-NOT: scf.if

// ---- the legacy addptr spelling must keep matching (no regression) ----
// ADDPTR-LABEL: tt.func public @parallel_nsa_bwd_kernel_dq
// ADDPTR-NOT: scf.if
// ADDPTR: "tt.reduce"(%{{.*}}) <{axis = 2 : i32}>
// ADDPTR: tt.trans {{.*}} {order = array<i32: 1, 0>}
// ADDPTR: tt.store
// ADDPTR-NOT: scf.if
