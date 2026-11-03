// === #2686 Rule A (SelectedBlockScanRule, bit 65536) — driver accept/reject ===
//
// STATUS: landed 2026-10-31 — all four RUN lines are green.  The REJECT line
// must STAY green: an out-of-range bit must keep being rejected so nobody
// "fixes" the whitelist by widening it.
//
// Deployed as a standalone file (the same role the 18-line
// graph-optimize-driver.mlir plays for the legacy bits: option plumbing, no
// rewrite assertions).
//
// Bit arithmetic: 65536 = 1 << 16 = SelectedBlockScan.  1048576 = 1 << 20 is
// out of range on purpose (bit index 20 is past the registry).
// 68607 = 3071 | 65536 = the default rule set plus the new bit, i.e. what a
// production run with `rule_mask=65536`-style opt-in actually passes down
// (_graph_optimize_kwargs only forwards rule_mask when it differs from the
// default 3071: third_party/ascend/backend/compiler.py:289-300).

// RUN: triton-opt --split-input-file --verify-each %s -graph-optimize='rule-mask=65536' -o - | FileCheck %s
// RUN: triton-opt --split-input-file --verify-each %s -graph-optimize='rule-mask=68607' -o - | FileCheck %s --check-prefix=COMBINED
// RUN: not triton-opt %s -graph-optimize='rule-mask=1048576' -o /dev/null 2>&1 | FileCheck %s --check-prefix=REJECT

// The new bit must also be accepted on the real kernel IR (exit 0).  Rule A has
// no separate Rule B gate -- it always runs Rule B's real implementation (the
// QueryRowBatching.h materialize entry) -- so bit 65536 alone must produce
// exactly the same IR as the joint 65536|131072 run below.
// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns.mlir -graph-optimize='rule-mask=65536' -o /dev/null
// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns.mlir -graph-optimize='rule-mask=65536' -o %t.rulea
// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns.mlir -graph-optimize='rule-mask=196608' -o %t.joint
// RUN: diff %t.rulea %t.joint

// Joint Rule A + Rule B path: 65536 | 131072 = 196608.  Rule A rewrites the
// sparse loop and materializes Rule B's contract inside its own sandbox, so
// the output must carry the dynamic grid contract and the 128-row dot shapes.
// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns.mlir -graph-optimize='rule-mask=196608' -o - | FileCheck %s --check-prefix=JOINT

// CHECK-LABEL: tt.func @graph_optimize_selected_block_scan_accept()
tt.func @graph_optimize_selected_block_scan_accept() {
  // CHECK-NEXT: tt.return
  tt.return
  // CHECK-NEXT: }
}

// -----

// COMBINED-LABEL: tt.func @graph_optimize_selected_block_scan_with_defaults(
// COMBINED: tt.return %{{.*}} : i32
tt.func @graph_optimize_selected_block_scan_with_defaults(%value: i32) -> i32 {
  tt.return %value : i32
}

// -----

// The reject case needs a module to feed the option parser; it must reject
// before doing anything to it.
// REJECT: error: graph-optimize rule-mask contains unknown or out-of-range bits
// REJECT: 1048576
tt.func @graph_optimize_reject_unknown_bit() {
  tt.return
}

// -----

// Joint A+B contract checks (fed by the RUN line above from the real kernel).
// JOINT: hacc.program_grid_transforms
// JOINT-SAME: factor = 8
// JOINT: tensor<128x128xbf16>
