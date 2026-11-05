// RUN: split-file %s %t
// RUN: triton-opt %t/kernel.mlir --verify-each -graph-optimize='rule-mask=131072' -canonicalize -cse | FileCheck %s --check-prefix=LAST
// RUN: triton-opt %t/kernel.mlir --verify-each -graph-optimize='rule-mask=8' -canonicalize -cse | FileCheck %s --check-prefix=LAST
// Runtime offsets have no independently proven range. The input contract rules
// out wraparound; both callers must use the last row without a group maximum.
// RUN: sed 's/arith.addi %pid, %delta/arith.addi %pid, %bias/' %t/kernel.mlir > %t/unknown.mlir
// RUN: triton-opt %t/unknown.mlir --verify-each -graph-optimize='rule-mask=131072' -canonicalize -cse | FileCheck %s --check-prefix=LAST
// RUN: triton-opt %t/unknown.mlir --verify-each -graph-optimize='rule-mask=8' -canonicalize -cse | FileCheck %s --check-prefix=LAST
// RUN: sed 's/arith.addi %pid, %delta/arith.subi %pid, %bias/' %t/kernel.mlir > %t/sub.mlir
// RUN: triton-opt %t/sub.mlir --verify-each -graph-optimize='rule-mask=131072' -canonicalize -cse | FileCheck %s --check-prefix=LAST
// RUN: sed 's/arith.addi %pid, %delta/arith.addi %bias, %delta/' %t/kernel.mlir > %t/invariant.mlir
// RUN: triton-opt %t/invariant.mlir --verify-each -graph-optimize='rule-mask=131072' -canonicalize -cse | FileCheck %s --check-prefix=LAST

// LAST: hacc.coalesce_factor = {{(64|8)}} : i32
// LAST-LABEL: tt.func @bound_proof(
// LAST-NOT: "tt.reduce"
// LAST-NOT: arith.maxsi
// LAST: scf.for
// LAST: tt.dot
// LAST: scf.yield

//--- kernel.mlir
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, hacc.grid_num_tiles = 64 : i32} {
  tt.func @bound_proof(%A: !tt.ptr<f16>, %B: !tt.ptr<f16>, %C: !tt.ptr<f32>, %bias: i32) {
    %pid = tt.get_program_id x : i32
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %stride = arith.constant 256 : i32
    %base = arith.muli %pid, %stride : i32
    %delta = arith.constant 64 : i32
    %divisor = arith.constant 64 : i32
    %calc = arith.addi %pid, %delta : i32
    %ub = arith.divsi %calc, %divisor : i32
    %r = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %c = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %r2 = tt.expand_dims %r {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %c2d = tt.expand_dims %c {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %sixteen = arith.constant dense<16> : tensor<16x1xi32>
    %rs = arith.muli %r2, %sixteen : tensor<16x1xi32>
    %rr = tt.broadcast %rs : tensor<16x1xi32> -> tensor<16x16xi32>
    %cc = tt.broadcast %c2d : tensor<1x16xi32> -> tensor<16x16xi32>
    %local = arith.addi %rr, %cc : tensor<16x16xi32>
    %bases = tt.splat %base : i32 -> tensor<16x16xi32>
    %offset = arith.addi %bases, %local : tensor<16x16xi32>
    %ap = tt.splat %A : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>>
    %pa = tt.addptr %ap, %offset : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi32>
    %a = tt.load %pa : tensor<16x16x!tt.ptr<f16>>
    %bp = tt.splat %B : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>>
    %pb = tt.addptr %bp, %local : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi32>
    %zero = arith.constant dense<0.0> : tensor<16x16xf32>
    %loop = scf.for %iv = %c0 to %ub step %c1 iter_args(%acc = %zero) -> (tensor<16x16xf32>) : i32 {
      %b = tt.load %pb : tensor<16x16x!tt.ptr<f16>>
      %dot = tt.dot %a, %b, %acc : tensor<16x16xf16> * tensor<16x16xf16> -> tensor<16x16xf32>
      scf.yield %dot : tensor<16x16xf32>
    }
    %cp = tt.splat %C : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
    %pc = tt.addptr %cp, %offset : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    tt.store %pc, %loop : tensor<16x16x!tt.ptr<f32>>
    tt.return
  }
}
