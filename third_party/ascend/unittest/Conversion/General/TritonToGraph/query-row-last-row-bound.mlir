// RUN: split-file %S/query-row-bound-proof.mlir %t
// RUN: sed 's/, hacc.grid_num_tiles = 64 : i32//; s/arith.addi %pid, %delta/arith.addi %pid, %bias/' %t/kernel.mlir > %t/tail.mlir
// RUN: triton-opt %t/tail.mlir --verify-each -graph-optimize='rule-mask=131072' -canonicalize -cse | FileCheck %s --check-prefixes=TAIL,BATCH
// RUN: triton-opt %t/tail.mlir --verify-each -graph-optimize='rule-mask=8' -canonicalize -cse | FileCheck %s --check-prefixes=TAIL,DOT
// RUN: sed 's/arith.addi %pid, %delta/arith.divsi %pid, %bias/' %t/kernel.mlir > %t/unsafe-bound.mlir
// RUN: triton-opt %t/unsafe-bound.mlir --verify-each -graph-optimize='rule-mask=131072' | FileCheck %s --check-prefix=DECLINE
// RUN: triton-opt %t/unsafe-bound.mlir --verify-each -graph-optimize='rule-mask=8' | FileCheck %s --check-prefix=DECLINE
// RUN: sed 's/%b = tt.load %pb/%bad = arith.divsi %iv, %bias : i32\n      %badvec = tt.splat %bad : i32 -> tensor<16x16xi32>\n      %bp2 = tt.addptr %pb, %badvec : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi32>\n      %b = tt.load %bp2/' %t/kernel.mlir > %t/unsafe-body.mlir
// RUN: triton-opt %t/unsafe-body.mlir --verify-each -graph-optimize='rule-mask=131072' | FileCheck %s --check-prefix=DECLINE
// RUN: triton-opt %t/unsafe-body.mlir --verify-each -graph-optimize='rule-mask=8' | FileCheck %s --check-prefix=DECLINE
// RUN: sed 's/arith.addi %pid, %delta/arith.subi %delta, %pid/; s/%delta = arith.constant 64/%delta = arith.constant 7/; s/%divisor = arith.constant 64/%divisor = arith.constant 2/; s/step %c1/step %divisor/' %t/kernel.mlir > %t/decreasing-step2.mlir
// RUN: triton-opt %t/decreasing-step2.mlir --verify-each -graph-optimize='rule-mask=131072' -canonicalize -cse | FileCheck %s --check-prefix=DECLINE
// RUN: sed 's/%calc = arith.addi %pid, %delta : i32/%flag = arith.cmpi sge, %bias, %c0 : i32\n    %calc = scf.if %flag -> (i32) {\n      %biased = arith.addi %pid, %bias : i32\n      scf.yield %biased : i32\n    } else {\n      scf.yield %pid : i32\n    }/' %t/kernel.mlir > %t/if-bound.mlir
// RUN: triton-opt %t/if-bound.mlir --verify-each -graph-optimize='rule-mask=131072' -canonicalize -cse | FileCheck %s --check-prefix=DECLINE
// RUN: sed 's/%delta = arith.constant 64/%delta = arith.constant -32/; s/%divisor = arith.constant 64/%divisor = arith.constant 32/; s/%ub = arith.divsi %calc, %divisor : i32/%quot = arith.divsi %calc, %divisor : i32\n    %ub = arith.addi %quot, %c1 : i32/' %t/kernel.mlir > %t/negative-numerator.mlir
// RUN: triton-opt %t/negative-numerator.mlir --verify-each -graph-optimize='rule-mask=131072 mapping-l0a-capacity-bytes=1280 mapping-l0b-capacity-bytes=65536 mapping-l0c-capacity-bytes=262144' -canonicalize -cse | FileCheck %s --check-prefix=NEGATIVE
// RUN: sed 's/%a = tt.load %pa/%bad = arith.divsi %pid, %bias : i32\n    %badvec = tt.splat %bad : i32 -> tensor<16x16xi32>\n    %pa2 = tt.addptr %pa, %badvec : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi32>\n    %a = tt.load %pa2/' %t/tail.mlir > %t/unsafe-address.mlir
// RUN: triton-opt %t/unsafe-address.mlir --verify-each -graph-optimize='rule-mask=131072' | FileCheck %s --check-prefix=DECLINE
// RUN: triton-opt %t/unsafe-address.mlir --verify-each -graph-optimize='rule-mask=8' | FileCheck %s --check-prefix=DECLINE

// RUN: sed 's/    tt.return/    %ok = arith.cmpf oge, %loop, %zero : tensor<16x16xf32>\n    tt.assert %ok, "observe accumulator" : tensor<16x16xi1>\n    tt.return/' %t/tail.mlir > %t/assert-tail.mlir
// RUN: triton-opt %t/assert-tail.mlir --verify-each -graph-optimize='rule-mask=131072' -canonicalize -cse | FileCheck %s --check-prefix=ASSERT

// Last-row selection must clamp a partial tail to the original extent, while
// preserving the per-row activity guard. No scalar or tensor max is generated.
// BATCH: factor = 32 : i64
// DOT: factor = 8 : i64
// TAIL-LABEL: tt.func @bound_proof(
// TAIL: arith.minsi
// TAIL-NOT: "tt.reduce"
// TAIL-NOT: arith.maxsi
// TAIL: scf.for
// TAIL: arith.cmpi slt
// TAIL-NOT: arith.andi
// TAIL: arith.select
// TAIL: scf.yield

// DECLINE-NOT: hacc.coalesce_factor
// DECLINE-NOT: hacc.program_grid_transforms
// DECLINE-LABEL: tt.func @bound_proof(
// DECLINE: tt.dot {{.*}} : tensor<16x16xf16> * tensor<16x16xf16> -> tensor<16x16xf32>

// [-32,-31]/32+1 gives different trip counts: monotonicity does not permit
// removing the per-row guard just because the group and divisor are aligned.
// NEGATIVE: hacc.coalesce_factor = 2 : i32
// NEGATIVE-LABEL: tt.func @bound_proof(
// NEGATIVE-NOT: "tt.reduce"
// NEGATIVE: scf.for
// NEGATIVE: arith.cmpi slt
// NEGATIVE: arith.select
// NEGATIVE: scf.yield

// 调试断言可能观察循环结果，保留补齐行累加器的原值；普通路径不需要此 AND。
// ASSERT: hacc.program_grid_transforms
// ASSERT-LABEL: tt.func @bound_proof(
// ASSERT: %[[VALID:.*]] = arith.cmpi slt, {{.*}} : tensor<32xi32>
// ASSERT: scf.for
// ASSERT: %[[ACTIVE:.*]] = arith.cmpi slt, {{.*}} : tensor<32xi32>
// ASSERT: %[[GUARDED:.*]] = arith.andi %[[ACTIVE]], %[[VALID]] : tensor<32xi1>
// ASSERT: tt.expand_dims %[[GUARDED]]
// ASSERT: arith.select
// ASSERT: scf.yield
// ASSERT: tt.assert {{.*}}, "observe accumulator"
