// RUN: split-file %S/query-row-bound-proof.mlir %t
// RUN: sed 's/, hacc.grid_num_tiles = 64 : i32//; s/arith.addi %pid, %delta/arith.addi %pid, %bias/' %t/kernel.mlir > %t/tail.mlir
// RUN: triton-opt %t/tail.mlir --verify-each -graph-optimize='rule-mask=131072 mapping-l0a-capacity-bytes=1280 mapping-l0b-capacity-bytes=65536 mapping-l0c-capacity-bytes=262144' -canonicalize -cse | FileCheck %s --check-prefix=LAST
// RUN: sed 's/%bias: i32/%bias: i64/; s/%calc = arith.addi %pid, %bias : i32/%p64 = arith.extsi %pid : i32 to i64\n    %calc = arith.addi %p64, %bias : i64/; s/%ub = arith.divsi %calc, %divisor : i32/%div64 = arith.constant 9223372036854775807 : i64\n    %zero64 = arith.constant 0 : i64\n    %one64 = arith.constant 1 : i64\n    %ub = arith.divsi %calc, %div64 : i64/; s/scf.for %iv = %c0 to %ub step %c1/scf.for %iv = %zero64 to %ub step %one64/; s/-> (tensor<16x16xf32>) : i32 {/-> (tensor<16x16xf32>) : i64 {/' %t/tail.mlir > %t/i64.mlir
// RUN: triton-opt %t/i64.mlir --verify-each -graph-optimize='rule-mask=131072 mapping-l0a-capacity-bytes=1280 mapping-l0b-capacity-bytes=65536 mapping-l0c-capacity-bytes=262144' -canonicalize -cse | FileCheck %s --check-prefix=LAST
// RUN: triton-opt %t/tail.mlir --verify-each -graph-optimize='rule-mask=8' -canonicalize -cse | FileCheck %s --check-prefix=LAST
// RUN: triton-opt %t/tail.mlir --verify-each -graph-optimize='rule-mask=131072' -canonicalize -cse | FileCheck %s --check-prefix=LAST
// RUN: sed 's/%calc = arith.addi %pid, %bias : i32/%flag = arith.cmpi sge, %bias, %c0 : i32\n    %calc = scf.if %flag -> (i32) {\n      %biased = arith.addi %pid, %bias : i32\n      scf.yield %biased : i32\n    } else {\n      scf.yield %pid : i32\n    }/' %t/tail.mlir > %t/if.mlir
// RUN: triton-opt %t/if.mlir --verify-each -graph-optimize='rule-mask=131072 mapping-l0a-capacity-bytes=1280 mapping-l0b-capacity-bytes=65536 mapping-l0c-capacity-bytes=262144' -canonicalize -cse | FileCheck %s --check-prefix=DECLINE
// RUN: sed 's/%biased = arith.addi %pid, %bias : i32/%biased = scf.if %flag -> (i32) {\n        %nested = arith.addi %pid, %bias : i32\n        scf.yield %nested : i32\n      } else {\n        scf.yield %pid : i32\n      }/' %t/if.mlir > %t/nested.mlir
// RUN: triton-opt %t/nested.mlir --verify-each -graph-optimize='rule-mask=131072 mapping-l0a-capacity-bytes=1280 mapping-l0b-capacity-bytes=65536 mapping-l0c-capacity-bytes=262144' -canonicalize -cse | FileCheck %s --check-prefix=DECLINE
// RUN: sed 's/%bias: i32/%biasptr: !tt.ptr<i32>/; s/%calc =/%bias = tt.load %biasptr : !tt.ptr<i32>\n    %calc =/' %t/tail.mlir > %t/invariant-load.mlir
// RUN: triton-opt %t/invariant-load.mlir --verify-each -graph-optimize='rule-mask=131072 mapping-l0a-capacity-bytes=1280 mapping-l0b-capacity-bytes=65536 mapping-l0c-capacity-bytes=262144' -canonicalize -cse | FileCheck %s --check-prefix=LAST

// Small/large row factors and i32/i64 bounds all use one clamped scalar bound.
// LAST: hacc.program_grid_transforms
// LAST-LABEL: tt.func @bound_proof(
// LAST: arith.minsi
// LAST-NOT: "tt.reduce"
// LAST-NOT: arith.maxsi
// LAST: scf.for
// LAST: arith.cmpi slt
// LAST-NOT: arith.andi
// LAST: arith.select

// Region-valued bounds are outside the supported scalar arithmetic slice.
// DECLINE-NOT: hacc.program_grid_transforms
// DECLINE-LABEL: tt.func @bound_proof(
// DECLINE: tt.dot {{.*}} : tensor<16x16xf16> * tensor<16x16xf16> -> tensor<16x16xf32>
