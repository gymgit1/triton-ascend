// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns.mlir --verify-each -graph-optimize='rule-mask=196608 mapping-l0a-capacity-bytes=40960 mapping-l0b-capacity-bytes=65536 mapping-l0c-capacity-bytes=262144' -canonicalize -cse | FileCheck %s
// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns.mlir --verify-each -graph-optimize='rule-mask=196608 ub-capacity-bytes=262144 mapping-l0a-capacity-bytes=65536 mapping-l0b-capacity-bytes=65536 mapping-l0c-capacity-bytes=262144' -canonicalize -cse | FileCheck %s --check-prefix=UB
// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns.mlir --verify-each -graph-optimize='rule-mask=196608 ub-capacity-bytes=2048' -canonicalize -cse | FileCheck %s --check-prefix=DECLINE

// Real online-softmax fixture: G=16, V=128. Two half-width accumulators and
// m/l stay flat for every iteration, including masked recurrence updates.
// The table/membership axes keep their logical shape and K/V remain shared.
// R=8 here is constrained by L0A; no diagnostic IR replacement is involved.
// CHECK: factor = 8 : i64
// CHECK-LABEL: tt.func public @parallel_nsa_fwd_kernel
// CHECK: tt.load {{.*}} : tensor<8x16x!tt.ptr<i64>>
// CHECK: scf.for {{.*}} -> (tensor<128x64xf32>, tensor<128x64xf32>, tensor<128xf32>, tensor<128xf32>)
// CHECK: tt.load {{.*}} : !tt.ptr<tensor<128x64xbf16>>
// CHECK: tt.load {{.*}} : !tt.ptr<tensor<64x64xbf16>>
// CHECK: tt.load {{.*}} : !tt.ptr<tensor<64x64xbf16>>
// CHECK: %[[SCORES:.*]] = tt.dot {{.*}} -> tensor<128x64xf32>
// CHECK-NOT: tt.reshape %[[SCORES]]
// CHECK: arith.select {{.*}} : tensor<128x64xi1>, tensor<128x64xf32>
// CHECK: "tt.reduce"({{.*}}) <{axis = 1 : i32}>
// CHECK: math.exp {{.*}} : tensor<128xf32>
// CHECK: math.exp {{.*}} : tensor<128x64xf32>
// CHECK: "tt.reduce"({{.*}}) <{axis = 1 : i32}>
// CHECK: arith.truncf {{.*}} : tensor<128x64xf32> to tensor<128x64xbf16>
// CHECK-NOT: tt.reshape
// CHECK: tt.dot {{.*}} -> tensor<128x64xf32>
// CHECK-NOT: tt.reshape
// CHECK: tt.dot {{.*}} -> tensor<128x64xf32>
// CHECK: scf.yield {{.*}} : tensor<128x64xf32>, tensor<128x64xf32>, tensor<128xf32>, tensor<128xf32>
// CHECK: tt.reshape {{.*}} : tensor<128x64xf32> -> tensor<8x16x64xf32>
// CHECK: tt.store {{.*}} : tensor<8x16x64x!tt.ptr<bf16>>

// The materialised UB check must retry a smaller factor. Multiplying the
// unlifted whole-function peak by R incorrectly rejects this candidate.
// UB: factor = 4 : i64
// UB: scf.for {{.*}} -> (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64xf32>)
// DECLINE-NOT: hacc.program_grid_transforms
// DECLINE: scf.for
