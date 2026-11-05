// RUN: triton-opt %S/Inputs/nsa_bwd_dq_wide_mptr.mlir --verify-each -graph-optimize='rule-mask=196608 ub-capacity-bytes=209715 mapping-ub-capacity-bytes=262144 reserved-ub-bytes=8192 ub-safety-percent=80 mapping-l0a-capacity-bytes=65536 mapping-l0b-capacity-bytes=65536 mapping-l0c-capacity-bytes=262144 l0-safety-percent=80 core-num=28' -canonicalize -cse | FileCheck %s
// RUN: triton-opt %S/Inputs/nsa_bwd_dq_wide_mptr.mlir --verify-each -graph-optimize='rule-mask=196608 ub-capacity-bytes=2048' -canonicalize -cse | FileCheck %s --check-prefix=FALLBACK

// Production B3/T111/H1/HQ32/K100, BS32: the real target budgets must admit
// R=2. Q scaling and lse/delta broadcasts use the dot layout before the loop;
// the accumulator goes directly from the last dot to the backedge. K stays a
// single shared load for QK and trans(K), with no extra half-width reloads.
// CHECK: factor = 2 : i64
// CHECK-LABEL: tt.func public @parallel_nsa_bwd_kernel_dq
// CHECK: %[[Q:.*]] = tt.load {{.*}} : tensor<2x32x128x!tt.ptr<f16>>
// CHECK: %[[QFLAT:.*]] = tt.reshape %[[Q]] : tensor<2x32x128xf16> -> tensor<64x128xf16>
// CHECK: arith.extf %[[QFLAT]] : tensor<64x128xf16> to tensor<64x128xf32>
// CHECK: arith.mulf {{.*}} : tensor<64x128xf32>
// CHECK: arith.truncf {{.*}} : tensor<64x128xf32> to tensor<64x128xf16>
// CHECK: scf.for {{.*}} -> (tensor<64x128xf32>)
// CHECK: %[[K:.*]] = tt.load {{.*}} : !tt.ptr<tensor<128x32xf16>>
// CHECK: tt.load {{.*}} : !tt.ptr<tensor<128x32xf16>>
// CHECK: tt.dot {{.*}}, %[[K]], {{.*}} -> tensor<64x32xf32>
// CHECK: arith.select {{.*}} : tensor<64x32xi1>, tensor<64x32xf32>
// CHECK: tt.dot {{.*}} -> tensor<64x32xf32>
// CHECK: %[[KT:.*]] = tt.trans %[[K]] {order = array<i32: 1, 0>} : tensor<128x32xf16> -> tensor<32x128xf16>
// CHECK: %[[DQ:.*]] = tt.dot {{.*}}, %[[KT]], {{.*}} -> tensor<64x128xf32>
// CHECK-NOT: arith.select
// CHECK: scf.yield %[[DQ]] : tensor<64x128xf32>
// CHECK: tt.store {{.*}} : tensor<2x32x128x!tt.ptr<f16>>

// FALLBACK-NOT: hacc.program_grid_transforms
// FALLBACK: scf.for
// FALLBACK: scf.if
