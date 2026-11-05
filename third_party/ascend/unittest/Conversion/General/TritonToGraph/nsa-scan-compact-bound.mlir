// RUN: triton-opt %S/Inputs/nsa_fwd_h1_dyn_ns.mlir --verify-each -graph-optimize='rule-mask=196608' -canonicalize -cse | FileCheck %s

// A/B 联合改写使用 (row + BN) / BN：末行标量和逐行张量均无需负位置 select。
// 负行的有符号商不大于零，零起点扫描不执行；有效输入约定 row + BN 不回绕。
// 此 fixture 的偏移在运行期传入，不能依靠常量折叠消除多余 select。
// CHECK: hacc.program_grid_transforms
// CHECK-LABEL: tt.func public @parallel_nsa_fwd_kernel(
// CHECK-DAG: %[[BN:.*]] = arith.constant 64 : i32
// CHECK-DAG: %[[BNS:.*]] = arith.constant dense<64> : tensor<32xi32>
// CHECK: %[[END:.*]] = arith.addi %{{.*}}, %[[BN]] : i32
// CHECK: %[[ENDS:.*]] = arith.addi %{{.*}}, %[[BNS]] : tensor<32xi32>
// CHECK: %[[BOUND:.*]] = arith.divsi %[[END]], %[[BN]] : i32
// CHECK: %[[BOUNDS:.*]] = arith.divsi %[[ENDS]], %[[BNS]] : tensor<32xi32>
// CHECK-NOT: arith.select
// CHECK: scf.for {{.*}} to %[[BOUND]]
// CHECK: %[[IVS:.*]] = tt.splat {{.*}} : i32 -> tensor<32xi32>
// CHECK: %[[ACTIVE:.*]] = arith.cmpi slt, %[[IVS]], %[[BOUNDS]] : tensor<32xi32>
// CHECK-NOT: arith.andi %[[ACTIVE]]
// CHECK: scf.yield
