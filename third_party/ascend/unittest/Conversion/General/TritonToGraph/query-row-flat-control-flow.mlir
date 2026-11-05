// RUN: triton-opt %s --verify-each -graph-optimize='rule-mask=131072 mapping-l0a-capacity-bytes=5120 mapping-l0b-capacity-bytes=65536 mapping-l0c-capacity-bytes=262144' -canonicalize -cse | FileCheck %s

// A row-invariant condition may branch around a dot and a row reduction. Both
// arms must carry the same flat types, including an arm returning old state.
// CHECK: factor = 8 : i64
// CHECK-LABEL: tt.func @flat_conditional_recurrence
// CHECK: scf.for {{.*}} -> (tensor<128x16xf32>, tensor<128xf32>)
// CHECK: scf.if {{.*}} -> (tensor<128x16xf32>, tensor<128xf32>)
// CHECK: tt.dot {{.*}} -> tensor<128x16xf32>
// CHECK-NOT: tt.reshape
// CHECK: "tt.reduce"({{.*}}) <{axis = 1 : i32}>
// CHECK: arith.addf {{.*}} : tensor<128xf32>
// CHECK: scf.yield {{.*}} : tensor<128x16xf32>, tensor<128xf32>
// CHECK: } else {
// CHECK: scf.yield {{.*}} : tensor<128x16xf32>, tensor<128xf32>
// CHECK: scf.yield {{.*}} : tensor<128x16xf32>, tensor<128xf32>
// CHECK: tt.store {{.*}} : tensor<8x16x16x!tt.ptr<f32>>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func @flat_conditional_recurrence(%A: !tt.ptr<f16>, %B: !tt.ptr<f16>, %O: !tt.ptr<f32>, %flag: i1) {
    %pid = tt.get_program_id x : i32
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c256 = arith.constant 256 : i32
    %c1_64 = arith.constant 1 : i64
    %c16_64 = arith.constant 16 : i64
    %base = arith.muli %pid, %c256 : i32
    %ap = tt.addptr %A, %base : !tt.ptr<f16>, i32
    %op = tt.addptr %O, %base : !tt.ptr<f32>, i32
    %qa = tt.make_tensor_ptr %ap, [%c16_64, %c16_64], [%c16_64, %c1_64], [%c0, %c0] {order = array<i32: 1, 0>} : <tensor<16x16xf16>>
    %kb = tt.make_tensor_ptr %B, [%c16_64, %c16_64], [%c16_64, %c1_64], [%c0, %c0] {order = array<i32: 1, 0>} : <tensor<16x16xf16>>
    %oo = tt.make_tensor_ptr %op, [%c16_64, %c16_64], [%c16_64, %c1_64], [%c0, %c0] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
    %q = tt.load %qa : !tt.ptr<tensor<16x16xf16>>
    %zero = arith.constant dense<0.0> : tensor<16x16xf32>
    %row_zero = arith.constant dense<0.0> : tensor<16xf32>
    %loop:2 = scf.for %iv = %c0 to %c2 step %c1 iter_args(%acc = %zero, %m = %row_zero) -> (tensor<16x16xf32>, tensor<16xf32>) : i32 {
      %next:2 = scf.if %flag -> (tensor<16x16xf32>, tensor<16xf32>) {
        %b = tt.load %kb : !tt.ptr<tensor<16x16xf16>>
        %d = tt.dot %q, %b, %acc : tensor<16x16xf16> * tensor<16x16xf16> -> tensor<16x16xf32>
        %r = "tt.reduce"(%d) <{axis = 1 : i32}> ({
        ^bb0(%x: f32, %y: f32):
          %sum = arith.addf %x, %y : f32
          tt.reduce.return %sum : f32
        }) : (tensor<16x16xf32>) -> tensor<16xf32>
        %n = arith.addf %m, %r : tensor<16xf32>
        scf.yield %d, %n : tensor<16x16xf32>, tensor<16xf32>
      } else {
        scf.yield %acc, %m : tensor<16x16xf32>, tensor<16xf32>
      }
      scf.yield %next#0, %next#1 : tensor<16x16xf32>, tensor<16xf32>
    }
    %m2 = tt.expand_dims %loop#1 {axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
    %m_full = tt.broadcast %m2 : tensor<16x1xf32> -> tensor<16x16xf32>
    %out = arith.addf %loop#0, %m_full : tensor<16x16xf32>
    tt.store %oo, %out : !tt.ptr<tensor<16x16xf32>>
    tt.return
  }
}
