module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @parallel_nsa_bwd_kernel_dq(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>, %arg3: !tt.ptr<f32>, %arg4: !tt.ptr<f32>, %arg5: !tt.ptr<f16>, %arg6: !tt.ptr<f16>, %arg7: f32, %arg8: !tt.ptr<i64>, %arg9: i32, %arg10: i32) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %c16_i32 = arith.constant 16 : i32
    %c3200_i64 = arith.constant 3200 : i64
    %cst = arith.constant dense<0.000000e+00> : tensor<32x128xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
    %c128_i32 = arith.constant 128 : i32
    %c0_i32 = arith.constant 0 : i32
    %c16_i64 = arith.constant 16 : i64
    %c1_i64 = arith.constant 1 : i64
    %c100_i64 = arith.constant 100 : i64
    %c32_i64 = arith.constant 32 : i64
    %c32_i32 = arith.constant 32 : i32
    %c3_i32 = arith.constant 3 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = tt.get_program_id z : i32
    %3 = arith.muli %arg10, %c3_i32 : i32
    %4 = arith.muli %2, %arg10 : i32
    %5 = arith.extsi %4 : i32 to i64
    %6 = arith.extsi %0 : i32 to i64
    %7 = arith.addi %5, %6 : i64
    %8 = arith.muli %7, %c3200_i64 : i64
    %9 = tt.addptr %arg0, %8 : !tt.ptr<f16>, i64
    %10 = tt.addptr %arg5, %8 : !tt.ptr<f16>, i64
    %11 = arith.muli %7, %c32_i64 : i64
    %12 = tt.addptr %arg3, %11 : !tt.ptr<f32>, i64
    %13 = tt.addptr %arg4, %11 : !tt.ptr<f32>, i64
    %14 = arith.muli %1, %3 : i32
    %15 = arith.extsi %14 : i32 to i64
    %16 = arith.addi %15, %5 : i64
    %17 = arith.addi %16, %6 : i64
    %18 = arith.muli %17, %c3200_i64 : i64
    %19 = tt.addptr %arg6, %18 : !tt.ptr<f16>, i64
    %20 = arith.muli %7, %c16_i64 : i64
    %21 = tt.addptr %arg8, %20 : !tt.ptr<i64>, i64
    %22 = arith.muli %5, %c100_i64 : i64
    %23 = tt.addptr %arg1, %22 : !tt.ptr<f16>, i64
    %24 = tt.addptr %arg2, %22 : !tt.ptr<f16>, i64
    %25 = tt.make_tensor_ptr %9, [%c32_i64, %c100_i64], [%c100_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x128xf16>>
    %26 = tt.make_tensor_ptr %19, [%c32_i64, %c100_i64], [%c100_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x128xf16>>
    %27 = tt.load %25 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<32x128xf16>>
    %28 = arith.extf %27 : tensor<32x128xf16> to tensor<32x128xf32>
    %29 = tt.splat %arg7 : f32 -> tensor<32x128xf32>
    %30 = arith.mulf %28, %29 : tensor<32x128xf32>
    %31 = arith.truncf %30 : tensor<32x128xf32> to tensor<32x128xf16>
    %32 = arith.muli %1, %c128_i32 : i32
    %33 = tt.make_tensor_ptr %10, [%c32_i64, %c100_i64], [%c100_i64, %c1_i64], [%c0_i32, %32] {order = array<i32: 1, 0>} : <tensor<32x128xf16>>
    %34 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %35 = tt.splat %12 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %36 = tt.addptr %35, %34 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %37 = tt.splat %13 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %38 = tt.addptr %37, %34 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %39 = tt.load %33 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<32x128xf16>>
    %40 = tt.load %36 : tensor<32x!tt.ptr<f32>>
    %41 = tt.load %38 : tensor<32x!tt.ptr<f32>>
    %42 = scf.for %arg11 = %c0_i32 to %c16_i32 step %c1_i32 iter_args(%arg12 = %cst) -> (tensor<32x128xf32>)  : i32 {
      %45 = tt.addptr %21, %arg11 : !tt.ptr<i64>, i32
      %46 = tt.load %45 : !tt.ptr<i64>
      %47 = arith.trunci %46 : i64 to i32
      %48 = arith.muli %47, %c32_i32 : i32
      %49 = arith.cmpi sle, %48, %0 : i32
      %50 = arith.cmpi sge, %48, %c0_i32 : i32
      %51 = arith.andi %49, %50 : i1
      %52 = scf.if %51 -> (tensor<32x128xf32>) {
        %53 = arith.extsi %arg10 : i32 to i64
        %54 = tt.make_tensor_ptr %23, [%c100_i64, %53], [%c1_i64, %c100_i64], [%c0_i32, %48] {order = array<i32: 0, 1>} : <tensor<128x32xf16>>
        %55 = tt.make_tensor_ptr %24, [%c100_i64, %53], [%c1_i64, %c100_i64], [%32, %48] {order = array<i32: 0, 1>} : <tensor<128x32xf16>>
        %56 = tt.load %54 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<128x32xf16>>
        %57 = tt.load %55 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<128x32xf16>>
        %58 = tt.dot %31, %56, %cst_0 : tensor<32x128xf16> * tensor<128x32xf16> -> tensor<32x32xf32>
        %59 = tt.expand_dims %40 {axis = 1 : i32} : tensor<32xf32> -> tensor<32x1xf32>
        %60 = tt.broadcast %59 : tensor<32x1xf32> -> tensor<32x32xf32>
        %61 = arith.subf %58, %60 : tensor<32x32xf32>
        %62 = math.exp %61 : tensor<32x32xf32>
        %63 = tt.splat %48 : i32 -> tensor<32xi32>
        %64 = arith.addi %63, %34 : tensor<32xi32>
        %65 = tt.splat %0 : i32 -> tensor<32xi32>
        %66 = arith.cmpi sge, %65, %64 : tensor<32xi32>
        %67 = tt.expand_dims %66 {axis = 0 : i32} : tensor<32xi1> -> tensor<1x32xi1>
        %68 = tt.broadcast %67 : tensor<1x32xi1> -> tensor<32x32xi1>
        %69 = arith.select %68, %62, %cst_0 : tensor<32x32xi1>, tensor<32x32xf32>
        %70 = tt.dot %39, %57, %cst_0 : tensor<32x128xf16> * tensor<128x32xf16> -> tensor<32x32xf32>
        %71 = tt.expand_dims %41 {axis = 1 : i32} : tensor<32xf32> -> tensor<32x1xf32>
        %72 = tt.broadcast %71 : tensor<32x1xf32> -> tensor<32x32xf32>
        %73 = arith.subf %70, %72 : tensor<32x32xf32>
        %74 = arith.mulf %69, %73 : tensor<32x32xf32>
        %75 = arith.truncf %74 : tensor<32x32xf32> to tensor<32x32xf16>
        %76 = tt.trans %56 {order = array<i32: 1, 0>} : tensor<128x32xf16> -> tensor<32x128xf16>
        %77 = tt.dot %75, %76, %arg12 : tensor<32x32xf16> * tensor<32x128xf16> -> tensor<32x128xf32>
        scf.yield %77 : tensor<32x128xf32>
      } else {
        scf.yield %arg12 : tensor<32x128xf32>
      }
      scf.yield %52 : tensor<32x128xf32>
    }
    %43 = arith.mulf %42, %29 : tensor<32x128xf32>
    %44 = arith.truncf %43 : tensor<32x128xf32> to tensor<32x128xf16>
    tt.store %26, %44 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<32x128xf16>>
    tt.return
  }
}

