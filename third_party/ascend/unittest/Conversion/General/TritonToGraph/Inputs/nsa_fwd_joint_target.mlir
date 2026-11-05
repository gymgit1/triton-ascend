// #2686 SelectedBlockScanRule + QueryRowBatchingRule JOINT golden standard (provenance)
//   source   : Inputs/nsa_fwd_const_ns.mlir  (the real NSA fwd kernel IR after
//              the early TTIR pipeline; Ascend950PR_9579, locs already dropped)
//   produced : triton-opt <source> -graph-optimize='rule-mask=196608' -o <this>
//              i.e. SelectedBlockScanRule (bit 65536) + QueryRowBatchingRule
//              (bit 131072) committed jointly.
//   target   : Ascend950PR_9579 (NOT Ascend910_95: isAscend950() gates the row lift and
//              the graph-optimize run below silently no-ops on 910_95)
//   locs     : none (no --mlir-print-debuginfo).  Never cite line numbers from a
//              loc-free fixture; cite the loc-bearing IR under .tmp-analysis.
//   note     : this artifact carries hacc.program_grid_transforms on purpose --
//              it is the launch-contract flavour of the joint rewrite.  The
//              in-kernel flavour (flat pid + `if active`) is NOT what the row lift
//              emits here; see .issue2686-notes/15-v3-realignment.md 0.1.
module attributes {hacc.program_grid_transforms = {extent_source = "runtime_original_grid", hidden_argument_order = ["originalGridX", "originalGridY"], hidden_argument_types = ["i32", "i32"], hidden_extent_axes = [0, 1], transforms = [{axis = 0 : i64, factor = 8 : i64, grid_stride_abi_verified = false, kind = "ceil_div", order = 0 : i64, persistent_coverage = false}], version = 2 : i64}, hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @parallel_nsa_fwd_kernel(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>, %arg2: !tt.ptr<bf16>, %arg3: !tt.ptr<bf16>, %arg4: !tt.ptr<f32>, %arg5: f32, %arg6: !tt.ptr<i64>, %arg7: !tt.ptr<i64>, %arg8: i32, %arg9: i32, %arg10: i32) attributes {noinline = false} {
    %cst = arith.constant dense<128> : tensor<8x16x64xi64>
    %cst_0 = arith.constant dense<64> : tensor<8x16x64xi64>
    %cst_1 = arith.constant dense<0> : tensor<8x16x64xi64>
    %cst_2 = arith.constant dense<0xFF800000> : tensor<8x16x64xf32>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<128x64xf32>
    %cst_4 = arith.constant dense<0.000000e+00> : tensor<8x64xf32>
    %cst_5 = arith.constant dense<0.000000e+00> : tensor<8x16xf32>
    %cst_6 = arith.constant dense<0xFF800000> : tensor<8x16xf32>
    %cst_7 = arith.constant dense<64> : tensor<8x16xi32>
    %cst_8 = arith.constant dense<0.000000e+00> : tensor<8x16x128xbf16>
    %cst_9 = arith.constant dense<128> : tensor<8x16x128xi64>
    %cst_10 = arith.constant dense<64> : tensor<8x16x128xi64>
    %cst_11 = arith.constant dense<0> : tensor<8x16x128xi64>
    %cst_12 = arith.constant dense<128> : tensor<8x16x1xi64>
    %cst_13 = arith.constant dense<8192> : tensor<8xi32>
    %cst_14 = arith.constant dense<64> : tensor<8xi32>
    %cst_15 = arith.constant dense<0.000000e+00> : tensor<8x16x64xf32>
    %cst_16 = arith.constant dense<0> : tensor<8x16xi64>
    %c7_i32 = arith.constant 7 : i32
    %c8_i32 = arith.constant 8 : i32
    %cst_17 = arith.constant dense<16> : tensor<16xi32>
    %cst_18 = arith.constant dense<15> : tensor<16xi32>
    %c1_i32 = arith.constant 1 : i32
    %c512_i64 = arith.constant 512 : i64
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c128_i64 = arith.constant 128 : i64
    %c64_i32 = arith.constant 64 : i32
    %c16_i32 = arith.constant 16 : i32
    %c128_i32 = arith.constant 128 : i32
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c8_i32 : i32
    %2 = arith.addi %1, %c7_i32 : i32
    %3 = arith.subi %arg9, %c1_i32 : i32
    %4 = arith.minsi %2, %3 : i32
    %5 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %6 = tt.splat %1 : i32 -> tensor<8xi32>
    %7 = arith.addi %6, %5 : tensor<8xi32>
    %8 = tt.splat %arg9 : i32 -> tensor<8xi32>
    %9 = arith.cmpi slt, %7, %8 : tensor<8xi32>
    %10 = tt.get_program_id y : i32
    %11 = tt.get_program_id z : i32
    %12 = arith.divsi %11, %c4_i32 : i32
    %13 = arith.remsi %11, %c4_i32 : i32
    %14 = arith.muli %12, %arg8 : i32
    %15 = arith.muli %14, %c4_i32 : i32
    %16 = arith.addi %15, %13 : i32
    %17 = arith.muli %16, %c128_i32 : i32
    %18 = tt.addptr %arg1, %17 : !tt.ptr<bf16>, i32
    %19 = tt.addptr %arg2, %17 : !tt.ptr<bf16>, i32
    %20 = tt.splat %14 : i32 -> tensor<8xi32>
    %21 = arith.addi %20, %7 : tensor<8xi32>
    %22 = arith.muli %21, %cst_14 : tensor<8xi32>
    %23 = arith.muli %13, %c16_i32 : i32
    %24 = tt.splat %23 : i32 -> tensor<8xi32>
    %25 = arith.addi %22, %24 : tensor<8xi32>
    %26 = tt.splat %arg6 : !tt.ptr<i64> -> tensor<8x!tt.ptr<i64>>
    %27 = tt.addptr %26, %25 : tensor<8x!tt.ptr<i64>>, tensor<8xi32>
    %28 = arith.muli %21, %cst_13 : tensor<8xi32>
    %29 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<8x!tt.ptr<bf16>>
    %30 = tt.addptr %29, %28 : tensor<8x!tt.ptr<bf16>>, tensor<8xi32>
    %31 = arith.extsi %23 : i32 to i64
    %32 = tt.splat %31 : i64 -> tensor<8x16x1xi64>
    %33 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %34 = arith.extsi %33 : tensor<16xi32> to tensor<16xi64>
    %35 = tt.expand_dims %34 {axis = 0 : i32} : tensor<16xi64> -> tensor<1x16xi64>
    %36 = tt.expand_dims %35 {axis = 2 : i32} : tensor<1x16xi64> -> tensor<1x16x1xi64>
    %37 = tt.broadcast %36 : tensor<1x16x1xi64> -> tensor<8x16x1xi64>
    %38 = arith.addi %32, %37 : tensor<8x16x1xi64>
    %39 = arith.muli %38, %cst_12 : tensor<8x16x1xi64>
    %40 = tt.expand_dims %30 {axis = 1 : i32} : tensor<8x!tt.ptr<bf16>> -> tensor<8x1x!tt.ptr<bf16>>
    %41 = tt.expand_dims %40 {axis = 2 : i32} : tensor<8x1x!tt.ptr<bf16>> -> tensor<8x1x1x!tt.ptr<bf16>>
    %42 = tt.broadcast %41 : tensor<8x1x1x!tt.ptr<bf16>> -> tensor<8x16x1x!tt.ptr<bf16>>
    %43 = tt.addptr %42, %39 : tensor<8x16x1x!tt.ptr<bf16>>, tensor<8x16x1xi64>
    %44 = tt.broadcast %43 : tensor<8x16x1x!tt.ptr<bf16>> -> tensor<8x16x128x!tt.ptr<bf16>>
    %45 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %46 = arith.extsi %45 : tensor<128xi32> to tensor<128xi64>
    %47 = tt.expand_dims %46 {axis = 0 : i32} : tensor<128xi64> -> tensor<1x128xi64>
    %48 = tt.expand_dims %47 {axis = 1 : i32} : tensor<1x128xi64> -> tensor<1x1x128xi64>
    %49 = tt.broadcast %48 : tensor<1x1x128xi64> -> tensor<8x16x128xi64>
    %50 = tt.addptr %44, %49 : tensor<8x16x128x!tt.ptr<bf16>>, tensor<8x16x128xi64>
    %51 = tt.splat %31 : i64 -> tensor<8x16x128xi64>
    %52 = tt.broadcast %36 : tensor<1x16x1xi64> -> tensor<8x16x128xi64>
    %53 = arith.addi %51, %52 : tensor<8x16x128xi64>
    %54 = arith.cmpi sge, %53, %cst_11 : tensor<8x16x128xi64>
    %55 = arith.cmpi slt, %53, %cst_10 : tensor<8x16x128xi64>
    %56 = arith.andi %54, %55 : tensor<8x16x128xi1>
    %57 = arith.cmpi sge, %49, %cst_11 : tensor<8x16x128xi64>
    %58 = arith.cmpi slt, %49, %cst_9 : tensor<8x16x128xi64>
    %59 = arith.andi %57, %58 : tensor<8x16x128xi1>
    %60 = arith.andi %56, %59 : tensor<8x16x128xi1>
    %61 = tt.expand_dims %9 {axis = 1 : i32} : tensor<8xi1> -> tensor<8x1xi1>
    %62 = tt.expand_dims %61 {axis = 2 : i32} : tensor<8x1xi1> -> tensor<8x1x1xi1>
    %63 = tt.broadcast %62 : tensor<8x1x1xi1> -> tensor<8x16x128xi1>
    %64 = arith.andi %60, %63 : tensor<8x16x128xi1>
    %65 = tt.load %50, %64, %cst_8 {padding = 1 : i32} : tensor<8x16x128x!tt.ptr<bf16>>
    %66 = arith.extf %65 : tensor<8x16x128xbf16> to tensor<8x16x128xf32>
    %67 = tt.splat %arg5 : f32 -> tensor<8x16x128xf32>
    %68 = arith.mulf %66, %67 : tensor<8x16x128xf32>
    %69 = arith.truncf %68 : tensor<8x16x128xf32> to tensor<8x16x128xbf16>
    %70 = tt.splat %arg3 : !tt.ptr<bf16> -> tensor<8x!tt.ptr<bf16>>
    %71 = tt.addptr %70, %28 : tensor<8x!tt.ptr<bf16>>, tensor<8xi32>
    %72 = arith.muli %10, %c128_i32 : i32
    %73 = arith.addi %72, %c64_i32 : i32
    %74 = arith.extsi %72 : i32 to i64
    %75 = tt.expand_dims %71 {axis = 1 : i32} : tensor<8x!tt.ptr<bf16>> -> tensor<8x1x!tt.ptr<bf16>>
    %76 = tt.expand_dims %75 {axis = 2 : i32} : tensor<8x1x!tt.ptr<bf16>> -> tensor<8x1x1x!tt.ptr<bf16>>
    %77 = tt.broadcast %76 : tensor<8x1x1x!tt.ptr<bf16>> -> tensor<8x16x1x!tt.ptr<bf16>>
    %78 = tt.addptr %77, %39 : tensor<8x16x1x!tt.ptr<bf16>>, tensor<8x16x1xi64>
    %79 = tt.broadcast %78 : tensor<8x16x1x!tt.ptr<bf16>> -> tensor<8x16x64x!tt.ptr<bf16>>
    %80 = tt.splat %74 : i64 -> tensor<8x16x64xi64>
    %81 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %82 = arith.extsi %81 : tensor<64xi32> to tensor<64xi64>
    %83 = tt.expand_dims %82 {axis = 0 : i32} : tensor<64xi64> -> tensor<1x64xi64>
    %84 = tt.expand_dims %83 {axis = 1 : i32} : tensor<1x64xi64> -> tensor<1x1x64xi64>
    %85 = tt.broadcast %84 : tensor<1x1x64xi64> -> tensor<8x16x64xi64>
    %86 = arith.addi %80, %85 : tensor<8x16x64xi64>
    %87 = tt.addptr %79, %86 : tensor<8x16x64x!tt.ptr<bf16>>, tensor<8x16x64xi64>
    %88 = arith.extsi %73 : i32 to i64
    %89 = tt.splat %88 : i64 -> tensor<8x16x64xi64>
    %90 = arith.addi %89, %85 : tensor<8x16x64xi64>
    %91 = tt.addptr %79, %90 : tensor<8x16x64x!tt.ptr<bf16>>, tensor<8x16x64xi64>
    %92 = tt.splat %arg4 : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %93 = tt.addptr %92, %22 : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %94 = tt.addptr %93, %24 : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %95 = tt.expand_dims %94 {axis = 1 : i32} : tensor<8x!tt.ptr<f32>> -> tensor<8x1x!tt.ptr<f32>>
    %96 = tt.broadcast %95 : tensor<8x1x!tt.ptr<f32>> -> tensor<8x16x!tt.ptr<f32>>
    %97 = tt.expand_dims %33 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %98 = tt.broadcast %97 : tensor<1x16xi32> -> tensor<8x16xi32>
    %99 = tt.addptr %96, %98 : tensor<8x16x!tt.ptr<f32>>, tensor<8x16xi32>
    %100 = arith.minsi %33, %cst_18 : tensor<16xi32>
    %101 = tt.expand_dims %27 {axis = 1 : i32} : tensor<8x!tt.ptr<i64>> -> tensor<8x1x!tt.ptr<i64>>
    %102 = tt.broadcast %101 : tensor<8x1x!tt.ptr<i64>> -> tensor<8x16x!tt.ptr<i64>>
    %103 = tt.expand_dims %100 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %104 = tt.broadcast %103 : tensor<1x16xi32> -> tensor<8x16xi32>
    %105 = tt.addptr %102, %104 : tensor<8x16x!tt.ptr<i64>>, tensor<8x16xi32>
    %106 = tt.broadcast %61 : tensor<8x1xi1> -> tensor<8x16xi1>
    %107 = tt.load %105, %106, %cst_16 : tensor<8x16x!tt.ptr<i64>>
    %108 = arith.trunci %107 : tensor<8x16xi64> to tensor<8x16xi32>
    %109 = arith.muli %108, %cst_7 : tensor<8x16xi32>
    %110 = arith.cmpi slt, %33, %cst_17 : tensor<16xi32>
    %111 = arith.addi %4, %c64_i32 : i32
    %112 = arith.addi %7, %cst_14 : tensor<8xi32>
    %113 = arith.divsi %111, %c64_i32 : i32
    %114 = arith.divsi %112, %cst_14 : tensor<8xi32>
    %115 = arith.extsi %arg8 : i32 to i64
    %116 = tt.expand_dims %109 {axis = 1 : i32} : tensor<8x16xi32> -> tensor<8x1x16xi32>
    %117 = tt.expand_dims %110 {axis = 0 : i32} : tensor<16xi1> -> tensor<1x16xi1>
    %118 = tt.expand_dims %7 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
    %119 = tt.broadcast %118 : tensor<8x1xi32> -> tensor<8x64xi32>
    %120 = tt.splat %arg8 : i32 -> tensor<64xi32>
    %121 = tt.expand_dims %117 {axis = 0 : i32} : tensor<1x16xi1> -> tensor<1x1x16xi1>
    %122 = tt.broadcast %121 : tensor<1x1x16xi1> -> tensor<8x1x16xi1>
    %123 = tt.reshape %69 : tensor<8x16x128xbf16> -> tensor<128x128xbf16>
    %124:4 = scf.for %arg11 = %c0_i32 to %113 step %c1_i32 iter_args(%arg12 = %cst_15, %arg13 = %cst_15, %arg14 = %cst_6, %arg15 = %cst_5) -> (tensor<8x16x64xf32>, tensor<8x16x64xf32>, tensor<8x16xf32>, tensor<8x16xf32>)  : i32 {
      %150 = tt.splat %arg11 : i32 -> tensor<8xi32>
      %151 = arith.cmpi slt, %150, %114 : tensor<8xi32>
      %152 = arith.muli %arg11, %c64_i32 : i32
      %153 = tt.make_tensor_ptr %18, [%c128_i64, %115], [%c1_i64, %c512_i64], [%c0_i32, %152] {order = array<i32: 0, 1>} : <tensor<128x64xbf16>>
      %154 = tt.make_tensor_ptr %19, [%115, %c128_i64], [%c512_i64, %c1_i64], [%152, %72] {order = array<i32: 1, 0>} : <tensor<64x64xbf16>>
      %155 = tt.make_tensor_ptr %19, [%115, %c128_i64], [%c512_i64, %c1_i64], [%152, %73] {order = array<i32: 1, 0>} : <tensor<64x64xbf16>>
      %156 = tt.load %153 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<128x64xbf16>>
      %157 = tt.load %154 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x64xbf16>>
      %158 = tt.load %155 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x64xbf16>>
      %159 = tt.splat %152 : i32 -> tensor<8x1x16xi32>
      %160 = arith.cmpi eq, %116, %159 : tensor<8x1x16xi32>
      %161 = arith.andi %160, %122 : tensor<8x1x16xi1>
      %162 = arith.extui %161 : tensor<8x1x16xi1> to tensor<8x1x16xi32>
      %163 = "tt.reduce"(%162) <{axis = 2 : i32}> ({
      ^bb0(%arg16: i32, %arg17: i32):
        %220 = arith.addi %arg16, %arg17 : i32
        tt.reduce.return %220 : i32
      }) : (tensor<8x1x16xi32>) -> tensor<8x1xi32>
      %164 = arith.sitofp %163 : tensor<8x1xi32> to tensor<8x1xf32>
      %165 = tt.expand_dims %164 {axis = 2 : i32} : tensor<8x1xf32> -> tensor<8x1x1xf32>
      %166 = tt.broadcast %165 : tensor<8x1x1xf32> -> tensor<8x1x64xf32>
      %167 = tt.reshape %166 : tensor<8x1x64xf32> -> tensor<8x64xf32>
      %168 = tt.splat %152 : i32 -> tensor<64xi32>
      %169 = arith.addi %168, %81 : tensor<64xi32>
      %170 = arith.cmpf ogt, %167, %cst_4 : tensor<8x64xf32>
      %171 = tt.expand_dims %169 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
      %172 = tt.broadcast %171 : tensor<1x64xi32> -> tensor<8x64xi32>
      %173 = arith.cmpi sle, %172, %119 : tensor<8x64xi32>
      %174 = arith.andi %170, %173 : tensor<8x64xi1>
      %175 = arith.cmpi slt, %169, %120 : tensor<64xi32>
      %176 = tt.expand_dims %175 {axis = 0 : i32} : tensor<64xi1> -> tensor<1x64xi1>
      %177 = tt.broadcast %176 : tensor<1x64xi1> -> tensor<8x64xi1>
      %178 = arith.andi %174, %177 : tensor<8x64xi1>
      %179 = tt.dot %123, %156, %cst_3 : tensor<128x128xbf16> * tensor<128x64xbf16> -> tensor<128x64xf32>
      %180 = tt.reshape %179 : tensor<128x64xf32> -> tensor<8x16x64xf32>
      %181 = tt.expand_dims %178 {axis = 1 : i32} : tensor<8x64xi1> -> tensor<8x1x64xi1>
      %182 = tt.broadcast %181 : tensor<8x1x64xi1> -> tensor<8x16x64xi1>
      %183 = arith.select %182, %180, %cst_2 : tensor<8x16x64xi1>, tensor<8x16x64xf32>
      %184 = "tt.reduce"(%183) <{axis = 2 : i32}> ({
      ^bb0(%arg16: f32, %arg17: f32):
        %220 = arith.maxnumf %arg16, %arg17 : f32
        tt.reduce.return %220 : f32
      }) : (tensor<8x16x64xf32>) -> tensor<8x16xf32>
      %185 = arith.maxnumf %arg14, %184 : tensor<8x16xf32>
      %186 = arith.cmpf oeq, %185, %cst_6 : tensor<8x16xf32>
      %187 = arith.select %186, %cst_5, %185 : tensor<8x16xi1>, tensor<8x16xf32>
      %188 = arith.subf %arg14, %187 : tensor<8x16xf32>
      %189 = math.exp %188 : tensor<8x16xf32>
      %190 = tt.expand_dims %187 {axis = 2 : i32} : tensor<8x16xf32> -> tensor<8x16x1xf32>
      %191 = tt.broadcast %190 : tensor<8x16x1xf32> -> tensor<8x16x64xf32>
      %192 = arith.subf %183, %191 : tensor<8x16x64xf32>
      %193 = math.exp %192 : tensor<8x16x64xf32>
      %194 = tt.expand_dims %167 {axis = 1 : i32} : tensor<8x64xf32> -> tensor<8x1x64xf32>
      %195 = tt.broadcast %194 : tensor<8x1x64xf32> -> tensor<8x16x64xf32>
      %196 = arith.mulf %193, %195 : tensor<8x16x64xf32>
      %197 = arith.mulf %arg15, %189 : tensor<8x16xf32>
      %198 = "tt.reduce"(%196) <{axis = 2 : i32}> ({
      ^bb0(%arg16: f32, %arg17: f32):
        %220 = arith.addf %arg16, %arg17 : f32
        tt.reduce.return %220 : f32
      }) : (tensor<8x16x64xf32>) -> tensor<8x16xf32>
      %199 = arith.addf %197, %198 : tensor<8x16xf32>
      %200 = tt.expand_dims %189 {axis = 2 : i32} : tensor<8x16xf32> -> tensor<8x16x1xf32>
      %201 = tt.broadcast %200 : tensor<8x16x1xf32> -> tensor<8x16x64xf32>
      %202 = arith.mulf %arg12, %201 : tensor<8x16x64xf32>
      %203 = arith.mulf %arg13, %201 : tensor<8x16x64xf32>
      %204 = arith.truncf %196 : tensor<8x16x64xf32> to tensor<8x16x64xbf16>
      %205 = tt.reshape %204 : tensor<8x16x64xbf16> -> tensor<128x64xbf16>
      %206 = tt.reshape %202 : tensor<8x16x64xf32> -> tensor<128x64xf32>
      %207 = tt.reshape %203 : tensor<8x16x64xf32> -> tensor<128x64xf32>
      %208 = tt.dot %205, %157, %206 : tensor<128x64xbf16> * tensor<64x64xbf16> -> tensor<128x64xf32>
      %209 = tt.reshape %208 : tensor<128x64xf32> -> tensor<8x16x64xf32>
      %210 = tt.dot %205, %158, %207 : tensor<128x64xbf16> * tensor<64x64xbf16> -> tensor<128x64xf32>
      %211 = tt.reshape %210 : tensor<128x64xf32> -> tensor<8x16x64xf32>
      %212 = tt.expand_dims %151 {axis = 1 : i32} : tensor<8xi1> -> tensor<8x1xi1>
      %213 = tt.expand_dims %212 {axis = 2 : i32} : tensor<8x1xi1> -> tensor<8x1x1xi1>
      %214 = tt.broadcast %213 : tensor<8x1x1xi1> -> tensor<8x16x64xi1>
      %215 = arith.select %214, %209, %arg12 : tensor<8x16x64xi1>, tensor<8x16x64xf32>
      %216 = arith.select %214, %211, %arg13 : tensor<8x16x64xi1>, tensor<8x16x64xf32>
      %217 = tt.broadcast %212 : tensor<8x1xi1> -> tensor<8x16xi1>
      %218 = arith.select %217, %185, %arg14 : tensor<8x16xi1>, tensor<8x16xf32>
      %219 = arith.select %217, %199, %arg15 : tensor<8x16xi1>, tensor<8x16xf32>
      scf.yield %215, %216, %218, %219 : tensor<8x16x64xf32>, tensor<8x16x64xf32>, tensor<8x16xf32>, tensor<8x16xf32>
    }
    %125 = tt.expand_dims %124#3 {axis = 2 : i32} : tensor<8x16xf32> -> tensor<8x16x1xf32>
    %126 = tt.broadcast %125 : tensor<8x16x1xf32> -> tensor<8x16x64xf32>
    %127 = arith.divf %124#0, %126 : tensor<8x16x64xf32>
    %128 = arith.divf %124#1, %126 : tensor<8x16x64xf32>
    %129 = math.log %124#3 : tensor<8x16xf32>
    %130 = arith.addf %124#2, %129 : tensor<8x16xf32>
    %131 = arith.truncf %127 : tensor<8x16x64xf32> to tensor<8x16x64xbf16>
    %132 = arith.truncf %128 : tensor<8x16x64xf32> to tensor<8x16x64xbf16>
    %133 = tt.splat %31 : i64 -> tensor<8x16x64xi64>
    %134 = tt.broadcast %36 : tensor<1x16x1xi64> -> tensor<8x16x64xi64>
    %135 = arith.addi %133, %134 : tensor<8x16x64xi64>
    %136 = arith.cmpi sge, %135, %cst_1 : tensor<8x16x64xi64>
    %137 = arith.cmpi slt, %135, %cst_0 : tensor<8x16x64xi64>
    %138 = arith.andi %136, %137 : tensor<8x16x64xi1>
    %139 = arith.cmpi sge, %86, %cst_1 : tensor<8x16x64xi64>
    %140 = arith.cmpi slt, %86, %cst : tensor<8x16x64xi64>
    %141 = arith.andi %139, %140 : tensor<8x16x64xi1>
    %142 = arith.andi %138, %141 : tensor<8x16x64xi1>
    %143 = tt.broadcast %62 : tensor<8x1x1xi1> -> tensor<8x16x64xi1>
    %144 = arith.andi %142, %143 : tensor<8x16x64xi1>
    tt.store %87, %131, %144 : tensor<8x16x64x!tt.ptr<bf16>>
    %145 = arith.cmpi sge, %90, %cst_1 : tensor<8x16x64xi64>
    %146 = arith.cmpi slt, %90, %cst : tensor<8x16x64xi64>
    %147 = arith.andi %145, %146 : tensor<8x16x64xi1>
    %148 = arith.andi %138, %147 : tensor<8x16x64xi1>
    %149 = arith.andi %148, %143 : tensor<8x16x64xi1>
    tt.store %91, %132, %149 : tensor<8x16x64x!tt.ptr<bf16>>
    tt.store %99, %130, %106 : tensor<8x16x!tt.ptr<f32>>
    tt.return
  }
}
