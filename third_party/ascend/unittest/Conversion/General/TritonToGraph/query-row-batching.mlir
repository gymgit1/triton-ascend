// RUN: triton-opt --split-input-file --verify-each %s -graph-optimize='rule-mask=131072' -canonicalize -cse -o %t
// RUN: FileCheck %s < %t
// A second rule run must not lift rows twice.  The canonicalizer needs its own
// second round to fold the duplicated scf.if, which is unrelated to rule B, so
// the idempotency check drives the rule and CSE only.
// RUN: triton-opt --split-input-file --verify-each %t -graph-optimize='rule-mask=131072' -cse -o %t.twice
// RUN: diff %t %t.twice
// RUN: triton-opt --split-input-file --verify-each %s -graph-optimize='rule-mask=0' -canonicalize -cse -o %t.disabled
// RUN: triton-opt --split-input-file %s -canonicalize -cse -o %t.control
// RUN: diff %t.control %t.disabled

// Rule B is Ascend 950 only: the same candidates must stay untouched elsewhere.
// RUN: sed 's/Ascend950PR_9579/Ascend910B2/g' %s > %t.a2.mlir
// RUN: triton-opt --split-input-file %t.a2.mlir -canonicalize -cse -o %t.a2.control
// RUN: triton-opt --split-input-file --verify-each %t.a2.mlir -graph-optimize='rule-mask=131072' -canonicalize -cse -o %t.a2
// RUN: diff %t.a2.control %t.a2
// Missing target information must not enable the rule either.
// RUN: sed 's/hacc.target = #hacc.target<"Ascend950PR_9579">, //g' %s > %t.no-target.mlir
// RUN: triton-opt --split-input-file %t.no-target.mlir -canonicalize -cse -o %t.no-target.control
// RUN: triton-opt --split-input-file --verify-each %t.no-target.mlir -graph-optimize='rule-mask=131072' -canonicalize -cse -o %t.no-target
// RUN: diff %t.no-target.control %t.no-target
// The row-count policy keeps one full wave: a static 64-program grid may only
// coalesce to 64/R >= 28, so R drops from 8 to 2.
// RUN: triton-opt --split-input-file --verify-each %s -graph-optimize='rule-mask=131072 core-num=28' -canonicalize -cse -o %t.corenum
// RUN: FileCheck %s --check-prefix=CORENUM < %t.corenum
// CORENUM-NOT: hacc.coalesce_factor = 8 : i32
// CORENUM: hacc.coalesce_factor = 2 : i32

// Rule B lifts the query-row axis of the loop that rule A handed over.  The
// matcher no longer judges whether the loop is sequential.  It accepts a
// pid-dependent upper bound only when the bound is provably non-decreasing over
// the whole pid range, so the last row of the group supplies the loop trip
// count and the per-row copy becomes the active mask.

// CHECK: module attributes {{.*}}hacc.coalesce_factor = 64 : i32
// CHECK-LABEL: tt.func @pid_bound_loop(
// CHECK: tt.load {{.*}} : tensor<64x16x16x!tt.ptr<f16>>
// CHECK: scf.for
// CHECK: arith.cmpi slt, {{.*}} : tensor<64xi32>
// CHECK: %[[RHS:.*]] = tt.load {{.*}} : tensor<16x16x!tt.ptr<f16>>
// CHECK: tt.dot %{{.*}}, %[[RHS]], %{{.*}} : tensor<1024x16xf16> * tensor<16x16xf16> -> tensor<1024x16xf32>
// CHECK: arith.select {{.*}} : tensor<64x16x16xi1>, tensor<64x16x16xf32>
// CHECK: tt.store {{.*}} : tensor<64x16x16x!tt.ptr<f32>>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, hacc.grid_num_tiles = 64 : i32} {
  tt.func @pid_bound_loop(%A: !tt.ptr<f16>, %B: !tt.ptr<f16>, %C: !tt.ptr<f32>) {
    %pid = tt.get_program_id x : i32
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %stride = arith.constant 256 : i32
    %base = arith.muli %pid, %stride : i32
    %ub0 = arith.muli %pid, %c2 : i32
    %ub = arith.addi %ub0, %c1 : i32
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
    %zero0 = arith.constant dense<0.0> : tensor<16x16xf32>
    %loop = scf.for %iv = %c0 to %ub step %c1 iter_args(%acc0 = %zero0) -> (tensor<16x16xf32>) : i32 {
      %b = tt.load %pb : tensor<16x16x!tt.ptr<f16>>
      %dot0 = tt.dot %a, %b, %acc0 : tensor<16x16xf16> * tensor<16x16xf16> -> tensor<16x16xf32>
      scf.yield %dot0 : tensor<16x16xf32>
    }
    %cp = tt.splat %C : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
    %pc = tt.addptr %cp, %offset : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    tt.store %pc, %loop : tensor<16x16x!tt.ptr<f32>>
    tt.return
  }
}
// -----

// A row-invariant scf.if may select between two monotone bound chains: the
// envelope of both branches is monotone, and every row takes the same branch.

// CHECK: module attributes {{.*}}hacc.coalesce_factor = 64 : i32
// CHECK-LABEL: tt.func @row_invariant_if_bound(
// CHECK: scf.if {{.*}} -> (i32, tensor<64xi32>)
// CHECK: tt.load {{.*}} : tensor<64x16x16x!tt.ptr<f16>>
// CHECK: scf.for
// CHECK: tt.dot %{{.*}}, %{{.*}}, %{{.*}} : tensor<1024x16xf16> * tensor<16x16xf16> -> tensor<1024x16xf32>
// CHECK: arith.select {{.*}} : tensor<64x16x16xi1>, tensor<64x16x16xf32>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, hacc.grid_num_tiles = 64 : i32} {
  tt.func @row_invariant_if_bound(%A: !tt.ptr<f16>, %B: !tt.ptr<f16>, %C: !tt.ptr<f32>) {
    %pid = tt.get_program_id x : i32
    %pid1 = tt.get_program_id y : i32
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c3 = arith.constant 3 : i32
    %stride = arith.constant 256 : i32
    %base0 = arith.muli %pid, %stride : i32
    %yz = arith.muli %pid1, %stride : i32
    %base = arith.addi %base0, %yz : i32
    %flag = arith.cmpi sgt, %pid1, %c0 : i32
    %ub = scf.if %flag -> (i32) {
      %ub1a = arith.muli %pid, %c2 : i32
      %ub1 = arith.addi %ub1a, %c1 : i32
      scf.yield %ub1 : i32
    } else {
      %ub2a = arith.muli %pid, %c3 : i32
      %ub2 = arith.addi %ub2a, %c2 : i32
      scf.yield %ub2 : i32
    }
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
    %zero0 = arith.constant dense<0.0> : tensor<16x16xf32>
    %loop = scf.for %iv = %c0 to %ub step %c1 iter_args(%acc0 = %zero0) -> (tensor<16x16xf32>) : i32 {
      %b = tt.load %pb : tensor<16x16x!tt.ptr<f16>>
      %dot0 = tt.dot %a, %b, %acc0 : tensor<16x16xf16> * tensor<16x16xf16> -> tensor<16x16xf32>
      scf.yield %dot0 : tensor<16x16xf32>
    }
    %cp = tt.splat %C : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
    %pc = tt.addptr %cp, %offset : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    tt.store %pc, %loop : tensor<16x16x!tt.ptr<f32>>
    tt.return
  }
}
// -----

// A pid-invariant tt.reduce result shifts a monotone bound chain.  Its value is
// unknown, so the prover accepts the chain as monotone but never invents a
// numeric range for it.

// CHECK: module attributes {{.*}}hacc.coalesce_factor = 64 : i32
// CHECK-LABEL: tt.func @reduce_shifted_bound(
// CHECK: tt.reduce
// CHECK: tt.load {{.*}} : tensor<64x16x16x!tt.ptr<f16>>
// CHECK: scf.for
// CHECK: tt.dot %{{.*}}, %{{.*}}, %{{.*}} : tensor<1024x16xf16> * tensor<16x16xf16> -> tensor<1024x16xf32>
// CHECK: arith.select {{.*}} : tensor<64x16x16xi1>, tensor<64x16x16xf32>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, hacc.grid_num_tiles = 64 : i32} {
  tt.func @reduce_shifted_bound(%A: !tt.ptr<f16>, %B: !tt.ptr<f16>, %C: !tt.ptr<f32>) {
    %pid = tt.get_program_id x : i32
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %stride = arith.constant 256 : i32
    %base = arith.muli %pid, %stride : i32
    %cst = arith.constant dense<4> : tensor<16xi32>
    %red = "tt.reduce"(%cst) <{axis = 0 : i32}> ({
    ^bb0(%x: i32, %y: i32):
      %s = arith.addi %x, %y : i32
      tt.reduce.return %s : i32
    }) : (tensor<16xi32>) -> i32
    %ub0 = arith.muli %pid, %c2 : i32
    %ub = arith.addi %ub0, %red : i32
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
    %zero0 = arith.constant dense<0.0> : tensor<16x16xf32>
    %loop = scf.for %iv = %c0 to %ub step %c1 iter_args(%acc0 = %zero0) -> (tensor<16x16xf32>) : i32 {
      %b = tt.load %pb : tensor<16x16x!tt.ptr<f16>>
      %dot0 = tt.dot %a, %b, %acc0 : tensor<16x16xf16> * tensor<16x16xf16> -> tensor<16x16xf32>
      scf.yield %dot0 : tensor<16x16xf32>
    }
    %cp = tt.splat %C : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
    %pc = tt.addptr %cp, %offset : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    tt.store %pc, %loop : tensor<16x16x!tt.ptr<f32>>
    tt.return
  }
}
// -----

// A decreasing pid-dependent bound has no computable group trip count: the
// merge is declined and the function is left exactly as it was.

// CHECK-NOT: hacc.coalesce_factor
// CHECK-LABEL: tt.func @nonmonotone_pid_bound(
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args
// CHECK: tt.dot %{{.*}}, %{{.*}}, %{{.*}} : tensor<16x16xf16> * tensor<16x16xf16> -> tensor<16x16xf32>
// CHECK: tt.store {{.*}} : tensor<16x16x!tt.ptr<f32>>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, hacc.grid_num_tiles = 64 : i32} {
  tt.func @nonmonotone_pid_bound(%A: !tt.ptr<f16>, %B: !tt.ptr<f16>, %C: !tt.ptr<f32>) {
    %pid = tt.get_program_id x : i32
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %stride = arith.constant 256 : i32
    %base = arith.muli %pid, %stride : i32
    %ub = arith.subi %c2, %pid : i32
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
    %zero0 = arith.constant dense<0.0> : tensor<16x16xf32>
    %loop = scf.for %iv = %c0 to %ub step %c1 iter_args(%acc0 = %zero0) -> (tensor<16x16xf32>) : i32 {
      %b = tt.load %pb : tensor<16x16x!tt.ptr<f16>>
      %dot0 = tt.dot %a, %b, %acc0 : tensor<16x16xf16> * tensor<16x16xf16> -> tensor<16x16xf32>
      scf.yield %dot0 : tensor<16x16xf32>
    }
    %cp = tt.splat %C : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
    %pc = tt.addptr %cp, %offset : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    tt.store %pc, %loop : tensor<16x16x!tt.ptr<f32>>
    tt.return
  }
}
// -----

// A loop that still reads its block index from a scalar table is lifted as
// well: rule A owns the sequentialization, so rule B no longer infers "has A
// run?" from an in-loop table read.  The table pointer is lifted with the query
// rows, so every row reads its own index lane under the per-row active mask.

// CHECK: module attributes {{.*}}hacc.coalesce_factor = 64 : i32
// CHECK-LABEL: tt.func @scalar_table_load_in_loop(
// CHECK: scf.for
// CHECK: tt.load {{.*}} : tensor<64x!tt.ptr<i32>>
// CHECK: tt.load {{.*}} : tensor<64x16x16x!tt.ptr<f16>>
// CHECK: tt.dot %{{.*}}, %{{.*}}, %{{.*}} : tensor<1024x16xf16> * tensor<16x16xf16> -> tensor<1024x16xf32>
// CHECK: arith.select {{.*}} : tensor<64x16x16xi1>, tensor<64x16x16xf32>
// CHECK: tt.store {{.*}} : tensor<64x16x16x!tt.ptr<f32>>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, hacc.grid_num_tiles = 64 : i32} {
  tt.func @scalar_table_load_in_loop(%A: !tt.ptr<f16>, %B: !tt.ptr<f16>, %C: !tt.ptr<f32>, %S: !tt.ptr<i32>) {
    %pid = tt.get_program_id x : i32
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %stride = arith.constant 256 : i32
    %base = arith.muli %pid, %stride : i32
    %ub0 = arith.muli %pid, %c2 : i32
    %ub = arith.addi %ub0, %c1 : i32
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
    %bp = tt.splat %B : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>>
    %pb = tt.addptr %bp, %local : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi32>
    %si = tt.addptr %S, %pid : !tt.ptr<i32>, i32
    %one = arith.constant dense<1> : tensor<16x16xi32>
    %cm = arith.cmpi slt, %local, %one : tensor<16x16xi32>
    %zero0 = arith.constant dense<0.0> : tensor<16x16xf32>
    %loop = scf.for %iv = %c0 to %ub step %c1 iter_args(%acc0 = %zero0) -> (tensor<16x16xf32>) : i32 {
      %sp = tt.addptr %si, %iv : !tt.ptr<i32>, i32
      %blk = tt.load %sp : !tt.ptr<i32>
      %blks = tt.splat %blk : i32 -> tensor<16x16xi32>
      %pqa = tt.addptr %pa, %blks : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi32>
      %a = tt.load %pqa : tensor<16x16x!tt.ptr<f16>>
      %b = tt.load %pb : tensor<16x16x!tt.ptr<f16>>
      %dot0 = tt.dot %a, %b, %acc0 : tensor<16x16xf16> * tensor<16x16xf16> -> tensor<16x16xf32>
      scf.yield %dot0 : tensor<16x16xf32>
    }
    %cp = tt.splat %C : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
    %pc = tt.addptr %cp, %offset : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    tt.store %pc, %loop, %cm : tensor<16x16x!tt.ptr<f32>>
    tt.return
  }
}
// -----

// A runtime trip count is lifted too.  The bound stays a scalar load because it
// is pid-invariant, and each row's active mask ends its own iterations.

// CHECK: module attributes {{.*}}hacc.coalesce_factor = 64 : i32
// CHECK-LABEL: tt.func @table_read_runtime_bound(
// CHECK: tt.load {{.*}} : !tt.ptr<i32>
// CHECK: scf.for
// CHECK: tt.load {{.*}} : tensor<64x!tt.ptr<i32>>
// CHECK: tt.load {{.*}} : tensor<64x16x16x!tt.ptr<f16>>
// CHECK: tt.dot %{{.*}}, %{{.*}}, %{{.*}} : tensor<1024x16xf16> * tensor<16x16xf16> -> tensor<1024x16xf32>
// CHECK: arith.select {{.*}} : tensor<64x16x16xi1>, tensor<64x16x16xf32>
// CHECK: tt.store {{.*}} : tensor<64x16x16x!tt.ptr<f32>>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, hacc.grid_num_tiles = 64 : i32} {
  tt.func @table_read_runtime_bound(%A: !tt.ptr<f16>, %B: !tt.ptr<f16>, %C: !tt.ptr<f32>, %S: !tt.ptr<i32>, %N: !tt.ptr<i32>) {
    %pid = tt.get_program_id x : i32
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %stride = arith.constant 256 : i32
    %base = arith.muli %pid, %stride : i32
    %ub = tt.load %N : !tt.ptr<i32>
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
    %bp = tt.splat %B : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>>
    %pb = tt.addptr %bp, %local : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi32>
    %si = tt.addptr %S, %pid : !tt.ptr<i32>, i32
    %zero0 = arith.constant dense<0.0> : tensor<16x16xf32>
    %loop = scf.for %iv = %c0 to %ub step %c1 iter_args(%acc0 = %zero0) -> (tensor<16x16xf32>) : i32 {
      %sp = tt.addptr %si, %iv : !tt.ptr<i32>, i32
      %blk = tt.load %sp : !tt.ptr<i32>
      %blks = tt.splat %blk : i32 -> tensor<16x16xi32>
      %pqa = tt.addptr %pa, %blks : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi32>
      %a = tt.load %pqa : tensor<16x16x!tt.ptr<f16>>
      %b = tt.load %pb : tensor<16x16x!tt.ptr<f16>>
      %dot0 = tt.dot %a, %b, %acc0 : tensor<16x16xf16> * tensor<16x16xf16> -> tensor<16x16xf32>
      scf.yield %dot0 : tensor<16x16xf32>
    }
    %cp = tt.splat %C : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
    %pc = tt.addptr %cp, %offset : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    tt.store %pc, %loop : tensor<16x16x!tt.ptr<f32>>
    tt.return
  }
}
