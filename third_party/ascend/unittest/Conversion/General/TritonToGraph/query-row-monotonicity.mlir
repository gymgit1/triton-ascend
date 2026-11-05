// Under the no-wrap input contract, ordered arithmetic with runtime offsets
// uses the last row. Negative multipliers, generic selects and narrowing casts
// remain unsupported and must leave the original kernel unchanged.
// RUN: split-file %S/query-row-bound-proof.mlir %t
// RUN: sed 's/arith.addi %pid, %delta/arith.addi %pid, %bias overflow<nsw>/' %t/kernel.mlir > %t/nsw.mlir
// RUN: triton-opt %t/nsw.mlir --verify-each -graph-optimize='rule-mask=131072' -canonicalize -cse | FileCheck %s --check-prefix=LAST
// RUN: triton-opt %t/nsw.mlir --verify-each -graph-optimize='rule-mask=8' -canonicalize -cse | FileCheck %s --check-prefix=LAST
// RUN: sed 's/overflow<nsw>/overflow<nuw>/' %t/nsw.mlir > %t/nuw.mlir
// RUN: triton-opt %t/nuw.mlir --verify-each -graph-optimize='rule-mask=131072' | FileCheck %s --check-prefix=LAST
// RUN: triton-opt %t/nuw.mlir --verify-each -graph-optimize='rule-mask=8' | FileCheck %s --check-prefix=LAST
// RUN: sed 's/%calc = arith.addi %pid, %delta/%flag = arith.cmpi sge, %bias, %c0 : i32\n    %calc = arith.select %flag, %pid, %bias/' %t/kernel.mlir > %t/select.mlir
// RUN: triton-opt %t/select.mlir --verify-each -graph-optimize='rule-mask=131072' -canonicalize -cse | FileCheck %s --check-prefix=DECLINE
// RUN: triton-opt %t/select.mlir --verify-each -graph-optimize='rule-mask=8' -canonicalize -cse | FileCheck %s --check-prefix=DECLINE
// RUN: sed 's/arith.addi %pid, %delta/arith.maxsi %pid, %bias/' %t/kernel.mlir > %t/max.mlir
// RUN: triton-opt %t/max.mlir --verify-each -graph-optimize='rule-mask=131072' -canonicalize -cse | FileCheck %s --check-prefix=DECLINE
// RUN: triton-opt %t/max.mlir --verify-each -graph-optimize='rule-mask=8' -canonicalize -cse | FileCheck %s --check-prefix=DECLINE
// RUN: sed 's/%divisor =/%c64 = arith.constant 64 : i32\n    %divisor =/; s/%calc = arith.addi %pid, %delta/%lo = arith.cmpi sge, %bias, %c0 : i32\n    %hi = arith.cmpi sle, %bias, %c64 : i32\n    %flag = arith.andi %lo, %hi : i1\n    %biased = arith.addi %pid, %bias : i32\n    %calc = arith.select %flag, %biased, %pid/' %t/kernel.mlir > %t/guard.mlir
// RUN: triton-opt %t/guard.mlir --verify-each -graph-optimize='rule-mask=131072' -canonicalize -cse | FileCheck %s --check-prefix=DECLINE
// RUN: triton-opt %t/guard.mlir --verify-each -graph-optimize='rule-mask=8' -canonicalize -cse | FileCheck %s --check-prefix=DECLINE
// RUN: sed 's/arith.divsi %calc, %divisor/arith.divsi %biased, %divisor/' %t/guard.mlir > %t/unguarded.mlir
// RUN: triton-opt %t/unguarded.mlir --verify-each -graph-optimize='rule-mask=131072' | FileCheck %s --check-prefix=LAST
// RUN: triton-opt %t/unguarded.mlir --verify-each -graph-optimize='rule-mask=8' | FileCheck %s --check-prefix=LAST
// RUN: sed 's/%calc = arith.addi/%biased2 = arith.addi/; s/%ub =/%factor = arith.constant 2 : i32\n    %calc = arith.muli %biased2, %factor overflow<nsw> : i32\n    %ub =/' %t/nsw.mlir > %t/nsw-mul.mlir
// RUN: triton-opt %t/nsw-mul.mlir --verify-each -graph-optimize='rule-mask=131072' -canonicalize -cse | FileCheck %s --check-prefix=LAST
// RUN: triton-opt %t/nsw-mul.mlir --verify-each -graph-optimize='rule-mask=8' -canonicalize -cse | FileCheck %s --check-prefix=LAST
// RUN: sed 's/%biased2, %factor overflow<nsw>/%biased2, %factor/' %t/nsw-mul.mlir > %t/mul-wrap.mlir
// RUN: triton-opt %t/mul-wrap.mlir --verify-each -graph-optimize='rule-mask=131072' -canonicalize -cse | FileCheck %s --check-prefix=LAST
// RUN: triton-opt %t/mul-wrap.mlir --verify-each -graph-optimize='rule-mask=8' -canonicalize -cse | FileCheck %s --check-prefix=LAST
// RUN: sed 's/constant 2 : i32/constant -2 : i32/' %t/nsw-mul.mlir > %t/negative-mul.mlir
// RUN: triton-opt %t/negative-mul.mlir --verify-each -graph-optimize='rule-mask=131072' -canonicalize -cse | FileCheck %s --check-prefix=DECLINE
// RUN: triton-opt %t/negative-mul.mlir --verify-each -graph-optimize='rule-mask=8' -canonicalize -cse | FileCheck %s --check-prefix=DECLINE
// RUN: sed 's/%calc = arith.addi/%biased2 = arith.addi/; s/%ub =/%calc = arith.addi %biased2, %delta : i32\n    %ub =/' %t/nsw.mlir > %t/after-nsw-wrap.mlir
// RUN: triton-opt %t/after-nsw-wrap.mlir --verify-each -graph-optimize='rule-mask=131072' -canonicalize -cse | FileCheck %s --check-prefix=LAST
// RUN: triton-opt %t/after-nsw-wrap.mlir --verify-each -graph-optimize='rule-mask=8' -canonicalize -cse | FileCheck %s --check-prefix=LAST
// RUN: sed 's/%bias: i32/%bias: i64/; s/%calc = arith.addi %pid, %delta : i32/%pid64 = arith.extsi %pid : i32 to i64\n    %wide_sum = arith.addi %pid64, %bias overflow<nsw> : i64\n    %calc = arith.trunci %wide_sum : i64 to i32/' %t/kernel.mlir > %t/trunc-wrap.mlir
// RUN: triton-opt %t/trunc-wrap.mlir --verify-each -graph-optimize='rule-mask=131072' -canonicalize -cse | FileCheck %s --check-prefix=DECLINE
// RUN: triton-opt %t/trunc-wrap.mlir --verify-each -graph-optimize='rule-mask=8' -canonicalize -cse | FileCheck %s --check-prefix=DECLINE
// RUN: sed 's/%ub = arith.divsi %calc, %divisor : i32/%quot = arith.divsi %calc, %divisor : i32\n    %ub = arith.addi %quot, %c1 : i32/' %t/nsw.mlir > %t/div-add.mlir
// RUN: triton-opt %t/div-add.mlir --verify-each -graph-optimize='rule-mask=131072' -canonicalize -cse | FileCheck %s --check-prefix=LAST
// RUN: triton-opt %t/div-add.mlir --verify-each -graph-optimize='rule-mask=8' -canonicalize -cse | FileCheck %s --check-prefix=LAST
// RUN: sed 's/%ub = arith.divsi %calc, %divisor : i32/%wide = arith.extsi %calc : i32 to i64\n    %div64 = arith.constant 2147483647 : i64\n    %one64 = arith.constant 1 : i64\n    %quot64 = arith.divsi %wide, %div64 : i64\n    %wide_sum = arith.addi %quot64, %one64 : i64\n    %ub = arith.trunci %wide_sum : i64 to i32/' %t/nsw.mlir > %t/widen.mlir
// RUN: triton-opt %t/widen.mlir --verify-each -graph-optimize='rule-mask=131072' -canonicalize -cse | FileCheck %s --check-prefix=DECLINE
// RUN: triton-opt %t/widen.mlir --verify-each -graph-optimize='rule-mask=8' -canonicalize -cse | FileCheck %s --check-prefix=DECLINE

// LAST: hacc.coalesce_factor
// LAST-LABEL: tt.func @bound_proof(
// LAST-NOT: "tt.reduce"
// LAST-NOT: arith.maxsi
// LAST: scf.for
// LAST: arith.select

// DECLINE-NOT: hacc.coalesce_factor
// DECLINE-LABEL: tt.func @bound_proof(
// DECLINE: tt.dot {{.*}} : tensor<16x16xf16> * tensor<16x16xf16> -> tensor<16x16xf32>
