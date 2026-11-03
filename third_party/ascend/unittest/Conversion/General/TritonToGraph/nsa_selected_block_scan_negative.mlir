// === #2686 Rule A (SelectedBlockScanRule, bit 65536) — NEGATIVE cases ===
//
// STATUS: post-implementation form; every RUN line below is green.  What the
// file now covers: (b) K/V anchored on the grid axis instead of the
// index table, (d) the table address is not the induction variable, (e) the
// causal mask carries a non-zero token-side shift, (f) the tile is not exactly
// one block wide on the token axis.  Sections kept the original lettering so the
// (c) hole below is visible rather than silently renumbered away.
//
// (g) is GONE (deleted 2026-11-02, not renumbered): it used to pin a refusal for
// the bwd dq additive form (Inputs/nsa_bwd_dq_const_ns.mlir).  Rule A now
// ACCEPTS that shape -- the additive branch (single iter_arg carried state,
// tt.addptr + explicit-mask tiles, exp -> where -> *w -> (do v - delta) ->
// dot(ds, trans(k), acc)) landed -- so the fixture is a POSITIVE case whose
// assertions live in nsa_selected_block_scan.mlir, group G, check prefix
// BWDREW.  Nothing about it belongs in this file any more: "declines and is
// byte-identical" is the opposite of the current contract.
//
// What "must hold" means here, precisely: with the escape hatch OPEN (so Rule B
// cannot be the reason nothing happened), Rule A's own matcher must still
// decline these shapes, therefore the output must be BYTE-IDENTICAL to the
// rule-off run, and triton-opt must not print a single line on stderr.  Both
// halves are asserted:
//   * diff  -> "no change"            (repo precedent row-coalescing-dot-fallback.mlir:1-7)
//   * FileCheck -> "the pre-rewrite pattern is really there", so the diff cannot
//     pass because the fixture silently stopped containing the shape.
// Why force-apply and not just rule-mask=65536: without it Rule A hands its
// sandbox to Rule B's real implementation, which declines these shapes on its
// own account, so the "no change" assertion would prove nothing about Rule A's
// matcher.  The escape hatch is written out as
// `selected-block-scan-force-apply=true`, like every other bool option in this
// repo, instead of as a bare bool leaning on MLIR's implicit true.
//
// The `echo -n` + `diff` pair is the "no diagnostics on the decline path" check:
// an unsupported rewrite must leave the input AND its diagnostics untouched.
// The empty file is created ONCE here, before the first section that needs it
// (section (c) below uses it too; it used to be created in section (a), i.e.
// after its first use, which only worked on a dirty Output dir).
// RUN: echo -n > %t.empty
//
// ---------- (c) BS is not a power of two -----------------------------------
// BS = 48 is not a power of two, so "one index-table entry == BS consecutive
// tokens" (the weight semantics) is not expressible and the rewrite must
// decline (SelectedBlockScanRule.cpp:927, isPowerOfTwo(blockSize)).
//
// Why this fixture is legal HERE: a 128x48 tile has a non-power-of-two element
// count (6144) that OpTrait::impl::verifyTensorSize rejects -- but this fork's
// build applies third_party/ascend/patch/triton-ascend-3.6.0.patch, which
// relaxes exactly that check ("FIXME:patched triton community").  So judge this
// case on the patched tree (what setup.py / CI build); on a tree where someone
// ran restore_sources.py the fixture does not even parse.
//
// The sibling guard is unreachable for any legal BS: after isPowerOfTwo(:927)
// and BN = max(64, BS)(:929), `scanTileWidth % blockSize != 0`(:930) can never
// fire.  The BS < 64 positive side (R = BN/BS = 2) is group F of
// nsa_selected_block_scan.mlir.
// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns_bs48.mlir -graph-optimize='rule-mask=0' -canonicalize -cse -o %t.bs48.ctl
// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns_bs48.mlir -graph-optimize='rule-mask=65536 selected-block-scan-force-apply=true' -canonicalize -cse -o %t.bs48
// RUN: diff %t.bs48.ctl %t.bs48
// RUN: FileCheck %s --check-prefix=BS48 < %t.bs48
// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns_bs48.mlir -graph-optimize='rule-mask=65536 selected-block-scan-force-apply=true' -canonicalize -cse -o /dev/null 2> %t.bs48.err
// RUN: diff %t.empty %t.bs48.err

// BS48-LABEL: tt.func public @parallel_nsa_fwd_kernel(
// BS48: %[[TOK48:.*]] = arith.muli %{{.*}}, %c48_i32 : i32
// BS48: scf.if %{{.*}} -> (tensor<16x128xf32>, tensor<16xf32>, tensor<16xf32>)
// BS48: tt.make_tensor_ptr %{{.*}}, [%c128_i64, %{{.*}}], [%c1_i64, %c512_i64], [%c0_i32, %[[TOK48]]] {order = array<i32: 0, 1>} : <tensor<128x48xbf16>>
// BS48: tt.make_range {end = 48 : i32, start = 0 : i32} : tensor<48xi32>
// BS48: arith.select %{{.*}}, %{{.*}}, %{{.*}} : tensor<16x48xi1>, tensor<16x48xf32>
// BS48: "tt.reduce"(

// ---------- (a) varlen: the joint gate still declines it -------------------
// (2026-11-03) Rule A now RECOGNIZES the varlen dispatch: the row position
// (token_indices_q) and the K/V base (cu_seqlens) are loop-invariant scalar
// loads, which is exactly what matchRowPosition / checkAddressOperand accept
// after the varlen relaxation.  What must still hold is the JOINT contract:
// with only Rule A's bit enabled and no escape hatch, the function is left
// BYTE-IDENTICAL because Rule B declines varlen and A must never land alone
// (A alone trades O(NS) block reads for an O(TK) token scan).  The rewritten
// shape under the escape hatch is asserted separately as group VARLENREW in
// nsa_selected_block_scan.mlir.
// RUN: triton-opt %S/Inputs/nsa_fwd_varlen.mlir -graph-optimize='rule-mask=0' -canonicalize -cse -o %t.varlen.ctl
// RUN: triton-opt %S/Inputs/nsa_fwd_varlen.mlir -graph-optimize='rule-mask=65536' -canonicalize -cse -o %t.varlen
// RUN: diff %t.varlen.ctl %t.varlen
// RUN: FileCheck %s --check-prefix=VARLEN < %t.varlen
// RUN: triton-opt %S/Inputs/nsa_fwd_varlen.mlir -graph-optimize='rule-mask=65536' -canonicalize -cse -o /dev/null 2> %t.varlen.err
// RUN: diff %t.empty %t.varlen.err

// VARLEN-LABEL: tt.func public @parallel_nsa_fwd_kernel(
// VARLEN-COUNT-4: tt.load %{{.*}} : !tt.ptr<i64>
// VARLEN: %[[NS:.*]] = tt.load %{{.*}} : !tt.ptr<i64>
// VARLEN: scf.for %{{.*}} = %c0_i64 to %[[NS]] step %c1_i64
// VARLEN: %[[TOK:.*]] = arith.muli %{{.*}}, %c64_i32 : i32
// VARLEN: scf.if %{{.*}} -> (tensor<16x128xf32>, tensor<16xf32>, tensor<16xf32>)
// VARLEN: tt.make_tensor_ptr %{{.*}}, [%c128_i64, %{{.*}}], [%c1_i64, %c512_i64], [%c0_i32, %[[TOK]]] {order = array<i32: 0, 1>} : <tensor<128x64xbf16>>
// VARLEN: "tt.reduce"(

// ---------- (b) the RHS is anchored on the grid axis ------------------------
// K and V tile token offsets are program_id.x (= the query position) instead of
// the index-table entry: the address chain no longer goes through the scanned
// block index, so this is a dense/window read, not a selected-block scan.
// Downstream reason: Rules/DotRowCoalescing.cpp:294-306 (a lifted,
// pid-dependent dot B operand must be refused).
// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns_rhs_grid.mlir -graph-optimize='rule-mask=0' -canonicalize -cse -o %t.rhs.ctl
// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns_rhs_grid.mlir -graph-optimize='rule-mask=65536 selected-block-scan-force-apply=true' -canonicalize -cse -o %t.rhs
// RUN: diff %t.rhs.ctl %t.rhs
// RUN: FileCheck %s --check-prefix=RHSGRID < %t.rhs
// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns_rhs_grid.mlir -graph-optimize='rule-mask=65536 selected-block-scan-force-apply=true' -canonicalize -cse -o /dev/null 2> %t.rhs.err
// RUN: diff %t.empty %t.rhs.err

// RHSGRID-LABEL: tt.func public @parallel_nsa_fwd_kernel(
// RHSGRID: %[[PIDX:.*]] = tt.get_program_id x : i32
// RHSGRID: scf.if %{{.*}} -> (tensor<16x128xf32>, tensor<16xf32>, tensor<16xf32>)
// RHSGRID: tt.make_tensor_ptr %{{.*}}, [%c128_i64, %{{.*}}], [%c1_i64, %c512_i64], [%c0_i32, %[[PIDX]]] {order = array<i32: 0, 1>} : <tensor<128x64xbf16>>
// RHSGRID: tt.make_tensor_ptr %{{.*}}, [%{{.*}}, %c128_i64], [%c512_i64, %c1_i64], [%[[PIDX]], %{{.*}}] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
// RHSGRID: "tt.reduce"(

// ---------- (d) the index-table address is not the induction variable ------
// The per-iteration table entry is addressed as table[P + i_h*S + (i + 1)]:
// the offset operand of the scalar tt.addptr is arith.addi(%iv, %c1_i32), so
// the baseline reads table[1..S+1).  Rule A's rewrite reads the whole table
// once as table[0, S) and then indexes it by the iteration number, which is
// the same slot only when that offset is exactly the induction variable; an
// a*iv+b form (scale != 1 or bias != 0) therefore pins the wrong slots and is
// refused (matcher A2/B1).  Measured before the fix: the rewrite went through
// and dropped the +1 (163 diff lines, empty stderr).
// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns_affine_offset.mlir -graph-optimize='rule-mask=0' -canonicalize -cse -o %t.aff.ctl
// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns_affine_offset.mlir -graph-optimize='rule-mask=65536 selected-block-scan-force-apply=true' -canonicalize -cse -o %t.aff
// RUN: diff %t.aff.ctl %t.aff
// RUN: FileCheck %s --check-prefix=AFFOFF < %t.aff
// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns_affine_offset.mlir -graph-optimize='rule-mask=65536 selected-block-scan-force-apply=true' -canonicalize -cse -o /dev/null 2> %t.aff.err
// RUN: diff %t.empty %t.aff.err

// AFFOFF-LABEL: tt.func public @parallel_nsa_fwd_kernel(
// AFFOFF: %[[BIA:.*]] = arith.addi %{{.*}}, %c1_i32 : i32
// AFFOFF: %{{.*}} = tt.addptr %{{.*}}, %[[BIA]] : !tt.ptr<i64>, i32
// AFFOFF: %[[TOK:.*]] = arith.muli %{{.*}}, %c64_i32 : i32
// AFFOFF: scf.if %{{.*}} -> (tensor<16x128xf32>, tensor<16xf32>, tensor<16xf32>)
// AFFOFF: tt.make_tensor_ptr %{{.*}}, [%c128_i64, %{{.*}}], [%c1_i64, %c512_i64], [%c0_i32, %[[TOK]]] {order = array<i32: 0, 1>} : <tensor<128x64xbf16>>
// AFFOFF: "tt.reduce"(

// ---------- (e) the causal mask has a non-zero token offset ---------------
// The tensor-level causal mask compares (i_s + [0, BS) + 1) <= i_t, i.e. the
// token side carries a diagonal = 1 shift.  The rewrite rebuilds the token
// positions itself as token_start + [0, BN) with no offset, so the shift would
// be dropped (and with it one token per block moves in/out of the causal
// window).  Rule A now requires the token side to be exactly
// splat(token_offset) + tt.make_range{start = 0, end = BS} over the same SSA
// value as a tile token offset (matcher A8 site 3 / B2); any extra arithmetic
// is refused instead of normalized.  Measured before the fix: rewrite went
// through, +1 gone (164 diff lines, empty stderr).
// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns_mask_diagonal.mlir -graph-optimize='rule-mask=0' -canonicalize -cse -o %t.maskd.ctl
// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns_mask_diagonal.mlir -graph-optimize='rule-mask=65536 selected-block-scan-force-apply=true' -canonicalize -cse -o %t.maskd
// RUN: diff %t.maskd.ctl %t.maskd
// RUN: FileCheck %s --check-prefix=MASKD < %t.maskd
// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns_mask_diagonal.mlir -graph-optimize='rule-mask=65536 selected-block-scan-force-apply=true' -canonicalize -cse -o /dev/null 2> %t.maskd.err
// RUN: diff %t.empty %t.maskd.err

// MASKD-LABEL: tt.func public @parallel_nsa_fwd_kernel(
// MASKD: %[[DIDX:.*]] = arith.trunci %{{.*}} : i64 to i32
// MASKD: %[[DTOK:.*]] = arith.muli %[[DIDX]], %c64_i32 : i32
// MASKD: scf.if %{{.*}} -> (tensor<16x128xf32>, tensor<16xf32>, tensor<16xf32>)
// MASKD: tt.make_tensor_ptr %{{.*}}, [%c128_i64, %{{.*}}], [%c1_i64, %c512_i64], [%c0_i32, %[[DTOK]]] {order = array<i32: 0, 1>} : <tensor<128x64xbf16>>
// MASKD: %[[DRNG:.*]] = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
// MASKD: %[[DSPL:.*]] = tt.splat %[[DTOK]] : i32 -> tensor<64xi32>
// MASKD: %[[DBASE:.*]] = arith.addi %[[DSPL]], %[[DRNG]] : tensor<64xi32>
// MASKD: %[[DSHIFT:.*]] = arith.addi %[[DBASE]], %{{.*}} : tensor<64xi32>
// MASKD: arith.select %{{.*}}, %{{.*}}, %{{.*}} : tensor<16x64xi1>, tensor<16x64xf32>
// MASKD: "tt.reduce"(

// ---------- (f) the tile is not one block wide on the token axis -----------
// The K/V tile views (and the tensors mirroring them) are 32 tokens wide while
// the token offset still advances by BS = 64 per iteration: a legal module
// whose width/stride agreement is broken.  blockSize is read off the
// token-offset multiplier, and the rewrite then pins the tile token extent to
// BN = 64 unconditionally, so it would widen the tile and change how many
// tokens a selected block accumulates.  Rule A now requires the pointee
// shape[tokenAxis] of both tile pointers to equal blockSize (matcher A4/B3).
// Measured before the fix: rewrite went through, '<tensor<128x32xbf16>>'
// became '<tensor<128x64xbf16>>' (162 diff lines, empty stderr).
// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns_tile_width.mlir -graph-optimize='rule-mask=0' -canonicalize -cse -o %t.tilew.ctl
// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns_tile_width.mlir -graph-optimize='rule-mask=65536 selected-block-scan-force-apply=true' -canonicalize -cse -o %t.tilew
// RUN: diff %t.tilew.ctl %t.tilew
// RUN: FileCheck %s --check-prefix=TILEW < %t.tilew
// RUN: triton-opt %S/Inputs/nsa_fwd_const_ns_tile_width.mlir -graph-optimize='rule-mask=65536 selected-block-scan-force-apply=true' -canonicalize -cse -o /dev/null 2> %t.tilew.err
// RUN: diff %t.empty %t.tilew.err

// TILEW-LABEL: tt.func public @parallel_nsa_fwd_kernel(
// TILEW: %[[WIDX:.*]] = arith.trunci %{{.*}} : i64 to i32
// TILEW: %[[WTOK:.*]] = arith.muli %[[WIDX]], %c64_i32 : i32
// TILEW: scf.if %{{.*}} -> (tensor<16x128xf32>, tensor<16xf32>, tensor<16xf32>)
// TILEW: tt.make_tensor_ptr %{{.*}}, [%c128_i64, %{{.*}}], [%c1_i64, %c512_i64], [%c0_i32, %[[WTOK]]] {order = array<i32: 0, 1>} : <tensor<128x32xbf16>>
// TILEW: tt.make_tensor_ptr %{{.*}}, [%{{.*}}, %c128_i64], [%c512_i64, %c1_i64], [%[[WTOK]], %{{.*}}] {order = array<i32: 1, 0>} : <tensor<32x128xbf16>>
// TILEW: tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
// TILEW: arith.select %{{.*}}, %{{.*}}, %{{.*}} : tensor<16x32xi1>, tensor<16x32xf32>
// TILEW: "tt.reduce"(
