/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

// SelectedBlockScanRule (Rule A, #2686, rule bit 65536).
//
// A selected-block gather loop reads one block index per iteration from an
// index table and, guarded by a causal predicate, accumulates the block into
// a carried state. This rule rewrites that loop into a linear scan over
// BN-wide token tiles: the whole index table is read once as a vector before
// the loop, the tile's K/V data is always loaded, and "was this block
// selected" becomes a per-token weight in the score domain. The scan bound
// becomes cdiv(i_t + 1, BN), which depends only on the query position and
// never on a loaded index.
//
// Two carried-state forms are recognized: the online-softmax (o, m, l) triple
// of the forward kernel, and the single accumulator of the backward-dq kernel
// (acc = dot(ds, k^T, acc)), which needs no rescale and no safe_m.
//
// The rewrite runs in a private sandbox and is committed only when the joint
// Rule B gate accepts the rewritten function; otherwise the original IR is
// left untouched. Declining is expressed by finding no candidate, so apply()
// is an unfailable commit (the shared driver turns an apply() failure into
// signalPassFailure).

#include "TritonToGraph/GraphOptimizationRule.h"
#include "TritonToGraph/ProgramGridTransform.h"
#include "TritonToGraph/QueryRowBatching.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>

#define DEBUG_TYPE "graph-optimize"

using namespace mlir;
using namespace triton;
using namespace cfg;

// Joint Rule A + Rule B decision point: Rule A does not consult a read-only
// matcher any more.  It hands its private sandbox to
// `cfg::materializeQueryRowBatchingCandidate`, which rewrites Rule B's row
// lifting in place and verifies the result.  Landing the sandbox therefore
// means both rewrites are already in it; a failure means neither lands.
//
// This replaced the old `cfg::canQueryRowBatch` pure matcher: querying without
// rewriting left a window where Rule A could commit and Rule B then fail, and
// it could pick a different row factor than the rewrite would use.

namespace {

// ---- Constant values ----

static std::optional<int64_t> getConstantIntValue(Value v) {
  APInt value;
  if (matchPattern(v, m_ConstantInt(&value)))
    return value.getSExtValue();
  return std::nullopt;
}

static bool isPowerOfTwo(int64_t v) { return v > 0 && (v & (v - 1)) == 0; }

static bool isZeroSplatConstant(Value v) {
  if (auto def = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto dense = dyn_cast<DenseFPElementsAttr>(def.getValue()))
      return dense.isSplat() && dense.getSplatValue<APFloat>().isZero();
    if (auto denseInt = dyn_cast<DenseIntElementsAttr>(def.getValue()))
      return denseInt.isSplat() &&
             denseInt.getSplatValue<APInt>().isZero();
  }
  return false;
}

// Scalar value of a splat integer tensor constant. The bwd form spells its
// token stride and free-axis bound as tensors rather than as scalars.
static std::optional<int64_t> getSplatIntTensorValue(Value v) {
  if (auto def = v.getDefiningOp<arith::ConstantOp>())
    if (auto dense = dyn_cast<DenseIntElementsAttr>(def.getValue()))
      if (dense.isSplat())
        return dense.getSplatValue<APInt>().getSExtValue();
  return std::nullopt;
}

// The online-softmax form masks unselected positions with -inf, which is what
// makes exp() vanish and the scan's arithmetic exact.
static bool isNegativeInfSplatConstant(Value v) {
  if (auto def = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto dense = dyn_cast<DenseFPElementsAttr>(def.getValue()))
      return dense.isSplat() && dense.getSplatValue<APFloat>().isInfinity() &&
             dense.getSplatValue<APFloat>().isNegative();
    if (auto scalar = dyn_cast<FloatAttr>(def.getValue()))
      return scalar.getValue().isInfinity() &&
             scalar.getValue().isNegative();
  }
  if (auto splat = v.getDefiningOp<triton::SplatOp>()) {
    if (auto def = splat.getSrc().getDefiningOp<arith::ConstantOp>())
      if (auto scalar = dyn_cast<FloatAttr>(def.getValue()))
        return scalar.getValue().isInfinity() &&
               scalar.getValue().isNegative();
  }
  return false;
}

// ---- Dependence walks ----

// The walks are memoized and depth-bounded: without a visited set a shared
// subexpression tree makes them exponential, and a long linear chain
// overflows the stack. Every walk answers a monotone question, so remembering
// a visited value is sound. The depth cap returns the conservative answer,
// i.e. declines the match rather than accepting it.
constexpr unsigned kMaxDependenceDepth = 4096;

// True when v is transitively computed from pivot; block arguments are opaque
// roots.
static bool dependsOnImpl(Value v, Value pivot, llvm::DenseSet<Value> &visited,
                          unsigned depth) {
  if (v == pivot)
    return true;
  if (depth > kMaxDependenceDepth)
    return true; // too deep: assume dependence
  if (!visited.insert(v).second)
    return false;
  Operation *op = v.getDefiningOp();
  return op && llvm::any_of(op->getOperands(), [&](Value a) {
    return dependsOnImpl(a, pivot, visited, depth + 1);
  });
}

static bool dependsOn(Value v, Value pivot) {
  llvm::DenseSet<Value> visited;
  return dependsOnImpl(v, pivot, visited, 0);
}

// Dependence with one operation cut out of the walk. Rule A uses this to
// exclude the per-iteration index load, the mechanism the rewrite removes.
static bool dependsOnExcludingImpl(Value v, Value pivot, Operation *cut,
                                   llvm::DenseSet<Value> &visited,
                                   unsigned depth) {
  if (v == pivot)
    return true;
  if (depth > kMaxDependenceDepth)
    return true; // too deep: assume dependence
  Operation *op = v.getDefiningOp();
  if (!op || op == cut)
    return false;
  if (!visited.insert(v).second)
    return false;
  return llvm::any_of(op->getOperands(), [&](Value a) {
    return dependsOnExcludingImpl(a, pivot, cut, visited, depth + 1);
  });
}

static bool dependsOnExcluding(Value v, Value pivot, Operation *cut) {
  llvm::DenseSet<Value> visited;
  return dependsOnExcludingImpl(v, pivot, cut, visited, 0);
}

// A splat/expand_dims/broadcast chain down to the pivot.
static bool isBroadcastOf(Value v, Value pivot) {
  if (v == pivot)
    return true;
  Operation *op = v.getDefiningOp();
  if (!op)
    return false;
  if (isa<triton::SplatOp, triton::ExpandDimsOp, triton::BroadcastOp>(op))
    return isBroadcastOf(op->getOperand(0), pivot);
  return false;
}

// True when the value's defining chain crosses a load; forbidden in the K/V
// address chains. The varlen dispatch computes the K/V base and bound from
// loads.
static bool reachesLoadImpl(Value v, llvm::DenseSet<Value> &visited,
                            unsigned depth) {
  if (depth > kMaxDependenceDepth)
    return true; // too deep: assume it reaches a load
  Operation *op = v.getDefiningOp();
  if (!op)
    return false;
  if (isa<triton::LoadOp>(op))
    return true;
  if (!visited.insert(v).second)
    return false;
  return llvm::any_of(op->getOperands(), [&](Value a) {
    return reachesLoadImpl(a, visited, depth + 1);
  });
}

static bool reachesLoad(Value v) {
  llvm::DenseSet<Value> visited;
  return reachesLoadImpl(v, visited, 0);
}

static bool isDefinedOutsideLoop(Value v, scf::ForOp loop) {
  Operation *op = v.getDefiningOp();
  return !op || !loop->isAncestor(op);
}

// ---- Spelling and shape recognition ----

// a*iv + b, by value, for any integer width of iv. Recognizing the affine
// form makes a non-identity table walk diagnosable instead of silently
// folded: the vectorized table read evaluates table[s] for the iteration
// number s, which is only equivalent to the baseline gather when the offset
// is the induction variable.
struct IvAffine {
  int64_t scale = 0;
  int64_t bias = 0;
};
static std::optional<IvAffine> asAffineInIv(Value v, Value iv) {
  if (v == iv)
    return IvAffine{1, 0};
  if (auto constant = getConstantIntValue(v))
    return IvAffine{0, *constant};
  Operation *op = v.getDefiningOp();
  if (!op)
    return std::nullopt;
  if (auto add = dyn_cast<arith::AddIOp>(op)) {
    auto lhs = asAffineInIv(add.getLhs(), iv);
    auto rhs = asAffineInIv(add.getRhs(), iv);
    if (!lhs || !rhs)
      return std::nullopt;
    return IvAffine{lhs->scale + rhs->scale, lhs->bias + rhs->bias};
  }
  if (auto sub = dyn_cast<arith::SubIOp>(op)) {
    auto lhs = asAffineInIv(sub.getLhs(), iv);
    auto rhs = asAffineInIv(sub.getRhs(), iv);
    if (!lhs || !rhs)
      return std::nullopt;
    return IvAffine{lhs->scale - rhs->scale, lhs->bias - rhs->bias};
  }
  if (auto mul = dyn_cast<arith::MulIOp>(op)) {
    auto lhs = asAffineInIv(mul.getLhs(), iv);
    auto rhs = asAffineInIv(mul.getRhs(), iv);
    if (!lhs || !rhs)
      return std::nullopt;
    if (lhs->scale == 0)
      return IvAffine{rhs->scale * lhs->bias, rhs->bias * lhs->bias};
    if (rhs->scale == 0)
      return IvAffine{lhs->scale * rhs->bias, lhs->bias * rhs->bias};
    return std::nullopt;
  }
  return std::nullopt;
}

// Strip the integer width casts the front end may wrap around the loaded index.
static Value stripIntegerCasts(Value v) {
  while (Operation *op = v.getDefiningOp()) {
    if (!isa<arith::TruncIOp, arith::ExtSIOp, arith::ExtUIOp>(op))
      break;
    v = op->getOperand(0);
  }
  return v;
}

// Recognize idx * BS by value: muli(x, BS) or shli(x, log2(BS)), with x a
// cast chain of the raw index load. Matching by value keeps both spellings
// and non-CSE'd duplicates working.
static std::optional<int64_t> matchIdxTimesBS(Value offset, Value rawIndex) {
  Operation *op = offset.getDefiningOp();
  if (!op)
    return std::nullopt;
  Value other;
  int64_t factor = 0;
  if (auto mul = dyn_cast<arith::MulIOp>(op)) {
    if (auto c = getConstantIntValue(mul.getRhs())) {
      other = mul.getLhs();
      factor = *c;
    } else if (auto c = getConstantIntValue(mul.getLhs())) {
      other = mul.getRhs();
      factor = *c;
    } else {
      return std::nullopt;
    }
  } else if (auto shift = dyn_cast<arith::ShLIOp>(op)) {
    auto c = getConstantIntValue(shift.getRhs());
    if (!c || *c < 0 || *c > 62)
      return std::nullopt;
    other = shift.getLhs();
    factor = int64_t{1} << *c;
  } else {
    return std::nullopt;
  }
  if (stripIntegerCasts(other) != rawIndex)
    return std::nullopt;
  return factor;
}

// The causal query position. It is the axis-0 program id itself, or the
// program id advanced by a loop-invariant offset: the decode path compares
// block tokens against `pid + Q_OFFSET`, where Q_OFFSET = TK - TQ is a kernel
// argument. Returns the SSA value every causal use must share, or a null value
// when the operand is load-derived (the varlen dispatch) or computed inside the
// loop (an iteration-dependent row position cannot be a single scan bound).
static Value matchRowPosition(Value v, Value pid, scf::ForOp loop) {
  Value stripped = stripIntegerCasts(v);
  if (stripped == pid)
    return pid;
  auto add = stripped.getDefiningOp<arith::AddIOp>();
  if (!add)
    return Value();
  Value lhs = stripIntegerCasts(add.getLhs());
  Value rhs = stripIntegerCasts(add.getRhs());
  Value offset;
  if (lhs == pid)
    offset = rhs;
  else if (rhs == pid)
    offset = lhs;
  else
    return Value();
  if (!isDefinedOutsideLoop(offset, loop) || reachesLoad(offset))
    return Value();
  return stripped;
}

// The [G] vector a 2-D operand broadcasts from, or v itself.
static Value peelBroadcasts(Value v) {
  while (Operation *op = v.getDefiningOp())
    if (isa<triton::SplatOp, triton::ExpandDimsOp, triton::BroadcastOp>(op))
      v = op->getOperand(0);
    else
      break;
  return v;
}

// ---- Op predicates ----

// The single combiner op of a canonical two-argument reduce region, or
// nullptr for anything else.
static Operation *getReduceCombiner(triton::ReduceOp reduce) {
  if (reduce.getNumOperands() != 1 || reduce->getNumResults() != 1)
    return nullptr;
  if (reduce.getCombineOp().empty())
    return nullptr;
  Operation *combiner = nullptr;
  for (Operation &op : reduce.getCombineOp().front().without_terminator()) {
    if (op.getNumResults() == 0)
      continue;
    if (combiner)
      return nullptr;
    combiner = &op;
  }
  return combiner;
}

// ---- Match result ----

// The additive (bwd dq) form. The bwd kernel carries ONE accumulator and
// accumulates into it through the dot's C operand (acc' = dot(ds, k^T, acc));
// the forward form instead rescales the carried state by exp(m_old - m_new)
// and adds a separate dot.

// The additive form's tile address. The bwd kernel never uses
// tt.make_tensor_ptr: it addresses K/V with two nested tt.addptr's
//
//   addptr(broadcast(addptr(splat(base), expand_dims(free, 1))),
//          broadcast(expand_dims(tokens, 0) * stride))
//
// and loads with an explicit mask (free < F) & (tokens < T) plus other = 0.
// The token-range term is the only guard against reading past the row's token
// count, so the bound is captured here and reproduced verbatim by the builder.
struct AddPtrTile {
  Value base;               // scalar pointer, loop-invariant and load-free
  Value freeVec;            // [F] integer vector, loop-invariant
  Value freeBoundRhs;       // [F,1] tensor the free-axis mask compares against
  Value tokenBound;         // scalar bound of the "tokens < T" mask
  Value blockStart;         // idx*BS: the value the causal mask also uses
  Type tokenType;           // width of the token arithmetic (i32 or i64)
  int64_t tokenStride = 0;  // stride of one token along the address
  int64_t freeExtent = 0;   // F, the tile's free-axis extent
  int64_t blockSize = 0;    // BS, the tile's token extent
};

struct AdditiveState {
  triton::DotOp scoreDot;     // dot(q, k, 0)
  triton::DotOp valueDot;     // dot(dO, v, 0)
  triton::DotOp accDot;       // dot(ds, k^T, acc)
  arith::SelectOp maskSelect; // select(causal, p, 0): the neutral path
  triton::TransOp transpose;  // k^T, its order replayed verbatim by the builder
  Value lseVec;               // [G] f32, loop-invariant
  Value deltaVec;             // [G] f32, loop-invariant
};

// ---- The scan candidate ----

struct ScanCandidate {
  triton::FuncOp function;
  scf::ForOp loop;

  // The single query position fed to the table row base, the causal predicate
  // and the tensor mask.
  Value pid;

  // The causal row position: `pid` itself, or `pid + offset` when the kernel
  // compares against a decode offset (Q_OFFSET = TK - TQ in the production
  // fwd kernel).  Every causal use of the row position reads this value.
  Value rowPosition;

  // The index-table row base; loop-invariant, i64 pointee in production.
  Value tableBase;
  int64_t tableLength = 0;  // S, from the i_h*S multiplier of the row base
  int64_t blockSize = 0;    // BS, the idx multiplier
  int64_t scanTileWidth = 0; // BN = max(64, BS)
  int64_t blocksPerTile = 0; // R = BN / BS
  Value scanBound;           // the old trip count == NS (constant or opaque)

  // The K/V token offsets are idx*BS; these are the two tile pointers.
  triton::MakeTensorPtrOp kPtr;
  triton::MakeTensorPtrOp vPtr;
  triton::LoadOp kLoad;
  triton::LoadOp vLoad;
  unsigned kTokenAxis = 0;
  unsigned vTokenAxis = 0;

  // The only loop-carried address computation, re-created outside the loop.
  Value dynamicBoundResult; // null when the bound is already outside
  Value dynamicBoundInput;  // null together with dynamicBoundResult

  // The two dots of the guarded region.
  triton::DotOp scoreDot;
  triton::DotOp valueDot;
  arith::SelectOp maskSelect; // the -inf select being replaced

  // Carried-state roles (o, m, l) in iter-argument order.
  unsigned oArg = 0;
  unsigned mArg = 0;
  unsigned lArg = 0;

  // Which rewrite template this candidate gets. Everything outside the
  // template (prologue, weights, tiles, scan bound) is shared.
  bool additive = false;
  unsigned accArg = 0;       // additive: the carried accumulator
  triton::DotOp accDot;      // additive: dot(ds, k^T, acc)
  triton::TransOp transpose; // additive: k^T, order replayed verbatim
  AddPtrTile kTile;          // additive: the two addptr tile addresses
  AddPtrTile vTile;
  Value lseVec;              // additive: [G] f32, loop-invariant
  Value deltaVec;            // additive: [G] f32, loop-invariant
};

static bool sameScanCandidate(const ScanCandidate &a, const ScanCandidate &b) {
  return a.additive == b.additive && a.tableLength == b.tableLength &&
         a.blockSize == b.blockSize &&
         a.scanTileWidth == b.scanTileWidth &&
         a.blocksPerTile == b.blocksPerTile;
}

// ---- The matcher ----
//
// Ten ordered stages, each narrowing the candidate; a false declines. The
// anchors are semantic, not line-shaped: op order inside a region is never
// inspected, comparison direction and operand order are normalized, constants
// match by value, and the index multiplier is matched by value so muli/shli
// both work. Predicate algebra variants such as idx <= i_t/BS instead of
// idx*BS <= i_t are rejected: they need an equivalence prover, not a matcher.

// The matcher's working state; the stage functions run in field order, so
// each field is written before it is first read.
struct ScanMatchContext {
  triton::FuncOp fn;

  // matchLoopSkeleton.
  scf::ForOp loop;
  unsigned numCarried = 0;
  bool additive = false;

  // matchIndexTable.
  triton::LoadOp indexLoad;
  Value tableBase;
  int64_t tableLength = 0;
  Value pid;
  // matchBlockGuard: pid, or pid advanced by a loop-invariant offset.
  Value rowPosition;

  // matchBlockGuard.
  Value rawIndex;
  scf::IfOp branch;
  SmallVector<int64_t, 2> predicateFactors;

  // matchCarriedState.
  // Set by whichever classifier matches; the shared anchors below read them.

  triton::DotOp scoreDot;
  triton::DotOp valueDot;
  arith::SelectOp maskSelect;
  unsigned oIdx = 0;
  unsigned mIdx = 0;
  unsigned lIdx = 0;
  AdditiveState additiveState;

  // matchTiles.
  triton::LoadOp kLoad;
  triton::LoadOp vLoad;
  int64_t blockSize = 0;
  Value kTokenOffset;
  Value vTokenOffset;
  unsigned kTokenAxis = 0;
  unsigned vTokenAxis = 0;
  triton::MakeTensorPtrOp kPtr;
  triton::MakeTensorPtrOp vPtr;
  AddPtrTile kTile;
  AddPtrTile vTile;
  Value dynamicBoundResult;
  Value dynamicBoundInput;

  // matchTileContract.
  int64_t scanTileWidth = 0;
  int64_t blocksPerTile = 0;
};

// The form-specific matchers, called from the shared stages above.
static bool matchOnlineSoftmaxState(scf::ForOp loop, scf::IfOp branch,
                                    scf::YieldOp thenYield, unsigned &oIdx,
                                    unsigned &mIdx, unsigned &lIdx,
                                    triton::DotOp &scoreDot,
                                    triton::DotOp &valueDot,
                                    arith::SelectOp &maskSelect);
static bool matchAdditiveState(scf::ForOp loop, Region &thenRegion,
                               BlockArgument accArg, Value yielded,
                               AdditiveState &state);
static bool matchAddPtrTile(triton::LoadOp load, Value rawIndex,
                            scf::ForOp loop, AddPtrTile &tile);

// The loop skeleton and the outside-load budget.
static bool matchLoopSkeleton(ScanMatchContext &ctx) {
  triton::FuncOp fn = ctx.fn;
  scf::ForOp &loop = ctx.loop;
  // Exactly one scf.for, at function-body top level, with all-tensor iter
  // args, lb 0 and step 1. Rule B requires the same shape, and the rewrite
  // assumes no other loop nest: scf.while and scf.parallel are rejected.
  unsigned forOps = 0;
  fn.walk([&](scf::ForOp forOp) {
    ++forOps;
    loop = forOp;
  });
  if (forOps != 1 || !loop || loop->getBlock() != &fn.getBody().front())
    return false;
  bool foreignControlFlow = false;
  fn.walk([&](Operation *op) {
    if (isa<scf::WhileOp, scf::ParallelOp>(op))
      foreignControlFlow = true;
  });
  // The carried state is either the online-softmax triple (fwd: o/m/l) or the
  // single additive accumulator (bwd dq); the classifier below decides which.
  const unsigned numCarried = loop.getNumResults();
  if (foreignControlFlow || (numCarried != 3 && numCarried != 1))
    return false;
  const bool additive = numCarried == 1;
  for (Value init : loop.getInitArgs())
    if (!isa<RankedTensorType>(init.getType()))
      return false;
  if (getConstantIntValue(loop.getLowerBound()) != std::optional<int64_t>(0) ||
      getConstantIntValue(loop.getStep()) != std::optional<int64_t>(1))
    return false;

  // At most one scalar load outside the loop. The dense production path has
  // zero; the dyn-NS path has exactly one (the block_counts read, which must
  // survive the rewrite); a varlen dispatch has five and is declined.
  unsigned outsideScalarLoads = 0;
  fn.walk([&](triton::LoadOp load) {
    if (!load->getParentOfType<scf::ForOp>() &&
        !isa<RankedTensorType>(load.getType()))
      ++outsideScalarLoads;
  });
  if (outsideScalarLoads > 1)
    return false;
  ctx.numCarried = numCarried;
  ctx.additive = additive;
  return true;
}

// The index-table read, the row base and the table length S.
static bool matchIndexTable(ScanMatchContext &ctx) {
  triton::FuncOp fn = ctx.fn;
  scf::ForOp loop = ctx.loop;
  triton::LoadOp &indexLoad = ctx.indexLoad;
  Value &tableBase = ctx.tableBase;
  Value &pid = ctx.pid;
  int64_t &tableLength = ctx.tableLength;
  // Exactly one scalar load inside the loop, plus exactly two tensor loads
  // (the K/V tiles). The scalar address is the loop-invariant table base
  // advanced by the induction variable; this load is the core of the
  // per-block gather and the one thing the rewrite removes.
  unsigned inLoopLoads = 0;
  unsigned inLoopScalarLoads = 0;
  loop->walk([&](triton::LoadOp load) {
    ++inLoopLoads;
    if (!isa<RankedTensorType>(load.getType())) {
      ++inLoopScalarLoads;
      indexLoad = load;
    }
  });
  if (inLoopLoads != 3 || inLoopScalarLoads != 1)
    return false;
  auto indexAddPtr = indexLoad.getPtr().getDefiningOp<triton::AddPtrOp>();
  if (!indexAddPtr)
    return false;
  tableBase = indexAddPtr.getPtr();
  // The table entries must be wide enough for the i32 token arithmetic below;
  // production index tables are i64.
  auto tablePtrType = dyn_cast<triton::PointerType>(tableBase.getType());
  if (!tablePtrType)
    return false;
  auto tableElem = dyn_cast<IntegerType>(tablePtrType.getPointeeType());
  if (!tableElem || tableElem.getWidth() < 32)
    return false;
  // The rewrite merges the per-iteration loads into one vector read, so a
  // masked, "other"-padded or volatile entry load is not this pattern.
  if (indexLoad.getMask() || indexLoad.getOther() ||
      indexLoad.getIsVolatile())
    return false;
  if (!isDefinedOutsideLoop(tableBase, loop))
    return false;
  // The rewrite reads table[0, S) once and indexes it by the iteration
  // number, so the baseline offset must be exactly the induction variable:
  // an offset a*iv+b would read table[b], table[b+a], ... instead.
  auto tableOffset = asAffineInIv(indexAddPtr.getOffset(),
                                  loop.getInductionVar());
  if (!tableOffset || tableOffset->scale != 1 || tableOffset->bias != 0)
    return false;

  // The row base is addptr(root, P + i_h*S), where P carries the query
  // position and S is the table length. S cannot be read off a bare constant
  // (the same 16 is also G in these kernels), so it comes from the i_h*S
  // multiplier and is cross-checked against a constant trip count below.
  //
  // The function need not spell axis-0 tt.get_program_id exactly once: the
  // rewrite only needs the SSA value the row base is built from. The same
  // value is required by the causal predicate and the tensor mask, so an
  // inconsistent second pid cannot survive.
  SmallVector<Value, 4> axis0Pids;
  fn.walk([&](triton::GetProgramIdOp op) {
    if (op.getAxisAsInt() == 0)
      axis0Pids.push_back(op.getResult());
  });
  if (axis0Pids.empty())
    return false;
  // The single axis-0 program id v is built from, or nullopt when it is built
  // from none or from more than one (ambiguous: refuse to pick).
  auto uniquePidReaching = [&](Value v) -> std::optional<Value> {
    Value found;
    for (Value pid : axis0Pids) {
      if (!dependsOn(v, pid))
        continue;
      if (found)
        return std::nullopt;
      found = pid;
    }
    if (!found)
      return std::nullopt; // none reaches: not a query-position dependency
    return found;
  };
  auto rowAddPtr = tableBase.getDefiningOp<triton::AddPtrOp>();
  if (!rowAddPtr)
    return false;
  Value rowOffset = rowAddPtr.getOffset();
  Value slotTerm;
  if (auto offsetAdd = rowOffset.getDefiningOp<arith::AddIOp>()) {
    std::optional<Value> lhsPid = uniquePidReaching(offsetAdd.getLhs());
    std::optional<Value> rhsPid = uniquePidReaching(offsetAdd.getRhs());
    bool lhsPidCarries = lhsPid.has_value();
    bool rhsPidCarries = rhsPid.has_value();
    // Exactly one addend carries the query position; the other is the i_h*S
    // term of the table layout. Ambiguous on either side is refused as well.
    if (lhsPidCarries == rhsPidCarries)
      return false;
    pid = lhsPidCarries ? *lhsPid : *rhsPid;
    slotTerm = lhsPidCarries ? offsetAdd.getRhs() : offsetAdd.getLhs();
  } else if (std::optional<Value> rowPid = uniquePidReaching(rowOffset)) {
    pid = *rowPid;
    slotTerm = Value();
  } else {
    return false;
  }
  if (slotTerm) {
    // The front end may widen the i_h*S term (extsi(muli(i_h, S))), so peel
    // the integer width casts before reading the multiplier.
    auto mul = dyn_cast<arith::MulIOp>(stripIntegerCasts(slotTerm).getDefiningOp());
    if (!mul)
      return false;
    std::optional<int64_t> multiplier;
    if ((multiplier = getConstantIntValue(mul.getRhs())))
      tableLength = *multiplier;
    else if ((multiplier = getConstantIntValue(mul.getLhs())))
      tableLength = *multiplier;
    else
      return false;
  } else {
    // No i_h*S term (H == 1 collapses the layout): the table length is only
    // recoverable from a constant trip count.
    auto trip = getConstantIntValue(loop.getUpperBound());
    if (!trip)
      return false;
    tableLength = *trip;
  }
  // S is the STATIC row length of the index table: the host passes
  // block_indices as [i_h, S], so the i_h*S multiplier of the row base IS S.
  // NS is the RUNTIME value of block_counts, i.e. the old trip count, and it
  // is not necessarily S. The rewrite relies on the kernel contract
  //
  //     0 <= NS <= S
  //
  // which the top-k construction already satisfies. It is needed twice:
  // slot_valid = [0, S) < NS gates the stale [NS, S) tail of the fixed-width
  // vector read, and the builder clamps NS into [0, S] in its own width
  // before truncation. A constant trip count is the special case NS == S; it
  // is cross-checked below so the two paths cannot disagree.
  if (tableLength <= 0)
    return false;
  // S drives a vector read of S table entries, so an unbounded S would
  // materialize an arbitrarily wide tensor read. 4096 is a generous ceiling
  // that still bounds the cost.
  if (tableLength > 4096)
    return false;
  if (auto trip = getConstantIntValue(loop.getUpperBound());
      trip && *trip != tableLength)
    return false;
  return true;
}

// The scaled-index guard of the single scf.if.
static bool matchBlockGuard(ScanMatchContext &ctx) {
  scf::ForOp loop = ctx.loop;
  triton::LoadOp indexLoad = ctx.indexLoad;
  scf::IfOp &branch = ctx.branch;
  Value &rawIndex = ctx.rawIndex;
  SmallVector<int64_t, 2> &predicateFactors = ctx.predicateFactors;
  // The guard is andi(causal, range) over the SCALED index: i_s <= row and
  // 0 <= i_s. Direction flips and operand order are free; the causal operand
  // must be the axis-0 program id itself, or the program id advanced by a
  // loop-invariant offset (the decode Q_OFFSET). A load-derived value there is
  // the varlen dispatch. Comparisons on the unscaled index do not match.
  unsigned guards = 0;
  for (Operation &op : loop.getBody()->without_terminator())
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      branch = ifOp;
      ++guards;
    }
  if (guards != 1)
    return false;
  auto predicateAnd =
      branch.getCondition().getDefiningOp<arith::AndIOp>();
  if (!predicateAnd)
    return false;
  rawIndex = indexLoad.getResult();
  auto isScaledIndex = [&](Value v) {
    auto factor = matchIdxTimesBS(v, rawIndex);
    if (!factor || *factor <= 0)
      return false;
    predicateFactors.push_back(*factor);
    return true;
  };
  bool causalFound = false;
  bool rangeFound = false;
  for (Value operand : {predicateAnd.getLhs(), predicateAnd.getRhs()}) {
    auto cmp = operand.getDefiningOp<arith::CmpIOp>();
    if (!cmp || isa<RankedTensorType>(cmp.getLhs().getType()))
      return false;
    arith::CmpIPredicate pred = cmp.getPredicate();
    Value lhs = cmp.getLhs();
    Value rhs = cmp.getRhs();
    if (pred == arith::CmpIPredicate::sle && isScaledIndex(lhs)) {
      if (Value position = matchRowPosition(rhs, ctx.pid, loop)) {
        ctx.rowPosition = position;
        causalFound = true;
        continue;
      }
    } else if (pred == arith::CmpIPredicate::sge && isScaledIndex(rhs)) {
      if (Value position = matchRowPosition(lhs, ctx.pid, loop)) {
        ctx.rowPosition = position;
        causalFound = true;
        continue;
      }
    }
    auto isZero = [](Value v) {
      auto constant = getConstantIntValue(v);
      return constant && *constant == 0;
    };
    if ((pred == arith::CmpIPredicate::sge && isScaledIndex(lhs) &&
         isZero(rhs)) ||
        (pred == arith::CmpIPredicate::sle && isZero(lhs) &&
         isScaledIndex(rhs))) {
      rangeFound = true;
      continue;
    }
    return false;
  }
  if (!causalFound || !rangeFound)
    return false;
  return true;
}

static bool matchIfShape(ScanMatchContext &ctx) {
  scf::ForOp loop = ctx.loop;
  scf::IfOp branch = ctx.branch;
  const unsigned numCarried = ctx.numCarried;
  // The else region yields the iter args unchanged, i.e. skipping a block is
  // neutral for the carried state; that is what lets the rewrite replace the
  // branch with a zero weight.
  if (branch.getElseRegion().empty() ||
      !llvm::hasSingleElement(branch.getElseRegion()))
    return false;
  auto elseYield =
      dyn_cast<scf::YieldOp>(branch.getElseRegion().front().getTerminator());
  if (!elseYield || elseYield->getNumOperands() != numCarried)
    return false;
  for (unsigned i = 0; i < numCarried; ++i)
    if (elseYield->getOperand(i) != loop.getRegionIterArgs()[i])
      return false;
  auto latchYield = dyn_cast<scf::YieldOp>(loop.getBody()->getTerminator());
  if (!latchYield || latchYield->getNumOperands() != numCarried)
    return false;
  for (unsigned i = 0; i < numCarried; ++i)
    if (latchYield->getOperand(i) != branch->getResult(i))
      return false;
  return true;
}

// The carried-state classifier: fwd (o, m, l) vs bwd (one accumulator).
static bool matchCarriedState(ScanMatchContext &ctx) {
  scf::ForOp loop = ctx.loop;
  scf::IfOp branch = ctx.branch;
  const unsigned numCarried = ctx.numCarried;
  const bool additive = ctx.additive;
  unsigned &oIdx = ctx.oIdx;
  unsigned &mIdx = ctx.mIdx;
  unsigned &lIdx = ctx.lIdx;
  triton::DotOp &scoreDot = ctx.scoreDot;
  triton::DotOp &valueDot = ctx.valueDot;
  arith::SelectOp &maskSelect = ctx.maskSelect;
  AdditiveState &additiveState = ctx.additiveState;
  // The scan template is form-specific, so the shape of the recurrence
  // decides which template the builder emits:
  //   * online softmax (fwd, 3 iter args): (o, m, l), m updated with maxnumf,
  //     rescale by exp(m_old - m_new);
  //   * pure additive (bwd dq, 1 iter arg): one accumulator with neutral 0,
  //     lse/delta read as loop-invariant inputs, the contribution accumulated
  //     through the dot's C operand, no safe_m.
  // A recurrence in neither form is refused rather than half-matched: the two
  // templates are not interchangeable.
  if (branch.getThenRegion().empty() ||
      !llvm::hasSingleElement(branch.getThenRegion()))
    return false;
  Block &thenBlock = branch.getThenRegion().front();
  auto thenYield = dyn_cast<scf::YieldOp>(thenBlock.getTerminator());
  if (!thenYield || thenYield->getNumOperands() != numCarried)
    return false;
  if (!additive) {
    if (!matchOnlineSoftmaxState(loop, branch, thenYield, oIdx,
                                 mIdx, lIdx, scoreDot, valueDot,
                                 maskSelect))
      return false;
  } else {
    // The recurrence is read backwards from the dot that accumulates into the
    // carried state.
    if (!matchAdditiveState(loop, branch.getThenRegion(),
                            loop.getRegionIterArgs()[0],
                            thenYield->getOperand(0), additiveState))
      return false;
    scoreDot = additiveState.scoreDot;
    valueDot = additiveState.valueDot;
    maskSelect = additiveState.maskSelect;
  }
  return true;
}

// Dot operand dependence and loop invariance.
static bool matchDotOperands(ScanMatchContext &ctx) {
  scf::ForOp loop = ctx.loop;
  triton::LoadOp indexLoad = ctx.indexLoad;
  Value pid = ctx.pid;
  triton::DotOp scoreDot = ctx.scoreDot;
  triton::DotOp valueDot = ctx.valueDot;
  // Every dot lhs is query-row dependent (axis 0), every rhs is not.
  // Dependence through the per-iteration index load does not count: it is
  // exactly what the rewrite removes. Only axis-x dependence is considered.
  if (!dependsOnExcluding(scoreDot.getA(), pid, indexLoad))
    return false;
  if (dependsOnExcluding(scoreDot.getB(), pid, indexLoad))
    return false;
  if (!dependsOnExcluding(valueDot.getA(), pid, indexLoad))
    return false;
  if (dependsOnExcluding(valueDot.getB(), pid, indexLoad))
    return false;
  // The score dot's lhs is Q, loaded and scaled outside the loop, so the
  // rewrite can reuse the operand verbatim.
  if (!isDefinedOutsideLoop(scoreDot.getA(), loop))
    return false;
  return true;
}

// The K/V tiles, token offsets and address chains.
static bool matchTiles(ScanMatchContext &ctx) {
  scf::ForOp loop = ctx.loop;
  scf::IfOp branch = ctx.branch;
  const bool additive = ctx.additive;
  Value rawIndex = ctx.rawIndex;
  SmallVector<int64_t, 2> &predicateFactors = ctx.predicateFactors;
  triton::DotOp scoreDot = ctx.scoreDot;
  triton::DotOp valueDot = ctx.valueDot;
  triton::LoadOp &kLoad = ctx.kLoad;
  triton::LoadOp &vLoad = ctx.vLoad;
  int64_t &blockSize = ctx.blockSize;
  Value &kTokenOffset = ctx.kTokenOffset;
  Value &vTokenOffset = ctx.vTokenOffset;
  unsigned &kTokenAxis = ctx.kTokenAxis;
  unsigned &vTokenAxis = ctx.vTokenAxis;
  triton::MakeTensorPtrOp &kPtr = ctx.kPtr;
  triton::MakeTensorPtrOp &vPtr = ctx.vPtr;
  AddPtrTile &kTile = ctx.kTile;
  AddPtrTile &vTile = ctx.vTile;
  Value &dynamicBoundResult = ctx.dynamicBoundResult;
  Value &dynamicBoundInput = ctx.dynamicBoundInput;
  // Both forms read one K tile and one V tile per iteration, addressed from
  // the block index; they differ only in the spelling: the forward kernel
  // uses tt.make_tensor_ptr (the token axis is one of the view's offsets),
  // the bwd kernel two nested tt.addptr's plus an explicit mask.
  kLoad = scoreDot.getB().getDefiningOp<triton::LoadOp>();
  vLoad = valueDot.getB().getDefiningOp<triton::LoadOp>();
  if (!kLoad || !vLoad || kLoad == vLoad)
    return false;
  if (kLoad->getParentRegion() != &branch.getThenRegion() ||
      vLoad->getParentRegion() != &branch.getThenRegion())
    return false;
  // The additive form comes in two spellings: the original two-level addptr
  // chain with an explicit mask (the hand-written dq variant the rule was first
  // built on), and production nsa_bwd_dq's tt.make_tensor_ptr plus
  // boundaryCheck. The addptr spelling is tried first; when it does not hold,
  // the shared mptr path below runs and the additive geometry check after it
  // covers both spellings.
  bool useAddPtrTiles = false;
  int64_t kFreeExtent = 0;
  int64_t vFreeExtent = 0;
  if (additive) {
    AddPtrTile kAddPtrTile;
    AddPtrTile vAddPtrTile;
    if (matchAddPtrTile(kLoad, rawIndex, loop, kAddPtrTile) &&
        matchAddPtrTile(vLoad, rawIndex, loop, vAddPtrTile)) {
      // Both tiles must cover the same token block and carry the same
      // token-range mask: the builder rebuilds one bound for both.
      if (kAddPtrTile.blockSize != vAddPtrTile.blockSize ||
          kAddPtrTile.tokenBound != vAddPtrTile.tokenBound)
        return false;
      kTile = kAddPtrTile;
      vTile = vAddPtrTile;
      blockSize = kTile.blockSize;
      kTokenOffset = kTile.blockStart;
      vTokenOffset = vTile.blockStart;
      kFreeExtent = kTile.freeExtent;
      vFreeExtent = vTile.freeExtent;
      useAddPtrTiles = true;
    }
  }
  if (!useAddPtrTiles) {
    // The rewrite loads each tile unconditionally, so a per-element mask, an
    // "other" padding value or a volatile qualifier on the tile loads is out
    // of scope.
    auto isPlainLoad = [](triton::LoadOp load) {
      return !load.getMask() && !load.getOther() && !load.getIsVolatile();
    };
    if (!isPlainLoad(kLoad) || !isPlainLoad(vLoad))
      return false;
    kPtr = kLoad.getPtr().getDefiningOp<triton::MakeTensorPtrOp>();
    vPtr = vLoad.getPtr().getDefiningOp<triton::MakeTensorPtrOp>();
    if (!kPtr || !vPtr)
      return false;
    if (!mlir::isa<triton::PointerType>(kPtr.getType()) ||
        !mlir::isa<triton::PointerType>(vPtr.getType()))
      return false;
    auto identifyTile = [&](triton::MakeTensorPtrOp ptr, unsigned &tokenAxis,
                            Value &tokenOffset) -> bool {
      unsigned found = 0;
      for (unsigned axis = 0; axis < ptr.getOffsets().size(); ++axis) {
        Value offset = ptr.getOffsets()[axis];
        if (!dependsOn(offset, rawIndex))
          continue;
        ++found;
        tokenAxis = axis;
        auto factor = matchIdxTimesBS(offset, rawIndex);
        if (!factor || *factor <= 0)
          return false;
        if (blockSize && *factor != blockSize)
          return false;
        blockSize = *factor;
        tokenOffset = offset;
      }
      return found == 1;
    };
    if (!identifyTile(kPtr, kTokenAxis, kTokenOffset) ||
        !identifyTile(vPtr, vTokenAxis, vTokenOffset))
      return false;
    // The tile extent on the token axis must equal BS. The rewrite pins the
    // tile token extent to BN, so a narrower or wider tile would change how
    // many tokens a selected block accumulates, and a dynamic extent cannot
    // be proven equal to BN at all.
    auto hasOneBlockOnTokenAxis = [&](triton::MakeTensorPtrOp ptr,
                                      unsigned tokenAxis) -> bool {
      auto ptrType = cast<triton::PointerType>(ptr.getType());
      auto blockType = dyn_cast<RankedTensorType>(ptrType.getPointeeType());
      if (!blockType)
        return false;
      ArrayRef<int64_t> shape = blockType.getShape();
      return tokenAxis < shape.size() && shape[tokenAxis] == blockSize;
    };
    if (!hasOneBlockOnTokenAxis(kPtr, kTokenAxis) ||
        !hasOneBlockOnTokenAxis(vPtr, vTokenAxis))
      return false;
    // mptr offsets are i32, so the token offsets are i32 by construction.
    if (!kTokenOffset || !kTokenOffset.getType().isInteger(32) ||
        !vTokenOffset.getType().isInteger(32))
      return false;
    // No load may appear anywhere in the K/V address chains (base, view
    // shape, strides, non-token offsets).
    auto checkAddressOperand = [&](Value v) -> bool {
      if (reachesLoad(v))
        return false;
      Operation *def = v.getDefiningOp();
      if (!def || !loop->isAncestor(def))
        return true; // defined outside: reused verbatim
      // The only loop-carried address computation is the sign extension of
      // the dynamic K/V bound; it is loop-invariant and re-created outside.
      auto ext = dyn_cast<arith::ExtSIOp>(def);
      if (!ext || !isDefinedOutsideLoop(ext.getIn(), loop))
        return false;
      if (dynamicBoundResult && dynamicBoundResult != ext.getResult())
        return false;
      dynamicBoundResult = ext.getResult();
      dynamicBoundInput = ext.getIn();
      return true;
    };
    for (triton::MakeTensorPtrOp ptr : {kPtr, vPtr}) {
      if (!checkAddressOperand(ptr.getBase()))
        return false;
      for (Value v : ptr.getShape())
        if (!checkAddressOperand(v))
          return false;
      for (Value v : ptr.getStrides())
        if (!checkAddressOperand(v))
          return false;
      unsigned tokenAxis = ptr == kPtr ? kTokenAxis : vTokenAxis;
      for (unsigned axis = 0; axis < ptr.getOffsets().size(); ++axis) {
        if (axis == tokenAxis)
          continue;
        if (!checkAddressOperand(ptr.getOffsets()[axis]))
          return false;
      }
    }
  }
  // The additive geometry is spelling independent: dot(ds[G, BS], k^T[BS, F])
  // -> acc[G, F] means both tile free axes are the accumulator's column count
  // and both dots produce [G, BS].
  if (additive) {
    if (!useAddPtrTiles) {
      // The mptr spelling carries the free extent as the block shape on the
      // axis that is not the token axis.
      auto freeExtentOf = [&](triton::MakeTensorPtrOp ptr,
                              unsigned tokenAxis) -> int64_t {
        auto ptrType = cast<triton::PointerType>(ptr.getType());
        auto blockType = cast<RankedTensorType>(ptrType.getPointeeType());
        return blockType.getShape()[1 - tokenAxis];
      };
      kFreeExtent = freeExtentOf(kPtr, kTokenAxis);
      vFreeExtent = freeExtentOf(vPtr, vTokenAxis);
    }
    auto accType = cast<RankedTensorType>(loop.getInitArgs()[0].getType());
    if (accType.getRank() != 2)
      return false;
    auto isScoreTile = [&](Type type) {
      auto ranked = dyn_cast<RankedTensorType>(type);
      return ranked && ranked.getRank() == 2 &&
             ranked.getElementType().isF32() &&
             ranked.getShape()[0] == accType.getShape()[0] &&
             ranked.getShape()[1] == blockSize;
    };
    if (!isScoreTile(scoreDot.getType()) || !isScoreTile(valueDot.getType()) ||
        kFreeExtent != accType.getShape()[1] ||
        vFreeExtent != accType.getShape()[1])
      return false;
  }
  // The block predicates must scale the index by the same BS as the tile
  // offsets.
  for (int64_t factor : predicateFactors)
    if (factor != blockSize)
      return false;
  return true;
}

// The tensor-level causal mask.
static bool matchCausalMask(ScanMatchContext &ctx) {
  arith::SelectOp maskSelect = ctx.maskSelect;
  Value rowPosition = ctx.rowPosition;
  Value kTokenOffset = ctx.kTokenOffset;
  Value vTokenOffset = ctx.vTokenOffset;
  int64_t blockSize = ctx.blockSize;
  if (!rowPosition)
    return false;
  // The neutral-path select's condition broadcasts a token-vs-row comparison,
  // the row on the greater side, and the token side is built from the scaled
  // index. This pins the third row-position use to the same SSA value the
  // block guard read.
  Value maskCmp = maskSelect.getCondition();
  while (Operation *def = maskCmp.getDefiningOp())
    if (isa<triton::BroadcastOp, triton::ExpandDimsOp>(def))
      maskCmp = def->getOperand(0);
    else
      break;
  auto causalCmp = maskCmp.getDefiningOp<arith::CmpIOp>();
  if (!causalCmp || !isa<RankedTensorType>(causalCmp.getLhs().getType()))
    return false;
  // The query-position side is a broadcast of the row position, possibly
  // widened: the bwd form compares its tokens against extsi(pid_x) in i64.
  // Peeling the width casts is not a relaxation of identity: the value must
  // still be built from the same SSA the row base and the block guard use.
  auto isPidBroadcast = [&](Value v) {
    return stripIntegerCasts(peelBroadcasts(v)) == rowPosition;
  };
  Value pidSide, tokenSide;
  if (isPidBroadcast(causalCmp.getLhs())) {
    pidSide = causalCmp.getLhs();
    tokenSide = causalCmp.getRhs();
  } else if (isPidBroadcast(causalCmp.getRhs())) {
    pidSide = causalCmp.getRhs();
    tokenSide = causalCmp.getLhs();
  } else {
    return false;
  }
  {
    arith::CmpIPredicate pred = causalCmp.getPredicate();
    bool pidGreater =
        (pred == arith::CmpIPredicate::sge &&
         pidSide == causalCmp.getLhs()) ||
        (pred == arith::CmpIPredicate::sle && pidSide == causalCmp.getRhs());
    if (!pidGreater)
      return false;
  }
  // The token side must be exactly splat(X) + make_range{0, BS} over the same
  // SSA value as a tile token offset, with X the block's token start. The
  // builder recomputes the token positions from that start, so a diagonal
  // shift, a +-1 nudge or a rescaled index would be dropped silently; those
  // spellings are rejected instead of normalized. Only broadcasts above the
  // comparison are peeled.
  while (Operation *def = tokenSide.getDefiningOp())
    if (isa<triton::BroadcastOp, triton::ExpandDimsOp>(def))
      tokenSide = def->getOperand(0);
    else
      break;
  auto tokenAdd = tokenSide.getDefiningOp<arith::AddIOp>();
  if (!tokenAdd)
    return false;
  auto isBlockTokenRange = [&](Value maybeSplat, Value maybeRange) -> bool {
    auto splat = maybeSplat.getDefiningOp<triton::SplatOp>();
    // bwd spells the token range as extsi(make_range{0, BS}); the builder
    // recomputes the token positions, so the cast chain is not carried over.
    auto range =
        stripIntegerCasts(maybeRange).getDefiningOp<triton::MakeRangeOp>();
    if (!splat || !range)
      return false;
    Value index = splat.getOperand();
    if (index != kTokenOffset && index != vTokenOffset)
      return false;
    auto rangeType = dyn_cast<RankedTensorType>(Type(range.getType()));
    if (!rangeType || rangeType.getRank() != 1 ||
        !rangeType.getElementType().isInteger(32))
      return false;
    return rangeType.getShape()[0] == blockSize && range.getStart() == 0 &&
           range.getEnd() == blockSize;
  };
  if (!isBlockTokenRange(tokenAdd.getLhs(), tokenAdd.getRhs()) &&
      !isBlockTokenRange(tokenAdd.getRhs(), tokenAdd.getLhs()))
    return false;
  return true;
}

// The tile width contract (BS, BN = max(64, BS), R = BN/BS).
static bool matchTileContract(ScanMatchContext &ctx) {
  int64_t blockSize = ctx.blockSize;
  int64_t &scanTileWidth = ctx.scanTileWidth;
  int64_t &blocksPerTile = ctx.blocksPerTile;
  // BS must be a power of two, BN = max(64, BS), and BN % BS == 0: a block
  // number owns BS consecutive tokens. BN comes from IR constants, never a
  // new kernel argument.
  if (!isPowerOfTwo(blockSize))
    return false;
  scanTileWidth = std::max<int64_t>(64, blockSize);
  if (scanTileWidth % blockSize != 0)
    return false;
  blocksPerTile = scanTileWidth / blockSize;
  return true;
}

// The vocabulary gate over everything inside the gather loop.
static bool checkVocabulary(ScanMatchContext &ctx) {
  scf::ForOp loop = ctx.loop;
  const bool additive = ctx.additive;
  // Everything inside the gather loop must be either rebuilt or safely
  // deletable, so no unrecognized op may carry effects: a store, atomic or
  // call would be silently dropped. tt.trans is only rebuilt by the additive
  // template, so it stays unrecognized for the online-softmax form.
  bool supportedVocabulary = true;
  loop->walk([&](Operation *op) {
    if (isa<scf::ForOp, scf::IfOp, scf::YieldOp>(op))
      return;
    if (isa<triton::AddPtrOp, triton::LoadOp, triton::MakeTensorPtrOp,
            triton::DotOp, triton::MakeRangeOp, triton::SplatOp,
            triton::ExpandDimsOp, triton::BroadcastOp, triton::ReduceOp,
            triton::ReduceReturnOp, triton::AssertOp>(op))
      return;
    if (additive && isa<triton::TransOp>(op))
      return;
    StringRef ns = op->getName().getDialectNamespace();
    if (ns == "arith" || ns == "math")
      return;
    supportedVocabulary = false;
  });
  if (!supportedVocabulary)
    return false;
  return true;
}

// Run every stage in order, then assemble the candidate; a false from any
// stage declines the match.
static std::optional<ScanCandidate> matchSelectedBlockScan(triton::FuncOp fn) {
  ScanMatchContext ctx;
  ctx.fn = fn;
  auto check = [&](const char *stage, bool ok) {
    LLVM_DEBUG(if (!ok) llvm::dbgs()
                   << "selected-block-scan: stage " << stage << " declined\n");
    return ok;
  };
  if (!check("skeleton", matchLoopSkeleton(ctx)) ||
      !check("index-table", matchIndexTable(ctx)) ||
      !check("block-guard", matchBlockGuard(ctx)) ||
      !check("if-shape", matchIfShape(ctx)) ||
      !check("carried-state", matchCarriedState(ctx)) ||
      !check("dot-operands", matchDotOperands(ctx)) ||
      !check("tiles", matchTiles(ctx)) ||
      !check("causal-mask", matchCausalMask(ctx)) ||
      !check("tile-contract", matchTileContract(ctx)) ||
      !check("vocabulary", checkVocabulary(ctx)))
    return std::nullopt;

  ScanCandidate candidate;
  candidate.function = fn;
  candidate.loop = ctx.loop;
  candidate.pid = ctx.pid;
  candidate.rowPosition = ctx.rowPosition ? ctx.rowPosition : ctx.pid;
  candidate.tableBase = ctx.tableBase;
  candidate.tableLength = ctx.tableLength;
  candidate.blockSize = ctx.blockSize;
  candidate.scanTileWidth = ctx.scanTileWidth;
  candidate.blocksPerTile = ctx.blocksPerTile;
  candidate.scanBound = ctx.loop.getUpperBound();
  candidate.kPtr = ctx.kPtr;
  candidate.vPtr = ctx.vPtr;
  candidate.kLoad = ctx.kLoad;
  candidate.vLoad = ctx.vLoad;
  candidate.kTokenAxis = ctx.kTokenAxis;
  candidate.vTokenAxis = ctx.vTokenAxis;
  candidate.dynamicBoundResult = ctx.dynamicBoundResult;
  candidate.dynamicBoundInput = ctx.dynamicBoundInput;
  candidate.scoreDot = ctx.scoreDot;
  candidate.valueDot = ctx.valueDot;
  candidate.maskSelect = ctx.maskSelect;
  candidate.oArg = ctx.oIdx;
  candidate.mArg = ctx.mIdx;
  candidate.lArg = ctx.lIdx;
  candidate.additive = ctx.additive;
  candidate.accArg = 0;
  candidate.accDot = ctx.additiveState.accDot;
  candidate.transpose = ctx.additiveState.transpose;
  candidate.kTile = ctx.kTile;
  candidate.vTile = ctx.vTile;
  candidate.lseVec = ctx.additiveState.lseVec;
  candidate.deltaVec = ctx.additiveState.deltaVec;
  return candidate;
}

// ---- The online-softmax (fwd) carried state: (o, m, l), read from the
// then-yield. ----
static bool matchOnlineSoftmaxState(scf::ForOp loop, scf::IfOp branch,
                                    scf::YieldOp thenYield, unsigned &oIdx,
                                    unsigned &mIdx, unsigned &lIdx,
                                    triton::DotOp &scoreDot,
                                    triton::DotOp &valueDot,
                                    arith::SelectOp &maskSelect) {
  oIdx = 3, mIdx = 3, lIdx = 3;
  for (unsigned i = 0; i < 3; ++i) {
    Value yielded = thenYield->getOperand(i);
    if (yielded.getDefiningOp<arith::MaxNumFOp>() && mIdx == 3)
      mIdx = i;
    else if (yielded.getDefiningOp<arith::AddFOp>() && lIdx == 3)
      lIdx = i;
    else if (yielded.getDefiningOp<triton::DotOp>() && oIdx == 3)
      oIdx = i;
  }
  if (mIdx == 3 || lIdx == 3 || oIdx == 3)
    return false;
  BlockArgument oArg = loop.getRegionIterArgs()[oIdx];
  BlockArgument mArg = loop.getRegionIterArgs()[mIdx];
  BlockArgument lArg = loop.getRegionIterArgs()[lIdx];
  Value mNew = thenYield->getOperand(mIdx);
  Value lNew = thenYield->getOperand(lIdx);
  Value oNew = thenYield->getOperand(oIdx);

  // m_new = maxnumf(m_old, tile_max); tile_max = reduce{maxnumf}(masked
  // scores); masked scores = select(mask, dot(q, k), -inf).
  auto mUpdate = mNew.getDefiningOp<arith::MaxNumFOp>();
  if (!mUpdate)
    return false;
  Value tileMax;
  if (mUpdate.getLhs() == mArg)
    tileMax = mUpdate.getRhs();
  else if (mUpdate.getRhs() == mArg)
    tileMax = mUpdate.getLhs();
  else
    return false;
  auto tileReduce = tileMax.getDefiningOp<triton::ReduceOp>();
  if (!tileReduce || tileReduce.getAxis() != 1 ||
      tileReduce->getParentRegion() != &branch.getThenRegion())
    return false;
  if (auto *combiner = getReduceCombiner(tileReduce);
      !combiner || !isa<arith::MaxNumFOp>(combiner))
    return false;
  maskSelect = tileReduce->getOperand(0).getDefiningOp<arith::SelectOp>();
  if (!maskSelect || !isNegativeInfSplatConstant(maskSelect.getFalseValue()))
    return false;
  scoreDot = maskSelect.getTrueValue().getDefiningOp<triton::DotOp>();
  if (!scoreDot || scoreDot->getParentRegion() != &branch.getThenRegion())
    return false;
  if (!isZeroSplatConstant(scoreDot.getC()))
    return false; // a fresh matmul, not an accumulation into carried state

  // alpha = exp(m_old - m_new); l = l*alpha + sum(p); the subtraction order
  // is semantic, everything else is checked structurally.
  auto lUpdate = lNew.getDefiningOp<arith::AddFOp>();
  if (!lUpdate)
    return false;
  Value lSum;
  auto mulWithOperand = [](Value v, Value operand, Value &other) {
    auto mul = v.getDefiningOp<arith::MulFOp>();
    if (!mul)
      return false;
    if (mul.getLhs() == operand) {
      other = mul.getRhs();
      return true;
    }
    if (mul.getRhs() == operand) {
      other = mul.getLhs();
      return true;
    }
    return false;
  };
  Value alpha;
  if (mulWithOperand(lUpdate.getLhs(), lArg, alpha))
    lSum = lUpdate.getRhs();
  else if (mulWithOperand(lUpdate.getRhs(), lArg, alpha))
    lSum = lUpdate.getLhs();
  else
    return false;
  auto alphaExp = alpha.getDefiningOp<math::ExpOp>();
  if (!alphaExp)
    return false;
  auto alphaSub = alphaExp.getOperand().getDefiningOp<arith::SubFOp>();
  if (!alphaSub || alphaSub.getLhs() != mArg || alphaSub.getRhs() != mNew)
    return false;
  auto sumReduce = lSum.getDefiningOp<triton::ReduceOp>();
  if (!sumReduce || sumReduce.getAxis() != 1 ||
      sumReduce->getParentRegion() != &branch.getThenRegion())
    return false;
  if (auto *combiner = getReduceCombiner(sumReduce);
      !combiner || !isa<arith::AddFOp>(combiner))
    return false;
  Value pExp = sumReduce->getOperand(0);
  auto pExpOp = pExp.getDefiningOp<math::ExpOp>();
  if (!pExpOp)
    return false;
  auto pSub = pExpOp.getOperand().getDefiningOp<arith::SubFOp>();
  if (!pSub || pSub.getLhs() != tileReduce->getOperand(0))
    return false;
  if (!isBroadcastOf(pSub.getRhs(), mNew))
    return false;

  // o = dot(cast(p), V, o*alpha)
  valueDot = oNew.getDefiningOp<triton::DotOp>();
  if (!valueDot || valueDot->getParentRegion() != &branch.getThenRegion())
    return false;
  Value valueLhs = valueDot.getA();
  if (valueLhs != pExp) {
    Operation *lhsCast = valueLhs.getDefiningOp();
    if (!lhsCast || !isa<arith::TruncFOp, arith::ExtFOp>(lhsCast) ||
        lhsCast->getOperand(0) != pExp)
      return false;
  }
  auto oScale = valueDot.getC().getDefiningOp<arith::MulFOp>();
  if (!oScale)
    return false;
  if (!((oScale.getLhs() == oArg && isBroadcastOf(oScale.getRhs(), alpha)) ||
        (oScale.getRhs() == oArg && isBroadcastOf(oScale.getLhs(), alpha))))
    return false;
  return true;
}

// ---- The additive (bwd dq) matchers ----
//
// The carried-state recurrence is read backwards from the dot that
// accumulates into the state; the tile from its two-level address.
static bool matchAdditiveState(scf::ForOp loop, Region &thenRegion,
                               BlockArgument accArg, Value yielded,
                               AdditiveState &state) {
  // acc' = dot(ds, k^T, acc): the contribution lands in the carried
  // accumulator itself. A one-iter-arg recurrence that is anything else is
  // refused.
  auto accDot = yielded.getDefiningOp<triton::DotOp>();
  if (!accDot || accDot->getParentRegion() != &thenRegion ||
      accDot.getC() != accArg)
    return false;
  // k^T: a transpose of the K tile. Its order decides where the token axis
  // lands, so it is stored and replayed verbatim.
  auto transpose = accDot.getB().getDefiningOp<triton::TransOp>();
  if (!transpose || transpose->getParentRegion() != &thenRegion)
    return false;
  auto kLoad = transpose.getSrc().getDefiningOp<triton::LoadOp>();
  if (!kLoad || kLoad->getParentRegion() != &thenRegion)
    return false;
  // ds = [cast](p * (dp - delta)): the causal select times the
  // delta-corrected value gradient.
  Value ds = accDot.getA();
  if (auto cast = ds.getDefiningOp<arith::TruncFOp>())
    ds = cast.getIn();
  else if (auto cast = ds.getDefiningOp<arith::ExtFOp>())
    ds = cast.getIn();
  auto dsMul = ds.getDefiningOp<arith::MulFOp>();
  if (!dsMul)
    return false;
  auto lhsSelect = dsMul.getLhs().getDefiningOp<arith::SelectOp>();
  auto rhsSelect = dsMul.getRhs().getDefiningOp<arith::SelectOp>();
  if (static_cast<bool>(lhsSelect) == static_cast<bool>(rhsSelect))
    return false;
  auto maskSelect = lhsSelect ? lhsSelect : rhsSelect;
  auto dsSub = (lhsSelect ? dsMul.getRhs() : dsMul.getLhs())
                   .getDefiningOp<arith::SubFOp>();
  if (!dsSub)
    return false;
  // The neutral path is select(causal, p, 0): the additive form's neutral
  // element is zero (a -inf mask belongs to the online-softmax form).
  if (!isZeroSplatConstant(maskSelect.getFalseValue()))
    return false;
  // p = exp(s - lse), lse a loop-invariant [G] row vector read outside.
  auto pExp = maskSelect.getTrueValue().getDefiningOp<math::ExpOp>();
  if (!pExp)
    return false;
  auto scoreSub = pExp.getOperand().getDefiningOp<arith::SubFOp>();
  if (!scoreSub)
    return false;
  auto scoreDot = scoreSub.getLhs().getDefiningOp<triton::DotOp>();
  if (!scoreDot || scoreDot->getParentRegion() != &thenRegion ||
      !isZeroSplatConstant(scoreDot.getC()))
    return false;
  Value lseVec = peelBroadcasts(scoreSub.getRhs());
  if (!isDefinedOutsideLoop(lseVec, loop))
    return false;
  // dp = dot(dO, v, 0), and delta the same kind of loop-invariant row vector.
  auto valueDot = dsSub.getLhs().getDefiningOp<triton::DotOp>();
  if (!valueDot || valueDot->getParentRegion() != &thenRegion ||
      !isZeroSplatConstant(valueDot.getC()))
    return false;
  Value deltaVec = peelBroadcasts(dsSub.getRhs());
  if (!isDefinedOutsideLoop(deltaVec, loop))
    return false;
  // The score dot's rhs IS the transposed tile; the value dot reads the other.
  auto vLoad = valueDot.getB().getDefiningOp<triton::LoadOp>();
  if (scoreDot.getB() != transpose.getSrc() || !vLoad ||
      vLoad->getParentRegion() != &thenRegion)
    return false;
  // Both dots' lhs are loop-invariant operands (Q*scale and dO), and the two
  // row vectors have the accumulator's row count.
  if (!isDefinedOutsideLoop(scoreDot.getA(), loop) ||
      !isDefinedOutsideLoop(valueDot.getA(), loop))
    return false;
  auto accType = cast<RankedTensorType>(accArg.getType());
  auto isRowVector = [&](Value v) {
    auto type = dyn_cast<RankedTensorType>(v.getType());
    return type && type.getRank() == 1 && type.getElementType().isF32() &&
           type.getShape()[0] == accType.getShape()[0];
  };
  if (accType.getRank() != 2 || !isRowVector(lseVec) || !isRowVector(deltaVec))
    return false;
  state.scoreDot = scoreDot;
  state.valueDot = valueDot;
  state.accDot = accDot;
  state.maskSelect = maskSelect;
  state.transpose = transpose;
  state.lseVec = lseVec;
  state.deltaVec = deltaVec;
  return true;
}

// Identify one additive-form tile: the two-level address, its two mask factors
// and the token-range bound, i.e. everything the builder needs to re-emit the
// load at a different token start.
static bool matchAddPtrTile(triton::LoadOp load, Value rawIndex,
                            scf::ForOp loop, AddPtrTile &tile) {
  // The gather version reads exactly one block, so an unmasked tile read
  // would not be in bounds past the row's token count: this form always
  // carries a mask and other = 0.
  if (!load.getMask() || !load.getOther() || load.getIsVolatile())
    return false;
  if (!isZeroSplatConstant(load.getOther()))
    return false;
  // The rewrite reproduces the address and the mask, not the mptr-style
  // boundary check, so a load that carries one is out of scope.
  if (!load.getBoundaryCheck().empty() || load.getPadding())
    return false;
  auto block = dyn_cast<RankedTensorType>(load.getType());
  if (!block || block.getRank() != 2)
    return false;
  auto add = load.getPtr().getDefiningOp<triton::AddPtrOp>();
  if (!add)
    return false;
  auto ptrBroadcast = add.getPtr().getDefiningOp<triton::BroadcastOp>();
  auto offBroadcast = add.getOffset().getDefiningOp<triton::BroadcastOp>();
  if (!ptrBroadcast || !offBroadcast)
    return false;

  // Token term: expand_dims(tokens, 0) * stride, the stride a constant.
  auto tokenMul = offBroadcast.getSrc().getDefiningOp<arith::MulIOp>();
  if (!tokenMul)
    return false;
  std::optional<int64_t> stride;
  Value tokenTerm;
  if ((stride = getSplatIntTensorValue(tokenMul.getRhs())))
    tokenTerm = tokenMul.getLhs();
  else if ((stride = getSplatIntTensorValue(tokenMul.getLhs())))
    tokenTerm = tokenMul.getRhs();
  else
    return false;
  if (*stride <= 0)
    return false;
  auto tokenExpanded = tokenTerm.getDefiningOp<triton::ExpandDimsOp>();
  if (!tokenExpanded || tokenExpanded.getAxis() != 0)
    return false;
  Value tokens = tokenExpanded.getSrc();
  auto tokenVector = dyn_cast<RankedTensorType>(tokens.getType());
  if (!tokenVector || tokenVector.getRank() != 1)
    return false;
  auto tokenInt = dyn_cast<IntegerType>(tokenVector.getElementType());
  if (!tokenInt || tokenInt.getWidth() < 32)
    return false;
  // tokens = splat(idx*BS) + range{0, BS}: the block's token start and the
  // tile's token extent are the same product the block guard scales by.
  auto tokenAdd = tokens.getDefiningOp<arith::AddIOp>();
  if (!tokenAdd)
    return false;
  auto isBlockTokenRange = [&](Value maybeSplat, Value maybeRange) {
    auto splat = maybeSplat.getDefiningOp<triton::SplatOp>();
    if (!splat)
      return false;
    auto factor = matchIdxTimesBS(splat.getOperand(), rawIndex);
    if (!factor || *factor <= 0)
      return false;
    auto range =
        stripIntegerCasts(maybeRange).getDefiningOp<triton::MakeRangeOp>();
    if (!range)
      return false;
    auto rangeType = dyn_cast<RankedTensorType>(range.getType());
    if (!rangeType || rangeType.getRank() != 1 || range.getStart() != 0 ||
        range.getEnd() != *factor)
      return false;
    tile.blockStart = splat.getOperand();
    tile.blockSize = *factor;
    return true;
  };
  if (!isBlockTokenRange(tokenAdd.getLhs(), tokenAdd.getRhs()) &&
      !isBlockTokenRange(tokenAdd.getRhs(), tokenAdd.getLhs()))
    return false;
  if (block.getShape()[1] != tile.blockSize)
    return false;

  // Free term: addptr(splat(base), expand_dims(free, 1)).
  auto inner = ptrBroadcast.getSrc().getDefiningOp<triton::AddPtrOp>();
  if (!inner)
    return false;
  auto freeExpanded = inner.getOffset().getDefiningOp<triton::ExpandDimsOp>();
  if (!freeExpanded || freeExpanded.getAxis() != 1)
    return false;
  auto baseSplat = inner.getPtr().getDefiningOp<triton::SplatOp>();
  if (!baseSplat || !isa<triton::PointerType>(baseSplat.getSrc().getType()))
    return false;
  // No piece of the address chain may read memory, and the free axis is
  // loop-invariant (it never involves the block index).
  if (!isDefinedOutsideLoop(baseSplat.getSrc(), loop) ||
      reachesLoad(baseSplat.getSrc()))
    return false;
  if (!isDefinedOutsideLoop(freeExpanded.getSrc(), loop) ||
      reachesLoad(freeExpanded.getSrc()))
    return false;
  tile.base = baseSplat.getSrc();
  tile.freeVec = freeExpanded.getSrc();
  tile.freeExtent = block.getShape()[0];
  tile.tokenStride = *stride;
  tile.tokenType = tokenInt;

  // Mask: (free < F) & (tokens < T), each broadcast from one row. Splitting
  // them by "does it reach the token block start" keeps the token bound next to
  // the address it protects.
  auto maskAnd = load.getMask().getDefiningOp<arith::AndIOp>();
  if (!maskAnd)
    return false;
  auto peelBroadcast = [](Value v) -> Value {
    if (auto bc = v.getDefiningOp<triton::BroadcastOp>())
      return bc.getSrc();
    return Value();
  };
  Value maskLhs = peelBroadcast(maskAnd.getLhs());
  Value maskRhs = peelBroadcast(maskAnd.getRhs());
  if (!maskLhs || !maskRhs)
    return false;
  bool lhsIsToken = dependsOn(maskLhs, tile.blockStart);
  if (lhsIsToken == dependsOn(maskRhs, tile.blockStart))
    return false;
  Value tokenMask = lhsIsToken ? maskLhs : maskRhs;
  Value freeMask = lhsIsToken ? maskRhs : maskLhs;
  // The token mask must be built from the same token vector as the address
  // and compare it against a loop-invariant scalar bound.
  auto tokenMaskRow = tokenMask.getDefiningOp<triton::ExpandDimsOp>();
  if (!tokenMaskRow || tokenMaskRow.getAxis() != 0)
    return false;
  auto tokenCmp = tokenMaskRow.getSrc().getDefiningOp<arith::CmpIOp>();
  if (!tokenCmp)
    return false;
  Value boundSide;
  if (tokenCmp.getLhs() == tokens &&
      tokenCmp.getPredicate() == arith::CmpIPredicate::slt)
    boundSide = tokenCmp.getRhs();
  else if (tokenCmp.getRhs() == tokens &&
           tokenCmp.getPredicate() == arith::CmpIPredicate::sgt)
    boundSide = tokenCmp.getLhs();
  else
    return false;
  auto boundSplat = boundSide.getDefiningOp<triton::SplatOp>();
  if (!boundSplat || !isDefinedOutsideLoop(boundSplat.getSrc(), loop) ||
      reachesLoad(boundSplat.getSrc()))
    return false;
  tile.tokenBound = boundSplat.getSrc();
  // The free mask must be built from the same free offset as the address: a
  // shifted or rescaled free bound would silently drop padding rows.
  auto freeCmp = freeMask.getDefiningOp<arith::CmpIOp>();
  if (!freeCmp)
    return false;
  if (freeCmp.getLhs() == freeExpanded.getResult() &&
      freeCmp.getPredicate() == arith::CmpIPredicate::slt)
    tile.freeBoundRhs = freeCmp.getRhs();
  else if (freeCmp.getRhs() == freeExpanded.getResult() &&
           freeCmp.getPredicate() == arith::CmpIPredicate::sgt)
    tile.freeBoundRhs = freeCmp.getLhs();
  else
    return false;
  return isDefinedOutsideLoop(tile.freeBoundRhs, loop) &&
         !reachesLoad(tile.freeBoundRhs);
}

// ---- The builder ----
//
// Emits the scan form on the sandbox clone; the original function is
// untouched. Three constraints are hard:
//   1. safe_m subtracts from BOTH the rescale factor and the scores (online
//      softmax template only),
//   2. the per-token weight multiplies p in f32 BEFORE the bf16 truncation
//      (never the -inf mask, where -inf * 0 would be NaN),
//   3. BN is computed from IR constants; no kernel argument is added.

// ---- Shared scaffolding ----

// Build "tt.reduce" with a two-argument combine region. The result type is
// inferred from the input and the axis.
template <typename CombineOp>
static Value buildReduce(IRRewriter &rewriter, Location loc, Value input,
                         int32_t axis) {
  auto tensor = cast<RankedTensorType>(input.getType());
  auto reduce = rewriter.create<triton::ReduceOp>(
      loc, ValueRange{input}, static_cast<int>(axis));
  Block &combine = reduce.getCombineOp().emplaceBlock();
  combine.addArguments({tensor.getElementType(), tensor.getElementType()},
                       {loc, loc});
  rewriter.setInsertionPointToStart(&combine);
  Value combined =
      rewriter.create<CombineOp>(loc, combine.getArgument(0),
                                 combine.getArgument(1));
  rewriter.create<triton::ReduceReturnOp>(loc, ValueRange{combined});
  rewriter.setInsertionPointAfter(reduce.getOperation());
  return reduce->getResult(0);
}

// Build a tt.dot, carrying over the original dot's attributes (input
// precision and friends). The generic creation form avoids depending on a
// specific builder overload for defaulted attributes.
static Value buildDot(IRRewriter &rewriter, Location loc, Type resultType,
                      Value a, Value b, Value c, Operation *original) {
  SmallVector<Value> operands{a, b, c};
  Operation *dot = rewriter.create(loc, original->getName().getIdentifier(),
                                   operands, TypeRange{resultType},
                                   original->getAttrs());
  return dot->getResult(0);
}

// The prologue both templates share. The index table is read once as a
// vector, starts = ids*BS vectorizes the old i_s, slot_valid gates the stale
// [NS, S) tail of that read, and the scan bound is cdiv(i_t + 1, BN) folded
// to divsi(pid_x + BN, BN). Only the way the templates consume these values
// differs.
struct ScanPrologue {
  Value starts;         // [S] i32, ids * BS
  Value slotValid;      // [S] i1, offs < NS
  Value startsRow;      // [1, S] i32
  Value slotRow;        // [1, S] i1
  Value tokensRange;    // [BN] i32, [0, BN)
  Value pidRow;         // [BN] i32, splat(pid_x)
  Value bsConstant;     // i32 BS
  Value zeroTokens;     // [BN] f32 zero
  Value zeroRows;       // [G] f32 zero, used only by the fwd template
  Value negInfRows;     // [G] f32 -inf, used only by the fwd template
  Value scanUpperBound; // i32, the new trip count
  Value bound64;        // i64 K/V dynamic bound (online-softmax only)
  Value c0, c1, cBN;    // i32 constants of the new loop
};

static bool buildScanPrologue(IRRewriter &rewriter,
                              ScanCandidate &candidate, Type i32, Type i64,
                              Type f32, int64_t S, int64_t BS, int64_t BN,
                              int64_t G, ScanPrologue &prologue) {
  Location loc = candidate.loop.getLoc();
  auto tensorTy = [](ArrayRef<int64_t> shape, Type elem) {
    return RankedTensorType::get(shape, elem);
  };
  Value offs = rewriter.create<triton::MakeRangeOp>(loc, tensorTy({S}, i32),
                                                    0, S);
  Type tablePtrTensor =
      tensorTy({S}, candidate.tableBase.getType());
  Value tablePtrs = rewriter.create<triton::SplatOp>(
      loc, tablePtrTensor, candidate.tableBase);
  Value tablePtrVec = rewriter.create<triton::AddPtrOp>(
      loc, tablePtrTensor, tablePtrs, offs);
  Value ids = rewriter.create<triton::LoadOp>(
      loc, tablePtrVec, triton::CacheModifier::NONE,
      triton::EvictionPolicy::NORMAL, false);
  Value idsI32 = ids;
  if (cast<RankedTensorType>(ids.getType()).getElementType() != i32)
    idsI32 = rewriter.create<arith::TruncIOp>(loc, tensorTy({S}, i32), ids);

  // starts = ids * BS, the vectorized old i_s.
  Value bsConstant =
      rewriter.create<arith::ConstantIntOp>(loc, BS, 32);
  Value bsSplat =
      rewriter.create<triton::SplatOp>(loc, tensorTy({S}, i32), bsConstant);
  Value starts = rewriter.create<arith::MulIOp>(loc, idsI32, bsSplat);

  // slot_valid = offs < NS. The old trip count IS NS; when it is a
  // block_counts load it must survive here to gate the stale [NS, S) tail of
  // the fixed-width vector read.
  //
  // NS is clamped into the contract range [0, S] in ITS OWN integer width
  // before any truncation to i32: an out-of-contract NS > S would make
  // `offs < NS` true for every slot and drop the [S, NS) tail, and an i64
  // NS > INT32_MAX would truncate negative and silently yield an empty
  // result. The constant path (NS == S, cross-checked in the matcher) is left
  // alone so its output stays byte-identical.
  Value nsI32;
  if (auto nsConstant = getConstantIntValue(candidate.scanBound)) {
    nsI32 = rewriter.create<arith::ConstantIntOp>(loc, *nsConstant, 32);
  } else {
    Type nsType = candidate.scanBound.getType();
    if (!nsType.isInteger(32) && !nsType.isInteger(64))
      return false;
    Value nsZero =
        rewriter.create<arith::ConstantIntOp>(loc, nsType, int64_t{0});
    Value nsMax = rewriter.create<arith::ConstantIntOp>(loc, nsType, S);
    Value nsLow =
        rewriter.create<arith::MaxSIOp>(loc, candidate.scanBound, nsZero);
    Value nsClamped = rewriter.create<arith::MinSIOp>(loc, nsLow, nsMax);
    nsI32 = nsType.isInteger(32)
                ? nsClamped
                : rewriter.create<arith::TruncIOp>(loc, i32, nsClamped);
  }
  Value nsSplat =
      rewriter.create<triton::SplatOp>(loc, tensorTy({S}, i32), nsI32);
  Value slotValid = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::slt, offs, nsSplat);

  // ---- Scan bound: cdiv(i_t + 1, BN) folded to divsi(pid_x + BN, BN), i32
  // regardless of the old iv's width. ----
  // The bound and the causal mask are keyed by the causal row position, which
  // is pid_x or pid_x advanced by the decode offset (Q_OFFSET).
  Value rowPosition =
      candidate.rowPosition ? candidate.rowPosition : candidate.pid;
  if (rowPosition.getType() != i32) {
    unsigned width = rowPosition.getType().getIntOrFloatBitWidth();
    if (width > 32)
      rowPosition = rewriter.create<arith::TruncIOp>(loc, i32, rowPosition);
    else if (width < 32)
      rowPosition = rewriter.create<arith::ExtSIOp>(loc, i32, rowPosition);
    else
      return false;
  }
  Value c0 = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
  Value c1 = rewriter.create<arith::ConstantIntOp>(loc, 1, 32);
  Value cBN = rewriter.create<arith::ConstantIntOp>(loc, BN, 32);
  Value scanUpperBound = rewriter.create<arith::DivSIOp>(
      loc, rewriter.create<arith::AddIOp>(loc, rowPosition, cBN), cBN);

  // Re-create the K/V dynamic bound outside the loop, unfolded.
  Value bound64 = candidate.dynamicBoundResult;
  if (candidate.dynamicBoundInput)
    bound64 = rewriter.create<arith::ExtSIOp>(loc, i64,
                                              candidate.dynamicBoundInput);
  auto mapAddressOperand = [&](Value v) -> Value {
    return v == candidate.dynamicBoundResult ? bound64 : v;
  };

  // Loop-invariant rows of the weight matrix (hoisted ahead of the loop).
  Value startsRow = rewriter.create<triton::ExpandDimsOp>(
      loc, tensorTy({1, S}, i32), starts, 0);
  Value slotRow = rewriter.create<triton::ExpandDimsOp>(
      loc, tensorTy({1, S}, rewriter.getI1Type()), slotValid, 0);
  Value tokensRange =
      rewriter.create<triton::MakeRangeOp>(loc, tensorTy({BN}, i32), 0, BN);
  Value pidRow =
      rewriter.create<triton::SplatOp>(loc, tensorTy({BN}, i32), rowPosition);

  // Fresh neutral constants; later canonicalization/CSE folds them into the
  // surviving ones. Keeping them fresh makes the builder independent of the
  // original constants' shapes when BN != BS.
  Value zeroTokens = rewriter.create<arith::ConstantOp>(
      loc, DenseElementsAttr::get(tensorTy({BN}, f32),
                                  rewriter.getFloatAttr(f32, 0.0)));
  Value zeroRows = rewriter.create<arith::ConstantOp>(
      loc, DenseElementsAttr::get(tensorTy({G}, f32),
                                  rewriter.getFloatAttr(f32, 0.0)));
  Value negInfRows = rewriter.create<arith::ConstantOp>(
      loc,
      DenseElementsAttr::get(
          tensorTy({G}, f32),
          rewriter.getFloatAttr(
              f32, APFloat::getInf(APFloat::IEEEsingle(), true))));

  prologue.starts = starts;
  prologue.slotValid = slotValid;
  prologue.startsRow = startsRow;
  prologue.slotRow = slotRow;
  prologue.tokensRange = tokensRange;
  prologue.pidRow = pidRow;
  prologue.bsConstant = bsConstant;
  prologue.zeroTokens = zeroTokens;
  prologue.zeroRows = zeroRows;
  prologue.negInfRows = negInfRows;
  prologue.scanUpperBound = scanUpperBound;
  prologue.bound64 = bound64;
  prologue.c0 = c0;
  prologue.c1 = c1;
  prologue.cBN = cBN;
  return true;
}

// ---- The per-token weight both templates share ----
//
// The number of times the token's block was selected. BN == BS collapses the
// block vector to the tile start; R = BN/BS > 1 keeps one row per block and
// spreads each block's count over its BS tokens. valid = (weight > 0) &
// (token <= i_t) merges selection with the causal predicate, which is what
// lets the rewrite replace the per-block scf.if with a per-token zero.
struct ScanWeights {
  Value tokenCounts; // [BN] f32
  Value tokens;      // [BN] i32, token_start + [0, BN)
  Value valid;       // [BN] i1
};

static ScanWeights buildScanWeights(IRRewriter &rewriter, Location loc,
                                    Type i32, Type f32, int64_t S,
                                    int64_t BS, int64_t BN, int64_t R,
                                    Value iv, Value tokenStart,
                                    const ScanPrologue &prologue) {
  auto tensorTy = [](ArrayRef<int64_t> shape, Type elem) {
    return RankedTensorType::get(shape, elem);
  };
  Value startsRow = prologue.startsRow;
  Value slotRow = prologue.slotRow;
  Value bsConstant = prologue.bsConstant;
  Value zeroTokens = prologue.zeroTokens;
  Value pidRow = prologue.pidRow;
  Value tokensRange = prologue.tokensRange;
  Value matchesEq;
  Value slotMatrix = slotRow;
  if (R == 1) {
    Value blockStartRow = rewriter.create<triton::SplatOp>(
        loc, tensorTy({1, S}, i32), tokenStart);
    matchesEq = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, startsRow, blockStartRow);
  } else {
    Value cR = rewriter.create<arith::ConstantIntOp>(loc, R, 32);
    Value blocksBase = rewriter.create<arith::MulIOp>(loc, iv, cR);
    Value blocks = rewriter.create<arith::AddIOp>(
        loc,
        rewriter.create<triton::SplatOp>(loc, tensorTy({R}, i32), blocksBase),
        rewriter.create<triton::MakeRangeOp>(loc, tensorTy({R}, i32), 0, R));
    Value blockStarts = rewriter.create<arith::MulIOp>(
        loc, blocks,
        rewriter.create<triton::SplatOp>(loc, tensorTy({R}, i32), bsConstant));
    Value startsMatrix = rewriter.create<triton::BroadcastOp>(
        loc, tensorTy({R, S}, i32), startsRow);
    Value blockMatrix = rewriter.create<triton::BroadcastOp>(
        loc, tensorTy({R, S}, i32),
        rewriter.create<triton::ExpandDimsOp>(loc, tensorTy({R, 1}, i32),
                                              blockStarts, 1));
    matchesEq = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, startsMatrix, blockMatrix);
    slotMatrix = rewriter.create<triton::BroadcastOp>(
        loc, tensorTy({R, S}, rewriter.getI1Type()), slotRow);
  }
  Value matches = rewriter.create<arith::AndIOp>(loc, matchesEq, slotMatrix);
  SmallVector<int64_t> matrixShape{R, S};
  Value countsI = rewriter.create<arith::ExtUIOp>(
      loc, tensorTy(matrixShape, i32), matches);
  Value counts = buildReduce<arith::AddIOp>(rewriter, loc, countsI, 1);
  Value countsF = rewriter.create<arith::SIToFPOp>(
      loc, tensorTy({R}, f32), counts);
  Value tokenCounts2D = rewriter.create<triton::BroadcastOp>(
      loc, tensorTy({R, BS}, f32),
      rewriter.create<triton::ExpandDimsOp>(loc, tensorTy({R, 1}, f32),
                                            countsF, 1));
  Value tokenCounts = rewriter.create<triton::ReshapeOp>(
      loc, tensorTy({BN}, f32), tokenCounts2D, false);

  // valid = (weight > 0) & (token <= i_t): selection plus causality. A -1
  // padding sentinel never matches a tile start, so it needs no special case;
  // blocks past i_t are masked here and the scan bound skips their tiles
  // entirely. The token positions are recomputed as token_start + [0, BN), so
  // the matched causal offset must be exactly zero.
  Value tokens = rewriter.create<arith::AddIOp>(
      loc, rewriter.create<triton::SplatOp>(loc, tensorTy({BN}, i32),
                                            tokenStart),
      tokensRange);
  Value validWeight = rewriter.create<arith::CmpFOp>(
      loc, arith::CmpFPredicate::OGT, tokenCounts, zeroTokens);
  Value validCausal = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::sle, tokens, pidRow);
  Value valid =
      rewriter.create<arith::AndIOp>(loc, validWeight, validCausal);

  return ScanWeights{tokenCounts, tokens, valid};
}

// ---- The new loop: i32 iv, same iter args and initial values. ----
static scf::ForOp buildScanLoop(IRRewriter &rewriter, Location loc,
                                const ScanPrologue &prologue,
                                const SmallVector<Value> &inits) {
  auto scan = rewriter.create<scf::ForOp>(loc, prologue.c0,
                                          prologue.scanUpperBound,
                                          prologue.c1, inits);
  if (!scan.getBody()->empty())
    rewriter.eraseOp(scan.getBody()->getTerminator());
  rewriter.setInsertionPointToStart(scan.getBody());
  return scan;
}

// ---- Scores: the shared zero-operand dot(q, k). ----
struct ScanScores {
  Value zeroScores; // [G, BN] f32 zero, the neutral operand of the dots
  Value scores;     // [G, BN] f32, dot(q, k, 0)
};

static ScanScores buildScores(IRRewriter &rewriter, Location loc, Type f32,
                              int64_t G, int64_t BN, Value bK,
                              triton::DotOp scoreDot) {
  auto tensorTy = [&](ArrayRef<int64_t> shape, Type elem) {
    return RankedTensorType::get(shape, elem);
  };
  Value zeroScores = rewriter.create<arith::ConstantOp>(
      loc, DenseElementsAttr::get(tensorTy({G, BN}, f32),
                                  rewriter.getFloatAttr(f32, 0.0)));
  Value scores = buildDot(rewriter, loc, tensorTy({G, BN}, f32),
                          scoreDot.getA(), bK, zeroScores, scoreDot);
  return ScanScores{zeroScores, scores};
}

// The form-specific template bodies, called from the top-level builder.
static void buildMakeTensorPtrTile(IRRewriter &rewriter, Location loc,
                                   ScanCandidate &candidate, int64_t BN,
                                   Value tokenStart,
                                   const ScanPrologue &prologue, Value &bK,
                                   Value &bV);
static void buildOnlineSoftmaxBody(IRRewriter &rewriter, Location loc, Type f32,
                                   ScanCandidate &candidate, scf::ForOp scan,
                                   int64_t G, int64_t BN, Value scores,
                                   Value valid, Value tokenCounts, Value bV,
                                   BlockArgument oArg, BlockArgument mArg,
                                   BlockArgument lArg, Type oType,
                                   const ScanPrologue &prologue);
static void buildAdditiveBody(IRRewriter &rewriter, Location loc, Type f32,
                              ScanCandidate &candidate, scf::ForOp scan,
                              int64_t G, int64_t BN, Value scores,
                              Value zeroScores, Value valid, Value tokenCounts,
                              Value bK, Value bV, BlockArgument accArg,
                              Type stateType);
static Value buildAddPtrTileLoad(IRRewriter &rewriter, Location loc,
                                 const AddPtrTile &tile,
                                 triton::LoadOp original, Value tokensI32);

// The top-level builder: shared prologue and new loop, per-token weights,
// the scores dot, the template dispatch and the commit.
static bool buildSelectedBlockScan(IRRewriter &rewriter,
                                   ScanCandidate &candidate) {
  scf::ForOp oldLoop = candidate.loop;
  Location loc = oldLoop.getLoc();
  Type i32 = rewriter.getI32Type();
  Type i64 = rewriter.getI64Type();
  Type f32 = rewriter.getF32Type();

  const int64_t S = candidate.tableLength;
  const int64_t BS = candidate.blockSize;
  const int64_t BN = candidate.scanTileWidth;
  const int64_t R = candidate.blocksPerTile;
  const bool additive = candidate.additive;

  // The carried state keeps its original types and initial values. G is the
  // row count: the online-softmax max vector and the additive accumulator
  // share their first dimension with the query rows.
  RankedTensorType stateType = cast<RankedTensorType>(
      oldLoop.getInitArgs()[additive ? candidate.accArg : candidate.mArg]
          .getType());
  const int64_t G = stateType.getShape()[0];
  RankedTensorType oType =
      additive ? stateType
               : cast<RankedTensorType>(
                     oldLoop.getInitArgs()[candidate.oArg].getType());

  SmallVector<Value> inits;
  for (Value init : oldLoop.getInitArgs())
    inits.push_back(init);

  rewriter.setInsertionPoint(oldLoop);

  ScanPrologue prologue;
  if (!buildScanPrologue(rewriter, candidate, i32, i64, f32, S, BS, BN, G,
                         prologue))
    return false;

  scf::ForOp scan = buildScanLoop(rewriter, loc, prologue, inits);
  Value iv = scan.getInductionVar();
  BlockArgument oArg = scan.getRegionIterArgs()[candidate.oArg];
  BlockArgument mArg = scan.getRegionIterArgs()[candidate.mArg];
  BlockArgument lArg = scan.getRegionIterArgs()[candidate.lArg];

  BlockArgument accArg = scan.getRegionIterArgs()[candidate.accArg];

  Value tokenStart = rewriter.create<arith::MulIOp>(loc, iv, prologue.cBN);

  // ---- K/V tiles. The two templates read the same two tiles per
  // iteration; only the addressing differs. ----
  Value bK, bV;
  if (!additive || candidate.kPtr) {
    // The online-softmax form always reads through make_tensor_ptr; the
    // additive form does too when the baseline spelled the tiles that way
    // (production nsa_bwd_dq), and only the addptr spelling needs the rebuilt
    // two-level chain below.
    buildMakeTensorPtrTile(rewriter, loc, candidate, BN, tokenStart,
                           prologue, bK, bV);
  }

  ScanWeights weights = buildScanWeights(rewriter, loc, i32, f32, S, BS, BN,
                                         R, iv, tokenStart, prologue);
  Value tokenCounts = weights.tokenCounts;
  Value tokens = weights.tokens;
  Value valid = weights.valid;

  if (additive && !candidate.kPtr) {
    // Additive tiles, addptr spelling: the baseline's own two-level addptr at
    // the scan tile's token start, with the token-range mask preserved.
    bK = buildAddPtrTileLoad(rewriter, loc, candidate.kTile,
                             candidate.kLoad, tokens);
    bV = buildAddPtrTileLoad(rewriter, loc, candidate.vTile,
                             candidate.vLoad, tokens);
  }

  ScanScores scanScores = buildScores(rewriter, loc, f32, G, BN, bK,
                                      candidate.scoreDot);
  if (additive) {
    buildAdditiveBody(rewriter, loc, f32, candidate, scan, G, BN,
                      scanScores.scores, scanScores.zeroScores, valid,
                      tokenCounts, bK, bV, accArg, stateType);
  } else {
    buildOnlineSoftmaxBody(rewriter, loc, f32, candidate, scan, G, BN,
                           scanScores.scores, valid, tokenCounts, bV, oArg,
                           mArg, lArg, oType, prologue);
  }

  // ---- Commit within the sandbox: the scan loop replaces the gather loop
  // wholesale; the epilogue keeps reading the same (o, m, l) results. ----
  rewriter.replaceOp(oldLoop, scan->getResults());
  return true;
}

// ---- The online-softmax (fwd) scan body ----

// The fwd template's tile read: rebuild the two make_tensor_ptr views at
// the scan tile's token start and load them unconditionally.
static void buildMakeTensorPtrTile(IRRewriter &rewriter, Location loc,
                                   ScanCandidate &candidate, int64_t BN,
                                   Value tokenStart,
                                   const ScanPrologue &prologue, Value &bK,
                                   Value &bV) {
  // The re-created K/V dynamic bound is built inside the prologue so that the
  // forward template's op order is unchanged; the tile views read it through
  // this mapping.
  auto mapAddressOperand = [&](Value v) -> Value {
    return v == candidate.dynamicBoundResult ? prologue.bound64 : v;
  };
  // The same views and strides, token offset = token_start, loads
  // unconditional with the original boundary behaviour: the tile is always
  // fully read and selection happens in the score domain.
  //
  // Because the tile is read unconditionally, every element now takes part in
  // arithmetic the baseline skipped for unselected blocks. K is safe: the
  // score goes through select(valid, s, -inf) and exp(masked - safe_m), so the
  // -inf wins. V is NOT: the weight multiplies b_p in f32 and p_exp is 0 for
  // an unselected token, so the product is 0 * V and V == +-Inf yields NaN.
  // Every V element that can be multiplied by a zero weight must be finite.
  auto rebuildTilePtr = [&](triton::MakeTensorPtrOp ptr,
                            unsigned tokenAxis) -> Value {
    auto ptrType = cast<triton::PointerType>(ptr.getType());
    auto block = cast<RankedTensorType>(ptrType.getPointeeType());
    SmallVector<int64_t> blockShape(block.getShape().begin(),
                                    block.getShape().end());
    blockShape[tokenAxis] = BN;
    Type newPtrType = triton::PointerType::get(
        RankedTensorType::get(blockShape, block.getElementType()),
        ptrType.getAddressSpace());
    SmallVector<Value> shape;
    for (Value v : ptr.getShape())
      shape.push_back(mapAddressOperand(v));
    SmallVector<Value> strides;
    for (Value v : ptr.getStrides())
      strides.push_back(mapAddressOperand(v));
    SmallVector<Value> offsets;
    for (unsigned axis = 0; axis < ptr.getOffsets().size(); ++axis)
      offsets.push_back(axis == tokenAxis
                            ? tokenStart
                            : mapAddressOperand(ptr.getOffsets()[axis]));
    return rewriter.create<triton::MakeTensorPtrOp>(
        loc, newPtrType, mapAddressOperand(ptr.getBase()), shape, strides,
        offsets, ptr.getOrder());
  };
  Value kTilePtr = rebuildTilePtr(candidate.kPtr, candidate.kTokenAxis);
  Value vTilePtr = rebuildTilePtr(candidate.vPtr, candidate.vTokenAxis);
  bK = rewriter.create<triton::LoadOp>(
      loc, kTilePtr, candidate.kLoad.getBoundaryCheck(),
      candidate.kLoad.getPadding(), candidate.kLoad.getCache(),
      candidate.kLoad.getEvict(), candidate.kLoad.getIsVolatile());
  bV = rewriter.create<triton::LoadOp>(
      loc, vTilePtr, candidate.vLoad.getBoundaryCheck(),
      candidate.vLoad.getPadding(), candidate.vLoad.getCache(),
      candidate.vLoad.getEvict(), candidate.vLoad.getIsVolatile());
}

// The fwd template's loop body: safe_m online softmax over the scores,
// then the (o, m, l) yields.
static void buildOnlineSoftmaxBody(IRRewriter &rewriter, Location loc, Type f32,
                                   ScanCandidate &candidate, scf::ForOp scan,
                                   int64_t G, int64_t BN, Value scores,
                                   Value valid, Value tokenCounts, Value bV,
                                   BlockArgument oArg, BlockArgument mArg,
                                   BlockArgument lArg, Type oType,
                                   const ScanPrologue &prologue) {
  auto tensorTy = [&](ArrayRef<int64_t> shape, Type elem) {
    return RankedTensorType::get(shape, elem);
  };
  Value maskRow = rewriter.create<triton::ExpandDimsOp>(
      loc, tensorTy({1, BN}, rewriter.getI1Type()), valid, 0);
  Value maskMatrix = rewriter.create<triton::BroadcastOp>(
      loc, tensorTy({G, BN}, rewriter.getI1Type()), maskRow);
  Value negInfScores = rewriter.create<arith::ConstantOp>(
      loc,
      DenseElementsAttr::get(
          tensorTy({G, BN}, f32),
          rewriter.getFloatAttr(
              f32, APFloat::getInf(APFloat::IEEEsingle(), true))));
  Value maskedScores = rewriter.create<arith::SelectOp>(
      loc, maskMatrix, scores, negInfScores);

  // Online softmax with the safe_m guard. The guard replaces the baseline's
  // scf.if: a fully masked tile with m == -inf would otherwise compute
  // exp(-inf - -inf) = NaN and permanently poison o and l.
  Value tileMax =
      buildReduce<arith::MaxNumFOp>(rewriter, loc, maskedScores, 1);
  Value mNew = rewriter.create<arith::MaxNumFOp>(loc, mArg, tileMax);
  Value needsGuard = rewriter.create<arith::CmpFOp>(
      loc, arith::CmpFPredicate::OEQ, mNew, prologue.negInfRows);
  Value safeM = rewriter.create<arith::SelectOp>(loc, needsGuard,
                                                 prologue.zeroRows,
                                                 mNew);
  // Hard constraint 1: safe_m subtracts from BOTH the rescale factor and
  // the scores. Yielding the TRUE max keeps the epilogue's lse identical.
  Value rescale = rewriter.create<math::ExpOp>(
      loc, rewriter.create<arith::SubFOp>(loc, mArg, safeM));
  Value safeMatrix = rewriter.create<triton::BroadcastOp>(
      loc, tensorTy({G, BN}, f32),
      rewriter.create<triton::ExpandDimsOp>(loc, tensorTy({G, 1}, f32), safeM,
                                            1));
  Value pExp = rewriter.create<math::ExpOp>(
      loc, rewriter.create<arith::SubFOp>(loc, maskedScores, safeMatrix));
  // Hard constraint 2: the weight multiplies b_p, in f32, before the bf16
  // truncation: never the -inf mask (-inf * 0 = NaN) and never after the
  // cast (truncf(p*w) != truncf(p)*w).
  Value weightMatrix = rewriter.create<triton::BroadcastOp>(
      loc, tensorTy({G, BN}, f32),
      rewriter.create<triton::ExpandDimsOp>(loc, tensorTy({1, BN}, f32),
                                            tokenCounts, 0));
  Value pWeighted = rewriter.create<arith::MulFOp>(loc, pExp, weightMatrix);

  // l = l * rescale + sum(p)
  Value lScaled = rewriter.create<arith::MulFOp>(loc, lArg, rescale);
  Value pSum = buildReduce<arith::AddFOp>(rewriter, loc, pWeighted, 1);
  Value lNew = rewriter.create<arith::AddFOp>(loc, lScaled, pSum);

  // o = o * rescale + dot(p, V)
  Value rescaleMatrix = rewriter.create<triton::BroadcastOp>(
      loc, oType,
      rewriter.create<triton::ExpandDimsOp>(loc, tensorTy({G, 1}, f32),
                                            rescale, 1));
  Value oScaled = rewriter.create<arith::MulFOp>(loc, oArg, rescaleMatrix);
  Type valueLhsElem =
      cast<RankedTensorType>(candidate.valueDot.getA().getType())
          .getElementType();
  Value valueLhs = pWeighted;
  if (valueLhsElem != f32) {
    Type valueLhsType = tensorTy({G, BN}, valueLhsElem);
    if (isa<arith::ExtFOp>(candidate.valueDot.getA().getDefiningOp()))
      valueLhs =
          rewriter.create<arith::ExtFOp>(loc, valueLhsType, pWeighted);
    else
      valueLhs =
          rewriter.create<arith::TruncFOp>(loc, valueLhsType, pWeighted);
  }
  Value oNew = buildDot(rewriter, loc, oType, valueLhs, bV, oScaled,
                        candidate.valueDot);

  rewriter.setInsertionPointToEnd(scan.getBody());
  SmallVector<Value> yielded{Value(), Value(), Value()};
  yielded[candidate.oArg] = oNew;
  yielded[candidate.mArg] = mNew;
  yielded[candidate.lArg] = lNew;
  rewriter.create<scf::YieldOp>(loc, yielded);
}

// ---- The additive (bwd dq) scan body ----

// One additive-form tile read at the scan tile's token start: the same
// two-level addptr chain the baseline uses (base + free[:, None], then
// tokens * stride), with the mask rebuilt from the captured bound and the
// free-axis comparison. The token-range term is kept because it is the only
// guard that keeps the wider read inside the row's token count.
static Value buildAddPtrTileLoad(IRRewriter &rewriter, Location loc,
                                 const AddPtrTile &tile,
                                 triton::LoadOp original, Value tokensI32) {
  auto tokenShape = cast<RankedTensorType>(tokensI32.getType()).getShape();
  const int64_t BN = tokenShape[0];
  const int64_t F = tile.freeExtent;
  auto intType = cast<IntegerType>(tile.tokenType);
  Type freeElem =
      cast<RankedTensorType>(tile.freeVec.getType()).getElementType();
  Type baseType = tile.base.getType();
  Type elemType = cast<triton::PointerType>(baseType).getPointeeType();
  Type i1 = rewriter.getI1Type();
  auto tensorTy = [](ArrayRef<int64_t> shape, Type elem) {
    return RankedTensorType::get(shape, elem);
  };

  // Preserve the addressing width of the baseline: the bwd form does its
  // token arithmetic in the index table's width while the free axis stays
  // i32, so the tile's token term is rebuilt at that width.
  Type tokenTensor = tensorTy(tokenShape, intType);
  Value tokens = tokensI32;
  if (tokens.getType() != tokenTensor)
    tokens = rewriter.create<arith::ExtSIOp>(loc, tokenTensor, tokensI32);

  // base + free[:, None], broadcast over the tokens.
  Type freeOne = tensorTy({F, 1}, freeElem);
  Type ptrOne = tensorTy({F, 1}, baseType);
  Type ptrType = tensorTy({F, BN}, baseType);
  auto intOne = tensorTy({1, BN}, intType);
  Value freeOff =
      rewriter.create<triton::ExpandDimsOp>(loc, freeOne, tile.freeVec, 1);
  Value innerPtr = rewriter.create<triton::AddPtrOp>(
      loc, ptrOne,
      rewriter.create<triton::SplatOp>(loc, ptrOne, tile.base), freeOff);
  Value tokenOff = rewriter.create<arith::MulIOp>(
      loc, rewriter.create<triton::ExpandDimsOp>(loc, intOne, tokens, 0),
      rewriter.create<arith::ConstantOp>(
          loc, DenseElementsAttr::get(
                   intOne,
                   rewriter.getIntegerAttr(intType, tile.tokenStride))));
  Value tilePtr = rewriter.create<triton::AddPtrOp>(
      loc, ptrType,
      rewriter.create<triton::BroadcastOp>(loc, ptrType, innerPtr),
      rewriter.create<triton::BroadcastOp>(
          loc, tensorTy({F, BN}, intType), tokenOff));

  // mask = (free < F) & (tokens < T), with the bound captured by the
  // matcher: the token term is the baseline's own range guard, not a
  // boundary check the scan form could synthesize.
  Type maskType = tensorTy({F, BN}, i1);
  Value freeBound = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::slt, freeOff, tile.freeBoundRhs);
  Value tokenMask = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::slt, tokens,
      rewriter.create<triton::SplatOp>(loc, tokenTensor, tile.tokenBound));
  Value mask = rewriter.create<arith::AndIOp>(
      loc, rewriter.create<triton::BroadcastOp>(loc, maskType, freeBound),
      rewriter.create<triton::BroadcastOp>(
          loc, maskType,
          rewriter.create<triton::ExpandDimsOp>(
              loc, tensorTy({1, BN}, i1), tokenMask, 0)));
  // other = 0, the baseline's padding value for masked-off elements.
  auto valueType = tensorTy({F, BN}, elemType);
  Attribute zero = rewriter.getFloatAttr(elemType, 0.0);
  if (elemType.isIntOrIndex())
    zero = rewriter.getIntegerAttr(elemType, 0);
  Value other = rewriter.create<arith::ConstantOp>(
      loc, DenseElementsAttr::get(valueType, zero));
  return rewriter.create<triton::LoadOp>(
      loc, tilePtr, mask, other, original.getCache(), original.getEvict(),
      original.getIsVolatile());
}

// The bwd template's loop body: WHERE-then-weight scores, dp, the
// accumulation dot into the carried state and the single yield.
static void buildAdditiveBody(IRRewriter &rewriter, Location loc, Type f32,
                              ScanCandidate &candidate, scf::ForOp scan,
                              int64_t G, int64_t BN, Value scores,
                              Value zeroScores, Value valid, Value tokenCounts,
                              Value bK, Value bV, BlockArgument accArg,
                              Type stateType) {
  auto tensorTy = [&](ArrayRef<int64_t> shape, Type elem) {
    return RankedTensorType::get(shape, elem);
  };
  // Additive state update (bwd dq). The carried state is a single accumulator
  // with neutral 0 and no rescale, so a tile contributes through the dot's C
  // operand.
  //
  // The WHERE comes BEFORE the weight multiply. An empty query row has
  // lse = -inf, hence exp(s - lse) is +inf on every token; multiplying that by
  // the zero weight first would turn it into inf * 0 = NaN and poison the
  // whole dq row. Selecting to 0 first keeps every invalid lane an exact
  // zero. This is why the forward template's ordering must not be copied.
  Value lseMatrix = rewriter.create<triton::BroadcastOp>(
      loc, tensorTy({G, BN}, f32),
      rewriter.create<triton::ExpandDimsOp>(loc, tensorTy({G, 1}, f32),
                                            candidate.lseVec, 1));
  Value pExp = rewriter.create<math::ExpOp>(
      loc, rewriter.create<arith::SubFOp>(loc, scores, lseMatrix));
  Value validRow = rewriter.create<triton::ExpandDimsOp>(
      loc, tensorTy({1, BN}, rewriter.getI1Type()), valid, 0);
  Value validMatrix = rewriter.create<triton::BroadcastOp>(
      loc, tensorTy({G, BN}, rewriter.getI1Type()), validRow);
  Value pMasked =
      rewriter.create<arith::SelectOp>(loc, validMatrix, pExp, zeroScores);
  Value weightMatrix = rewriter.create<triton::BroadcastOp>(
      loc, tensorTy({G, BN}, f32),
      rewriter.create<triton::ExpandDimsOp>(loc, tensorTy({1, BN}, f32),
                                            tokenCounts, 0));
  Value pWeighted =
      rewriter.create<arith::MulFOp>(loc, pMasked, weightMatrix);

  // dp = dot(dO, v): a fresh matmul, not an accumulation into state.
  Value dp = buildDot(rewriter, loc, tensorTy({G, BN}, f32),
                      candidate.valueDot.getA(), bV, zeroScores,
                      candidate.valueDot);
  Value deltaMatrix = rewriter.create<triton::BroadcastOp>(
      loc, tensorTy({G, BN}, f32),
      rewriter.create<triton::ExpandDimsOp>(loc, tensorTy({G, 1}, f32),
                                            candidate.deltaVec, 1));
  Value ds = rewriter.create<arith::MulFOp>(
      loc, pWeighted,
      rewriter.create<arith::SubFOp>(loc, dp, deltaMatrix));

  // ds enters the accumulation dot in its input type; the baseline
  // truncates in the same place (one rounding of p*w*(dp-delta)).
  Type dsElem = cast<RankedTensorType>(candidate.accDot.getA().getType())
                    .getElementType();
  Value dsCast = ds;
  if (dsElem != f32) {
    Type dsType = tensorTy({G, BN}, dsElem);
    if (isa<arith::ExtFOp>(candidate.accDot.getA().getDefiningOp()))
      dsCast = rewriter.create<arith::ExtFOp>(loc, dsType, ds);
    else
      dsCast = rewriter.create<arith::TruncFOp>(loc, dsType, ds);
  }

  // k^T with the baseline's own order: the position of the token axis is
  // semantic, so a hard-coded [1, 0] would transpose the wrong axis for an
  // equivalent spelling.
  auto kTileType = cast<RankedTensorType>(bK.getType());
  SmallVector<int64_t> transShape;
  for (int32_t axis : candidate.transpose.getOrder())
    transShape.push_back(kTileType.getShape()[axis]);
  Value kT = rewriter.create<triton::TransOp>(
      loc, RankedTensorType::get(transShape, kTileType.getElementType()),
      bK, candidate.transpose.getOrder());

  Value accNew = buildDot(rewriter, loc, stateType, dsCast, kT, accArg,
                          candidate.accDot);
  rewriter.setInsertionPointToEnd(scan.getBody());
  rewriter.create<scf::YieldOp>(loc, accNew);
}

// ---- Plan / Rule / factory ----

// The plan: a privately prepared and verified sandbox plus a commit-only
// apply().

// Tidy the private sandbox before the joint consult, so Rule B judges exactly
// the body apply() commits. `make_ttir` in backend/compiler.py places
// add_graph_optimize last, with no canonicalize/CSE after it, so without this
// cleanup the rewrite would leave splat(constant) and a few dead ops behind
// for the downstream lowering.
static LogicalResult runSandboxStructuralCleanup(ModuleOp sandbox) {
  MLIRContext *context = sandbox.getContext();
  PassManager cleanup(context, ModuleOp::getOperationName());
  cleanup.addPass(createCanonicalizerPass());
  cleanup.addPass(createCSEPass());
  cleanup.addPass(createLoopInvariantCodeMotionPass());
  cleanup.addPass(createCanonicalizerPass());
  cleanup.addPass(createCSEPass());
  return cleanup.run(sandbox);
}

class SelectedBlockScanPlan final : public RewritePlan {
  ScanCandidate candidate;
  unsigned epoch;
  // 已接受并自行验证过的沙箱：里面既有规则 A 的扫描重建，也可能已有规则 B
  // 的行提升与启动契约。
  // The accepted sandbox: it already carries Rule A's scan rebuild and, when the
  // joint gate ran, Rule B's row lifting and launch contract.
  OwningOpRef<ModuleOp> prepared;

  SelectedBlockScanPlan(ScanCandidate candidate, unsigned epoch,
                        OwningOpRef<ModuleOp> prepared)
      : candidate(candidate), epoch(epoch), prepared(std::move(prepared)) {}

public:
  GraphOptimizationRuleId getRuleId() const override {
    return GraphOptimizationRuleId::SelectedBlockScan;
  }

  unsigned getBenefit() const override { return 1; }

  // The anchor stays the real function's gather loop: getAnchor() feeds the
  // driver's program-order map, which is built on the real function, so a
  // sandbox anchor would not be found.
  Operation *getAnchor() const override { return candidate.loop; }

  unsigned getCreationEpoch() const override { return epoch; }

  LogicalResult revalidate(GraphOptimizationContext &context) const override {
    if (context.getFunction() != candidate.function)
      return failure();
    // Re-run the full structural match and compare the anchor and parameters.
    std::optional<ScanCandidate> current =
        matchSelectedBlockScan(candidate.function);
    if (!current || current->loop != candidate.loop ||
        !sameScanCandidate(*current, candidate))
      return failure();
    // Rule B is not re-consulted here: the pre-commit function still has the
    // gather shape Rule B does not match, so a second consult would always
    // decline. The single consult is sound because the body apply() commits IS
    // the sandbox body that was accepted, and the match just re-proved that
    // this candidate rebuilds that body.
    return success();
  }

  LogicalResult apply(IRRewriter &rewriter) override {
    // 沙箱已经通过 verify；这里只做提交。运行期网格时规则 B 会追加隐藏的
    // extent 参数，因此必须连函数签名一起换，不能只 takeBody。
    // Commit only: the sandbox is already verified.  A runtime grid makes Rule B
    // append hidden extent arguments, so the function signature has to travel
    // with the body instead of a plain takeBody.
    static_cast<void>(rewriter);
    auto replacement = cast<triton::FuncOp>(&prepared->getBody()->front());
    ModuleOp preparedModule = *prepared;
    ModuleOp module = candidate.function->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    if (preparedModule->hasAttr(kProgramGridTransformsAttr)) {
      if (failed(commitProgramGridFunctionFromSandbox(candidate.function,
                                                      replacement)))
        return failure();
      module->setAttr(kProgramGridTransformsAttr,
                      preparedModule->getAttr(kProgramGridTransformsAttr));
      return success();
    }

    candidate.function.getBody().takeBody(replacement.getBody());
    for (StringLiteral name :
         {StringLiteral("hacc.coalesce_factor"),
          StringLiteral("hacc.coalesce_axis"),
          StringLiteral("hacc.coalesce_grid_ceil_div")})
      if (preparedModule->hasAttr(name))
        module->setAttr(name, preparedModule->getAttr(name));
    return success();
  }

  // Prepare and verify privately before offering a plan to the scheduler. An
  // unsupported rewrite must leave the input and its diagnostics untouched;
  // declining returns nullptr and never fails apply().
  static std::unique_ptr<SelectedBlockScanPlan>
  create(const ScanCandidate &candidate, unsigned epoch, bool forceApply,
         bool queryRowBatchingEnabled, const ResourceSnapshot &resources) {
    // The tablegen accessors used below are not const-qualified in this tree,
    // so work on a local mutable copy of the function.
    triton::FuncOp function = candidate.function;
    ModuleOp module = function->getParentOfType<ModuleOp>();
    if (!module)
      return nullptr;

    OwningOpRef<ModuleOp> sandbox(ModuleOp::create(function.getLoc()));
    // The sandbox must carry the module attributes: canQueryRowBatch reads
    // module-level gates, and a sandbox without them would decline forever.
    sandbox->getOperation()->setAttrs(module->getAttrs());

    IRMapping mapping;
    sandbox->getBody()->push_back(function->clone(mapping));
    auto cloned = cast<triton::FuncOp>(&sandbox->getBody()->front());

    // Re-match on the clone so recorded operations point into the sandbox.
    std::optional<ScanCandidate> sandboxCandidate =
        matchSelectedBlockScan(cloned);
    if (!sandboxCandidate || !sameScanCandidate(*sandboxCandidate, candidate))
      return nullptr;

    IRRewriter rewriter(module.getContext());
    ScopedDiagnosticHandler diagnostics(
        module.getContext(), [](Diagnostic &diag) {
          LLVM_DEBUG(llvm::dbgs()
                     << "selected-block-scan: skipped candidate: " << diag
                     << "\n");
          return success();
        });
    if (!buildSelectedBlockScan(rewriter, *sandboxCandidate))
      return nullptr;
    // Tidy and verify the private sandbox here so neither a malformed nor an
    // untidied body reaches the scheduler. The cleanup runs before the joint
    // consult. `cloned` stays valid: canonicalize/CSE/LICM rewrite operations
    // inside the function but never erase the tt.func op itself.
    if (failed(runSandboxStructuralCleanup(*sandbox)))
      return nullptr;
    if (failed(verify(*sandbox)))
      return nullptr;

    // 联合决策：只有规则 B 的规则位也打开时，才允许 A 在沙箱里连带完成 B 的
    // 行提升。materialize 就地重写并 verify，失败即整个计划作废；成功则沙箱
    // 里已经有 B 的改写与启动契约，apply() 只需把沙箱提交回去。
    // Joint decision: Rule A may only run Rule B's row lifting when Rule B's bit
    // is opted in as well.  `materialize...` rewrites and verifies in place, so a
    // failure discards the whole plan, and success means the sandbox already
    // carries both rewrites and the launch contract.
    if (!forceApply) {
      if (!queryRowBatchingEnabled) {
        LLVM_DEBUG(llvm::dbgs() << "selected-block-scan: rule B bit is off, "
                                   "declining @"
                                << function.getName() << "\n");
        return nullptr;
      }
      if (failed(materializeQueryRowBatchingCandidate(*sandbox, cloned,
                                                      resources))) {
        LLVM_DEBUG(llvm::dbgs() << "selected-block-scan: joint query-row "
                                   "batching materialization declined @"
                                << function.getName() << "\n");
        return nullptr;
      }
      LLVM_DEBUG(llvm::dbgs() << "selected-block-scan: joint query-row "
                                 "batching materialized @"
                              << function.getName() << "\n");
    } else {
      LLVM_DEBUG(llvm::dbgs() << "selected-block-scan: force-apply bypasses "
                                 "the joint gate for @"
                              << function.getName() << "\n");
    }

    return std::unique_ptr<SelectedBlockScanPlan>(
        new SelectedBlockScanPlan(candidate, epoch, std::move(sandbox)));
  }
};

class SelectedBlockScanRule final : public GraphOptimizationRule {
  bool forceApply;
  bool queryRowBatchingEnabled;

public:
  explicit SelectedBlockScanRule(const SelectedBlockScanRuleOptions &options)
      : forceApply(options.forceApply),
        queryRowBatchingEnabled(options.queryRowBatchingEnabled) {}

  GraphOptimizationRuleId getId() const override {
    return GraphOptimizationRuleId::SelectedBlockScan;
  }

  // 规则 A 需要 L0 预算快照，转交给规则 B 的 materialize 使用。
  // Rule A forwards the L0 budget snapshot to Rule B's materialize entry.
  AnalysisRequirement getAnalysisRequirements() const override {
    return AnalysisRequirement::ResourceCost;
  }

  LogicalResult findCandidates(
      GraphOptimizationContext &context,
      SmallVectorImpl<std::unique_ptr<RewritePlan>> &plans) override {
    // Pure read-only matching first: a miss is a silent continue with no
    // diagnostics and no IR change.
    std::optional<ScanCandidate> candidate =
        matchSelectedBlockScan(context.getFunction());
    if (!candidate)
      return success();

    LLVM_DEBUG(llvm::dbgs()
               << "selected-block-scan: matched gather loop in @"
               << candidate->function.getName()
               << " (BS=" << candidate->blockSize
               << " BN=" << candidate->scanTileWidth
               << " S=" << candidate->tableLength << ")\n");

    if (auto plan = SelectedBlockScanPlan::create(
            *candidate, context.getEpoch(), forceApply, queryRowBatchingEnabled,
            context.getResourceCostAnalysis().getResourceSnapshot()))
      plans.push_back(std::move(plan));
    return success();
  }
};

} // namespace

std::unique_ptr<GraphOptimizationRule>
cfg::createSelectedBlockScanRule(const SelectedBlockScanRuleOptions &options) {
  return std::make_unique<SelectedBlockScanRule>(options);
}
