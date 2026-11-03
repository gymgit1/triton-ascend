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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "TritonToGraph/QueryRowBatching.h"
#include "TritonToGraph/ProgramGridTransform.h"
#include "TritonToGraph/ResourceCostModel.h"
#include "TritonToGraph/RowLifting.h"

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
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

namespace {

// 规则 B（QueryRowBatchingRule，规则位 131072，ProgramMapping 阶段）只负责
// 行提升：保持循环结构，把与查询行相关的值前插行维 [R]（每个 program 处理 R
// 个查询行），并借助公共引擎完成逐行掩码访存与掩码写回。把「按下标表选择」
// 的稀疏循环（nsa 风格）改写成顺序扫描是规则 A（SelectedBlockScanRule，规则位
// 65536）的职责。按 RFC #2686，只有 A 的顺序化与 B 的行提升一起做才有收益，
// 所以 B 对 A 暴露的是 materialize 接口：A 在自己的沙箱里改成顺序扫描后直接
// 调用 B 完成重写，B 不再用「循环体内是否还有下标表读」这类形态代理去反推
// A 是否已经跑过。B 自己的调度入口与 A 的调用入口共用同一份重写实现。
// Rule B (QueryRowBatchingRule, bit 131072, ProgramMapping phase) owns row
// lifting only: it keeps the loop structure and prepends the query-row dimension
// [R] to row-dependent values (R query rows per program), relying on the shared
// engine for per-row masked accesses and masked write-back.  Turning a
// table-indexed sparse loop (nsa-style selection) into a sequential scan belongs
// to rule A (SelectedBlockScanRule, bit 65536).  RFC #2686 only pays off when A's
// sequentialization and B's row lifting happen together, so B exposes a
// materialize entry to A: A sequentializes inside its sandbox and calls B to
// perform the rewrite there.  B no longer infers "has A run?" from shape proxies
// such as an in-loop index table.  B's own scheduling entry shares one rewrite
// implementation with that A-facing entry.

constexpr llvm::StringLiteral kCoalesceFactorAttr = "hacc.coalesce_factor";
constexpr llvm::StringLiteral kCoalesceAxisAttr = "hacc.coalesce_axis";
constexpr llvm::StringLiteral kCoalesceGridCeilDivAttr =
    "hacc.coalesce_grid_ceil_div";

// 读取整数常量；不是常量时返回 nullopt。
static std::optional<int64_t> getConstantIntValue(Value v) {
  APInt value;
  if (matchPattern(v, m_ConstantInt(&value)))
    return value.getSExtValue();
  return std::nullopt;
}

// 计算 dot tile 的存储字节数，用作 L0A/L0C 的每因子需求。
static int64_t tensorStorageBytes(RankedTensorType tensor) {
  Type element = tensor.getElementType();
  const int64_t bytes = isa<triton::PointerType>(element)
                            ? 8
                            : (element.getIntOrFloatBitWidth() + 7) / 8;
  int64_t size = 0;
  if (__builtin_mul_overflow(tensor.getNumElements(), bytes, &size))
    return std::numeric_limits<int64_t>::max();
  return size;
}

// 识别可按规则 B 重建的查询行循环，返回候选（行数、能力开关、静态网格）。
// 这里只判行提升自身必需的前提，不判循环是不是顺序扫描：顺序化由规则 A 负责，
// A 通过 materializeQueryRowBatchingCandidate 调用 B 时已经完成改写。
// `l0aBudgetBytes` / `l0cBudgetBytes` 为零表示目标架构立方容量未知，因子
// 不受 L0 约束。
static std::optional<RowLiftCandidate>
matchQueryRows(triton::FuncOp fn, uint64_t l0aBudgetBytes,
               uint64_t l0cBudgetBytes,
               std::optional<int64_t> forcedRows = std::nullopt,
               bool allowSplit = true) {
  ModuleOp module = fn->getParentOfType<ModuleOp>();
  if (!module)
    return std::nullopt;

  auto grid = module->getAttrOfType<IntegerAttr>("hacc.grid_num_tiles");
  // Rule B does not stack on an already coalesced launch contract.  A shared
  // grid transform or a published coalesce factor means another rule owns this
  // function, so decline instead of rewriting it twice.
  if ((grid && (grid.getInt() <= 0 || grid.getInt() > INT32_MAX)) ||
      !llvm::hasSingleElement(fn.getBody()) || !fn.isPublic() ||
      llvm::count_if(module.getOps<triton::FuncOp>(),
                     [](triton::FuncOp entry) { return entry.isPublic(); }) !=
          1 ||
      module->hasAttr(kCoalesceFactorAttr) ||
      module->hasAttr(kProgramGridTransformsAttr))
    return std::nullopt;

  triton::GetProgramIdOp pid;
  scf::ForOp loop;
  triton::StoreOp output;
  unsigned pids = 0, loops = 0, dots = 0, stores = 0;
  unsigned blockPointers = 0, reduces = 0;
  int64_t maxDotRows = 1;
  int64_t maxTensorBytes = 1;
  // 原始 dot 的 L0A/L0C tile 最坏字节需求；重写按因子放大 lhs 与累加器。
  int64_t maxL0ABytes = 1;
  int64_t maxL0CBytes = 1;
  bool safe = true;
  DenseSet<Value> lifted;

  // One reject reason per declined operation keeps a "why did rule B not fire"
  // trace available without changing behaviour.
  auto reject = [&](Operation *bad, StringRef why) {
    LLVM_DEBUG(llvm::dbgs() << "query-row: reject " << why << " ("
                            << bad->getName() << ")\n");
    safe = false;
  };

  // Propagate row dependence while checking operations in definition order.
  fn.walk<WalkOrder::PreOrder>([&](Operation *op) {
    bool dependent = llvm::any_of(
        op->getOperands(), [&](Value value) { return lifted.contains(value); });
    if (dependent)
      lifted.insert(op->getResults().begin(), op->getResults().end());

    for (Type type : op->getResultTypes()) {
      if (auto tensor = dyn_cast<RankedTensorType>(type)) {
        Type element = tensor.getElementType();
        int64_t bytes = isa<triton::PointerType>(element)
                            ? 8
                            : (element.getIntOrFloatBitWidth() + 7) / 8;
        int64_t size;
        if (__builtin_mul_overflow(tensor.getNumElements(), bytes, &size))
          reject(op, "tensor size overflow");
        else
          maxTensorBytes = std::max(maxTensorBytes, size);
      }
    }

    if (auto p = dyn_cast<triton::GetProgramIdOp>(op)) {
      // Rule B lifts the query-row axis only.  Program ids on other axes stay
      // scalar and are shared by every lifted row, exactly as they are in
      // DotRowCoalescing, so they do not disqualify the function.
      if (p.getAxisAsInt() == 0) {
        pid = p;
        ++pids;
        lifted.insert(p.getResult());
      }
      return;
    }

    if (auto f = dyn_cast<scf::ForOp>(op)) {
      loop = f;
      ++loops;
      // Loop states are per-row even when their initial values are uniform.
      lifted.insert(f.getRegionIterArgs().begin(), f.getRegionIterArgs().end());
      lifted.insert(f.getResults().begin(), f.getResults().end());
      if (f->getBlock() != &fn.getBody().front())
        reject(f, "loop not in entry block");
      return;
    }

    if (auto makePtr = dyn_cast<triton::MakeTensorPtrOp>(op)) {
      ++blockPointers;
      // 描述符必须能展开成显式指针张量，否则重写阶段必然失败；规则 B 始终
      // 置位 tensorPointer 能力，所以这里传 true。
      if (!canExpandTensorPointer(makePtr, true))
        reject(makePtr, "unexpanded tensor pointer");
      return;
    }

    if (auto d = dyn_cast<triton::DotOp>(op)) {
      ++dots;
      auto a = dyn_cast<RankedTensorType>(d.getA().getType());
      auto b = dyn_cast<RankedTensorType>(d.getB().getType());
      if (!a || !b || a.getRank() != 2 || b.getRank() != 2 ||
          !(a.getElementType().isF16() || a.getElementType().isBF16()) ||
          (dependent &&
           (!lifted.contains(d.getA()) || lifted.contains(d.getB()))))
        reject(d, "dot operand or shape");
      else {
        maxDotRows = std::max(maxDotRows, a.getShape()[0]);
        maxL0ABytes = std::max(maxL0ABytes, tensorStorageBytes(a));
        if (auto acc = dyn_cast<RankedTensorType>(d.getC().getType()))
          maxL0CBytes = std::max(maxL0CBytes, tensorStorageBytes(acc));
        if (auto res = dyn_cast<RankedTensorType>(d.getD().getType()))
          maxL0CBytes = std::max(maxL0CBytes, tensorStorageBytes(res));
      }
      return;
    }

    if (auto l = dyn_cast<triton::LoadOp>(op)) {
      if (l.getIsVolatile()) {
        reject(l, "volatile load");
        return;
      }
      // 块指针与普通 load 都在这里放行：循环体内是否还按下标表读索引由规则 A
      // 负责消除，B 不再据此判断「A 是否已经跑过」。行相关的下标表读能否提升
      // 仍是重写路径的能力开关问题（features.tableRead 关闭），不在这里拦。
      // Both block-pointer and plain loads are accepted: whether the loop still
      // reads an index table is rule A's business, and B no longer infers "has A
      // run?" from it.  Lifting a row-dependent table read remains a rewrite-path
      // capability question (features.tableRead stays off), not a matcher rule.
      return;
    }

    if (auto s = dyn_cast<triton::StoreOp>(op)) {
      ++stores;
      if (op->getBlock() == &fn.getBody().front() && !output)
        output = s;
      return;
    }

    if (isa<triton::ReduceOp>(op)) {
      ++reduces;
      return;
    }

    // scf.if 只有在条件与查询行无关时才能被提升；逐行走不同分支的形态要
    // 在匹配阶段就拒绝，而不是等到重写才发现。
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      if (pid && !liftsRowInvariantIf(ifOp, pid.getResult()))
        reject(ifOp, "row-dependent scf.if");
      return;
    }

    if (isa<triton::FuncOp, triton::ReturnOp, scf::YieldOp,
            triton::ReduceReturnOp, triton::MakeRangeOp, triton::SplatOp,
            triton::ExpandDimsOp, triton::BroadcastOp, triton::AddPtrOp,
            triton::TransOp, triton::ReshapeOp, triton::AdvanceOp,
            triton::AssertOp>(op))
      return;
    StringRef ns = op->getName().getDialectNamespace();
    if ((ns != "arith" && ns != "math") || op->getNumRegions())
      reject(op, "unsupported op");
  });

  LLVM_DEBUG(llvm::dbgs() << "query-row: safe=" << safe << " pids=" << pids
                          << " loops=" << loops << " dots=" << dots
                          << " stores=" << stores
                          << " blockPtrs=" << blockPointers
                          << " reduces=" << reduces << "\n");

  // Only the query-row axis must be unique.  tt.reduce is counted for tracing
  // but is not required: the backward dq form carries a plain accumulator and
  // has no reduction, and rule B has to match both recurrence forms.  Block
  // pointers are likewise not required, because rule A replaces the K/V block
  // reads with dense tile loads; Q and O may keep a tensor pointer, and the
  // engine expands it when the capability is on.
  if (!safe || pids != 1 || loops != 1 || dots == 0 || stores == 0 || !output)
    return std::nullopt;
  if (loop.getNumResults() == 0 ||
      getConstantIntValue(loop.getLowerBound()) != std::optional<int64_t>(0))
    return std::nullopt;

  auto step = getConstantIntValue(loop.getStep());
  if (!step || *step <= 0)
    return std::nullopt;

  // A merge across unequal trip counts evaluates the bound chain and the body
  // for rows that stop early, so a query-row dependent bound needs a
  // monotonicity proof over the whole pid range (the last row then supplies the
  // group's trip count) plus a speculation check.  The dependence test looks
  // through regions: an scf.if result comes from its branch values and a
  // tt.reduce result from its sources, so neither may be mistaken for an
  // invariant bound.  Only monotonicity is required here: the last row's bound
  // is re-evaluated at run time, so the proof does not have to produce a
  // numeric range (an i32 `pid + 64` chain under a runtime grid is accepted).
  // 这里只要求单调性：末行上界在运行期用原表达式重新求值，证明器不必给出
  // 具体数值范围。
  if (!isPidInvariant(loop.getUpperBound(), pid.getResult())) {
    if (!isMonotoneInPid(loop.getUpperBound(), pid.getResult(),
                         grid ? grid.getInt() : INT32_MAX)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "query-row: reject non-monotone loop bound\n");
      return std::nullopt;
    }
    for (Operation &op : loop.getBody()->without_terminator()) {
      StringRef ns = op.getName().getDialectNamespace();
      if ((ns == "arith" || ns == "math") && !isSpeculatable(&op)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "query-row: reject speculated " << op.getName() << "\n");
        return std::nullopt;
      }
    }
  }

  RowLiftCandidate candidate;
  candidate.function = fn;
  candidate.anchor = pid.getOperation();
  candidate.runtimeGrid = !grid;
  candidate.grid = grid ? grid.getInt() : 0;
  // Only capabilities the shared engine already implements.  Rule B runs after
  // rule A, so the loop is already a sequential scan: the per-row active masks
  // keep a finished row off memory and the masked write-back is still needed.
  // The index-table capability stays off because lifting a row-dependent table
  // read is not implemented yet; the matcher no longer rejects such loops, so a
  // candidate that needs it fails in the rewrite path instead.
  candidate.features.reduce = true;
  candidate.features.loopInvariantIf = true;
  candidate.features.tensorPointer = true;
  candidate.features.maskedLoad = true;
  candidate.features.maskedStore = true;
  // 值维（PV 的 N 维）超过「M × 64 列」时按手写 kernel 的做法拆成两半，
  // 否则 V 补到 128 的形态会让 [R,G,128] 的 f32 累加器撑爆 UB。拆半是结构性
  // 的：V 的访存、PV dot、元素链与 O 的写回各复制成两条 64 宽链，与手写 kernel
  // 的 VB=min(BV,64) 一致，不引入值级 reshape/trans/split，所以它是首选档。
  // A value dimension (the N axis of PV) wider than "M × 64 columns" is split in
  // half like the hand-written kernel does; otherwise a V padded to 128 makes the
  // [R,G,128] f32 accumulator overflow UB.  The split is structural: the V access,
  // the PV dot, the elementwise chain and the O write-back each become two
  // 64-wide chains, matching the hand-written kernel's VB=min(BV,64), with no
  // value-level reshape/trans/split, so it is the primary tier.
  candidate.features.splitWideAccumulator = allowSplit;

  // The row count comes from the shared policy, so rule B carries no private
  // profitability model.  Rule B keeps its M dimension uncapped (targetDotRows
  // stays zero) and lets the split capability double the byte budget.  The
  // L0A/L0C demands are a correctness constraint: the rewrite widens the lhs
  // and accumulator tiles by the factor, and a zero budget disables the check.
  RowCountPolicy policy;
  policy.splitWideAccumulator = candidate.features.splitWideAccumulator;
  policy.maxL0ABytes = maxL0ABytes;
  policy.maxL0CBytes = maxL0CBytes;
  policy.l0ABudgetBytes = l0aBudgetBytes;
  policy.l0CBudgetBytes = l0cBudgetBytes;
  candidate.rowsPerProgram = selectRowsPerProgram(
      grid ? grid.getInt() : 0, maxDotRows, maxTensorBytes, policy);
  // 预算不够时上层会按 2 的幂下调重试；这里把重试因子钉在候选上，后续所有
  // 依赖行数的判据（L0 需求、单值预算、拆分判据）都用这同一个因子。
  // The caller retries with smaller powers of two when the budget does not fit;
  // pin that factor on the candidate so every row-count dependent judgement (L0
  // demand, per-value budget, split predicate) uses the same value.
  if (forcedRows)
    candidate.rowsPerProgram = std::min(candidate.rowsPerProgram, *forcedRows);
  if (candidate.rowsPerProgram <= 1)
    return std::nullopt;

  // 行数决策的全部输入留在候选里：片上 UB 预算超限时要按 2 的幂下调因子重选。
  // Keep every row-count input on the candidate so a candidate can be retried
  // with a smaller factor when the on-chip UB budget overflows.
  candidate.rowPolicy = policy;
  candidate.maxDotRows = maxDotRows;
  candidate.maxTensorBytes = maxTensorBytes;
  // 单值预算按「提升后的 M × 最多 64 个值列 × 4 字节」设定：值宽不超过 64 列
  // 的原样保留，超过的（V 补到 128）落进拆半路径，与手写 kernel 的 VB=64 一致。
  // The per-value budget is "lifted M × at most 64 value columns × 4 bytes": a
  // value no wider than 64 columns stays whole, and a wider one (V padded to
  // 128) takes the structural split path, matching the hand-written kernel's
  // VB=64.
  candidate.splitValueBudgetBytes =
      allowSplit ? candidate.maxDotRows * candidate.rowsPerProgram * 64 * 4
                 : std::numeric_limits<int64_t>::max();

  // 累加器拆分的判据原本只在重写路径里执行，这里在匹配阶段先判一遍：注定
  // 失败的形态不再进入沙箱克隆与 verify。重写路径仍会调用同一组判据，两处
  // 共用一份实现，不会漂移。表读判据仍然不做：顺序扫描由规则 A 保证。
  // The split predicate used to run only on the rewrite path.  Evaluating it
  // here keeps a doomed shape out of the sandbox clone and verify; the rewrite
  // path calls the same predicate.  The table-read predicate is still absent:
  // rule A guarantees the sequential scan.
  if (!canSplitWideAccumulators(loop, candidate.rowsPerProgram,
                                candidate.features.splitWideAccumulator,
                                candidate.splitValueBudgetBytes)) {
    LLVM_DEBUG(llvm::dbgs() << "query-row: reject rewrite pre-check\n");
    return std::nullopt;
  }

  return candidate;
}

// 提升后的函数体峰值活值是否放不进目标核的 UB 预算。UB 容量未知（零）时
// 不施加约束，沿用既有行为；估计不可得时同样不拒绝。
// True when the lifted body's peak live bytes do not fit the target core's UB
// budget.  An unknown capacity (zero) leaves the check disabled, and an
// unavailable estimate is not a reason to reject.
static bool exceedsUBBudget(triton::FuncOp function,
                            const ResourceSnapshot &resources) {
  const std::optional<uint64_t> safe = resources.getSafeUBBudget();
  if (!safe)
    return false;
  LiveByteEstimate estimate = estimatePeakLiveBytes(function.getOperation());
  if (!estimate.known)
    return false;
  LLVM_DEBUG(llvm::dbgs() << "query-row: peak live " << estimate.peakLiveBytes
                          << " bytes, UB budget " << *safe << " bytes\n");
  return estimate.peakLiveBytes > *safe;
}

// 规则 B 的唯一重写实现：在传入的模块上就地匹配、重建并验证，最后发布启动
// 契约。静态网格发布 coalesce 三属性；运行时网格给函数补隐藏 extent 参数并
// 发布 hacc.program_grid_transforms。调度入口与规则 A 的调用入口都走这里，
// 因此两条入口不会产生不同结果。调用方必须传私有沙箱：失败时模块可能已被
// 部分改写，不能是线上 IR。
// The single rewrite implementation for rule B.  It matches, rebuilds and
// verifies in place on the given module, then publishes the launch contract:
// static grids get the three coalesce attributes, runtime grids get the hidden
// extent argument plus hacc.program_grid_transforms.  Both the scheduler entry
// and rule A's entry call this, so the two paths cannot diverge.  The caller
// must pass a private sandbox, because a failure may leave the module partially
// rewritten.
static LogicalResult rewriteQueryRowBatchInPlace(
    ModuleOp module, triton::FuncOp function, const ResourceSnapshot &resources,
    std::optional<int64_t> forcedRows = std::nullopt, bool allowSplit = true) {
  // Zero budgets (unknown cube capacities) leave the factor unconstrained.
  uint64_t l0aBudgetBytes = 0;
  uint64_t l0cBudgetBytes = 0;
  if (resources.hasL0Budgets()) {
    l0aBudgetBytes = resources.getSafeL0ABudget().value_or(0);
    l0cBudgetBytes = resources.getSafeL0CBudget().value_or(0);
  }

  auto candidate =
      matchQueryRows(function, l0aBudgetBytes, l0cBudgetBytes, forcedRows,
                     allowSplit);
  if (!candidate)
    return failure();

  IRRewriter rw(module.getContext());
  // Diagnostics from a failed rewrite stay inside the sandbox: they are logged
  // for tracing and never reach the user's compilation.
  ScopedDiagnosticHandler diagnostics(
      module.getContext(), [](Diagnostic &diag) {
        LLVM_DEBUG(llvm::dbgs()
                   << "query-row: skipped candidate: " << diag << "\n");
        return success();
      });

  if (candidate->runtimeGrid &&
      failed(addProgramGridHiddenExtentArguments(function))) {
    LLVM_DEBUG(llvm::dbgs() << "query-row: hidden extent args failed\n");
    return failure();
  }

  RowLiftBuilder builder(rw, *candidate);
  if (!builder.run(function)) {
    LLVM_DEBUG(llvm::dbgs() << "query-row: builder.run failed\n");
    return failure();
  }
  if (failed(verify(module))) {
    LLVM_DEBUG(llvm::dbgs() << "query-row: verify failed\n");
    return failure();
  }

  if (candidate->runtimeGrid) {
    ProgramGridTransformContract contract;
    contract.transforms.push_back(
        {0, 0, candidate->rowsPerProgram, false, false});
    return setProgramGridTransformContract(module, contract);
  }

  module->setAttr(kCoalesceFactorAttr,
                  rw.getI32IntegerAttr(candidate->rowsPerProgram));
  module->setAttr(kCoalesceAxisAttr, rw.getI32IntegerAttr(0));
  module->setAttr(kCoalesceGridCeilDivAttr, rw.getI32IntegerAttr(1));
  return success();
}

class QueryRowBatchingPlan final : public RewritePlan {
  RowLiftCandidate candidate;
  unsigned epoch;
  OwningOpRef<ModuleOp> prepared;
  // 该计划采用哪一档：true 为「拆半」首选档，false 为整宽兜底档。重新匹配时
  // 必须用同一档，否则行数判据会得出不同的因子。
  // Which tier this plan was built with: true is the preferred split tier,
  // false the whole-width fallback.  Revalidation must reuse the same tier,
  // otherwise the row-count judgement would pick a different factor.
  bool allowSplit;
  // Safe L0 budgets captured at plan creation; zero means the target's cube
  // capacities are unknown and the factor stays unconstrained.
  uint64_t l0aBudgetBytes = 0;
  uint64_t l0cBudgetBytes = 0;

  // 保存候选、创建 epoch 以及已私有验证的沙箱模块。
  QueryRowBatchingPlan(RowLiftCandidate candidate, unsigned epoch,
                       OwningOpRef<ModuleOp> prepared, uint64_t l0aBudgetBytes,
                       uint64_t l0cBudgetBytes, bool allowSplit)
      : candidate(candidate), epoch(epoch), prepared(std::move(prepared)),
        allowSplit(allowSplit), l0aBudgetBytes(l0aBudgetBytes),
        l0cBudgetBytes(l0cBudgetBytes) {}

public:
  // 返回规则 B 的规则 ID。
  GraphOptimizationRuleId getRuleId() const override {
    return GraphOptimizationRuleId::QueryRowBatching;
  }

  // 规则 B 的收益固定为 1，沿用行合并规则的调度语义。
  unsigned getBenefit() const override { return 1; }

  // 锚点为候选函数，保证计划在同一函数上重新验证。
  Operation *getAnchor() const override { return candidate.function; }
  // 返回创建该计划的 epoch，供调度器判断是否已过期。
  unsigned getCreationEpoch() const override { return epoch; }

  // 重新匹配函数，确认锚点、行数与运行期网格没有变化。
  LogicalResult revalidate(GraphOptimizationContext &context) const override {
    if (context.getFunction() != candidate.function)
      return failure();

    auto current = matchQueryRows(candidate.function, l0aBudgetBytes,
                                  l0cBudgetBytes, std::nullopt, allowSplit);
    return success(current && current->anchor == candidate.anchor &&
                   current->rowsPerProgram == candidate.rowsPerProgram &&
                   current->runtimeGrid == candidate.runtimeGrid);
  }

  // 档位由调用方传入：拆半首选档把过宽的值维复制成两条半宽链，整宽兜底档
  // 只在结构拆半不可表达时使用。这里只按 2 的幂下调因子重试。
  // The tier is chosen by the caller: the preferred split tier duplicates an
  // over-wide value dimension into two half-width chains, and the whole-width
  // fallback is used only when the structural split cannot express the shape.
  // This function only walks the factor down by powers of two.
  static std::unique_ptr<QueryRowBatchingPlan> create(
      RowLiftCandidate candidate, unsigned epoch,
      const ResourceSnapshot &resources, bool allowSplit) {
    uint64_t l0aBudgetBytes = 0;
    uint64_t l0cBudgetBytes = 0;
    if (resources.hasL0Budgets()) {
      l0aBudgetBytes = resources.getSafeL0ABudget().value_or(0);
      l0cBudgetBytes = resources.getSafeL0CBudget().value_or(0);
    }

    ModuleOp module = candidate.function->getParentOfType<ModuleOp>();
    RowCountPolicy policy = candidate.rowPolicy;
    int64_t factor = candidate.rowsPerProgram;
    while (factor >= 2) {
      candidate.rowsPerProgram = factor;

      OwningOpRef<ModuleOp> sandbox(
          ModuleOp::create(candidate.function.getLoc()));
      sandbox->getOperation()->setAttrs(module->getAttrs());
      sandbox->getBody()->push_back(candidate.function->clone());
      auto cloned = cast<triton::FuncOp>(&sandbox->getBody()->front());
      LLVM_DEBUG(llvm::dbgs() << "query-row: trying factor " << factor
                              << (allowSplit ? " (split)" : " (no split)")
                              << "\n");
      if (succeeded(rewriteQueryRowBatchInPlace(*sandbox, cloned, resources,
                                                factor, allowSplit)) &&
          !exceedsUBBudget(cloned, resources))
        return std::unique_ptr<QueryRowBatchingPlan>(
            new QueryRowBatchingPlan(candidate, epoch, std::move(sandbox),
                                     l0aBudgetBytes, l0cBudgetBytes,
                                     allowSplit));

      if (factor == 2)
        break;
      policy.maxRows = factor / 2;
      factor = selectRowsPerProgram(candidate.grid, candidate.maxDotRows,
                                    candidate.maxTensorBytes, policy);
    }
    LLVM_DEBUG(llvm::dbgs() << "query-row: no factor fits the UB budget\n");
    return nullptr;
  }

  // 把沙箱里已验证的函数体与启动契约一次性提交回原函数。
  LogicalResult apply(IRRewriter &rw) override {
    (void)rw;
    // Commit the verified body and, for runtime grids, its hidden arguments.
    ModuleOp module = candidate.function->getParentOfType<ModuleOp>();
    auto replacement = cast<triton::FuncOp>(&prepared->getBody()->front());
    if (candidate.runtimeGrid) {
      if (failed(commitProgramGridFunctionFromSandbox(candidate.function,
                                                      replacement)))
        return failure();
      module->setAttr(
          kProgramGridTransformsAttr,
          prepared->getOperation()->getAttr(kProgramGridTransformsAttr));
      return success();
    }

    // Static grids: the coalesce attributes were published on the sandbox by
    // the shared rewrite, so commit them together with the body.
    candidate.function.getBody().takeBody(replacement.getBody());
    ModuleOp preparedModule = *prepared;
    module->setAttr(kCoalesceFactorAttr,
                    preparedModule->getAttr(kCoalesceFactorAttr));
    module->setAttr(kCoalesceAxisAttr,
                    preparedModule->getAttr(kCoalesceAxisAttr));
    module->setAttr(kCoalesceGridCeilDivAttr,
                    preparedModule->getAttr(kCoalesceGridCeilDivAttr));
    return success();
  }
};

class QueryRowBatchingRule final : public GraphOptimizationRule {
public:
  // 返回规则 B 的规则 ID。
  GraphOptimizationRuleId getId() const override {
    return GraphOptimizationRuleId::QueryRowBatching;
  }

  // 规则 B 依赖 program 轴依赖分析与资源代价模型（L0 预算）。
  AnalysisRequirement getAnalysisRequirements() const override {
    return AnalysisRequirement::ProgramAxisDependence |
           AnalysisRequirement::ResourceCost;
  }

  // 为当前函数构造规则 B 的候选计划。
  LogicalResult findCandidates(
      GraphOptimizationContext &context,
      SmallVectorImpl<std::unique_ptr<RewritePlan>> &plans) override {
    if (auto plan = createQueryRowBatchingPlan(
            context.getFunction(), context.getEpoch(),
            context.getResourceCostAnalysis().getResourceSnapshot()))
      plans.push_back(std::move(plan));
    return success();
  }
};

} // namespace

LogicalResult
// 规则 B 面向规则 A 的唯一接口：在给定沙箱模块上直接完成匹配、重写与 verify，
// 成功即代表 B 的行提升已经真实落地在沙箱里，A 可以连同自己的顺序化一起提交。
cfg::materializeQueryRowBatchingCandidate(ModuleOp module,
                                          triton::FuncOp function,
                                          const ResourceSnapshot &resources) {
  if (!module || !function ||
      function->getParentOfType<ModuleOp>() != module)
    return failure();

  // Rule B is supported only on Ascend 950, like the dot-row rule.
  if (!hacc::utils::isAscend950(module))
    return failure();

  // 与调度入口共用同一份「沙箱重写 + UB 预算重试」实现，避免两条入口出现
  // 不同的因子或不同的可行性判断。
  // Share the sandbox-rewrite-and-retry implementation with the scheduler entry,
  // so both entries pick the same factor and the same feasibility verdict.
  auto plan = createQueryRowBatchingPlan(function, /*epoch=*/0, resources);
  if (!plan)
    return failure();
  IRRewriter rw(module.getContext());
  return plan->apply(rw);
}

std::unique_ptr<RewritePlan>
// 规则 B 的调度入口：按资源快照提取 L0 预算，在沙箱中调用同一份重写实现并
// verify 后才交付计划。
// Scheduler entry: it extracts the L0 budgets, runs the shared rewrite on a
// sandbox and only hands out a plan once that sandbox verifies.
cfg::createQueryRowBatchingPlan(triton::FuncOp function, unsigned epoch,
                                const ResourceSnapshot &resources) {
  // Rule B is supported only on Ascend 950, like the dot-row rule.
  auto module = function->getParentOfType<ModuleOp>();
  if (!module || !hacc::utils::isAscend950(module))
    return nullptr;

  // Zero budgets (unknown cube capacities) leave the factor unconstrained.
  uint64_t l0aBudgetBytes = 0;
  uint64_t l0cBudgetBytes = 0;
  if (resources.hasL0Budgets()) {
    l0aBudgetBytes = resources.getSafeL0ABudget().value_or(0);
    l0cBudgetBytes = resources.getSafeL0CBudget().value_or(0);
  }

  // 档位从首选到兜底：先按手写 kernel 的做法把过宽的值维拆成两条半宽链，
  // 只有拆半被匹配阶段拒绝（形态不支持结构拆半）时才退回整宽档。两档用同一
  // 份匹配与重写实现，只是拆分能力开关不同。
  // Tiers run from the preferred shape to the fallback: the value dimension is
  // first halved the way the hand-written kernel does, and only a shape the
  // structural split cannot express falls back to the whole-width tier.  Both
  // tiers share one matcher and one rewrite, differing only in the split
  // capability.
  for (bool allowSplit : {true, false}) {
    auto tierCandidate = matchQueryRows(function, l0aBudgetBytes,
                                        l0cBudgetBytes, std::nullopt, allowSplit);
    if (!tierCandidate)
      continue;
    if (auto plan = QueryRowBatchingPlan::create(*tierCandidate, epoch,
                                                 resources, allowSplit))
      return plan;
  }
  return nullptr;
}

std::unique_ptr<GraphOptimizationRule>
// 创建规则 B 的规则对象。
cfg::createQueryRowBatchingRule() {
  return std::make_unique<QueryRowBatchingRule>();
}
