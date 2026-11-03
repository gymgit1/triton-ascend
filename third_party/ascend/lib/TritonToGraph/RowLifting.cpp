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

#include "TritonToGraph/RowLifting.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#include <algorithm>

#define DEBUG_TYPE "graph-optimize"

using namespace mlir;
using namespace triton;

namespace mlir {
namespace triton {
namespace cfg {

// 行提升公共引擎（规则 A「顺序扫描」与规则 B「下标表合并」共用）。
//
// 本文件把「一个 program 只处理一行查询」的函数体重建为「一个 program
// 处理 R 行查询」：所有与查询行相关的值在类型最前面插入行维 [R]，循环用
// 末行的上界作为组内共享 trip count，组内各行的差异由逐行有效掩码表达。
// 文件内函数与既定需求条目的对应关系：
//   1. 公共组件：本文件即从 DotRowCoalescing 抽出的行提升引擎，两条规则
//      各自维护 matcher，重建逻辑只保留这一份。
//   2. tt.reduce 与条件与行号无关的 scf.if：rewriteReduce / rewriteIfOp。
//   3. Q、O 的 tensor pointer 展开成指针张量：rewriteTensorPointer，以及
//      描述符 load/store 的重建。
//   4. 下标表改成每行一份地读：planTableRead 在循环外做一次带掩码的 [R,S]
//      整表读，循环内 extractTableColumn 只提取 [R] 列切片。
//   5. O 与 lse 的带掩码写回：maskedStore 能力 + rewriteDescriptorStore，
//      循环内保持压平形状，不出现逐轮 reshape。
//   6. 输出累加器过宽时沿宽度拆成两半：planWideAccumulatorSplits 与四个
//      rewriteSplit* 函数，能力默认关闭。
//   7. 「入口生成合并版与回退版」由规则层（Rule / ProgramGridTransform）
//      负责，不属于本文件。
//
// 约定：RowLiftFeatures 中的能力开关全部默认关闭，未置位的能力走原有的
// 失败关闭路径；DotRow 规则不置位任何新开关，因此其输出与基线指令序列
// 逐字节一致。

// 循环外下标表读的列宽上限。整张表每个 program 只物化一次，列数按上界
// 取 2 的幂，这里限制它不应超过 1024；行数已由单值字节预算约束。
// A hoisted index table is materialised once per program, so cap its column
// extent; the row count is already bounded by the per-value byte budget.
constexpr int64_t kMaxTableColumns = 1024;

// 判断 v 是否等于 pid 或由 pid 的顶层计算链产生；不深入 region。
bool dependsOnValue(Value v, Value pid) {
  if (v == pid)
    return true;

  Operation *op = v.getDefiningOp();
  return op && llvm::any_of(op->getOperands(),
                            [&](Value a) { return dependsOnValue(a, pid); });
}

// A block argument other than `pid` cannot introduce a dependence by itself, so
// only operands and nested regions have to be inspected.  The regions matter for
// the two shapes whose result is defined inside a body rather than by an
// operand: an scf.if yields its branch values, and a tt.reduce yields a
// combination of its sources.
// 递归检查 value 的操作数与所有嵌套 region 是否引入 pid 依赖；visited 去重。
static bool hasPidDependency(Value value, Value pid,
                             llvm::DenseSet<Value> &visited) {
  if (value == pid)
    return true;
  if (!visited.insert(value).second)
    return false;

  Operation *op = value.getDefiningOp();
  if (!op)
    return false;

  for (Value operand : op->getOperands())
    if (hasPidDependency(operand, pid, visited))
      return true;

  bool nested = false;
  op->walk([&](Operation *inner) {
    if (nested)
      return;
    for (Value operand : inner->getOperands())
      if (hasPidDependency(operand, pid, visited)) {
        nested = true;
        return;
      }
  });
  return nested;
}

// 判断 value 是否完全与 pid 无关，判定会穿透 scf.if / tt.reduce 的 region。
bool isPidInvariant(Value value, Value pid) {
  llvm::DenseSet<Value> visited;
  return !hasPidDependency(value, pid, visited);
}

// 读取整数常量；不是常量时返回 nullopt。
static std::optional<int64_t> getConstantIntValue(Value v) {
  APInt value;
  if (matchPattern(v, m_ConstantInt(&value)))
    return value.getSExtValue();

  return std::nullopt;
}

// 上界证明的三态结果：
//   None      —— 无法证明单调，调用方必须拒绝行提升；
//   Monotone  —— 已证明关于 pid 单调非降，但没有可用的数值范围。幅度未知的
//                项（例如对已加载 tile 做 tt.reduce 得到的标量）与范围放不进
//                表达式位宽，都归入此态；
//   Bounded   —— 给出了具体数值范围，末行取值可作循环上界；下标表提升要按
//                hi 定表宽，也只接受此态。
// 行提升只需要「末行取值是组内最大」这一单调性，末行上界是在运行期用原表达式
// 重新求值得到的，不依赖证明器算出的数值，所以 Monotone 已经够用。
// A pid-independent term only shifts the chain by a constant, which keeps the
// monotonicity but hides its magnitude (the scalar result of a tt.reduce over a
// loaded tile, for instance).  That case and a range that does not fit the
// expression's bit width are both reported as `Monotone`: row lifting only needs
// the last row's value to be the group maximum, which is re-evaluated from the
// original expression at run time.  Only index-table lifting needs `Bounded`,
// because it sizes the per-row table from the proven `hi`.
enum class BoundKind { None, Monotone, Bounded };

struct BoundProof {
  BoundKind kind = BoundKind::None;
  IntegerBounds bounds{0, 0};

  static BoundProof none() { return {}; }

  static BoundProof monotone() { return {BoundKind::Monotone, {0, 0}}; }

  static BoundProof bounded(IntegerBounds b) {
    return {BoundKind::Bounded, b};
  }

  bool ok() const { return kind != BoundKind::None; }
};

// 前向声明：证明 value 关于 pid 单调非降，返回三态证明结果。
static BoundProof proveBound(Value value, Value pid, int64_t grid);

// Narrow a bounded result to the bit width of the expression it came from.
// A range that does not fit the expression's width cannot be used as a numeric
// bound, but the chain is still syntactically monotone, so the result degrades
// to `Monotone` instead of failing: only index-table lifting needs the range.
// 把有界证明收窄到表达式位宽；范围放不进该位宽时退化为「单调但无范围」，
// 供行提升使用；需要数值范围的调用方（下标表提升）自行要求 Bounded。
static BoundProof checkWidth(Value value, BoundProof proof) {
  if (proof.kind != BoundKind::Bounded)
    return proof;

  auto integer = dyn_cast<IntegerType>(value.getType());
  if (!integer || integer.getWidth() > 64 ||
      !APInt(64, proof.bounds.lo, true).isSignedIntN(integer.getWidth()) ||
      !APInt(64, proof.bounds.hi, true).isSignedIntN(integer.getWidth()))
    return BoundProof::monotone();
  return proof;
}

// 证明 value 关于 pid 单调非降：区分无证明、仅有单调性、pid 无关常量与
// 有数值范围四种情况。
static BoundProof proveBound(Value value, Value pid, int64_t grid) {
  // 基例一：pid 本身，范围就是 [0, grid - 1]（grid 非正表示运行期范围）。
  if (value == pid)
    return BoundProof::bounded({0, grid - 1});

  // 基例二：整数常量，退化为单点范围。
  if (auto constant = getConstantIntValue(value))
    return BoundProof::bounded({*constant, *constant});

  // 有符号扩展不改变单调性，直接穿透到源操作数。
  if (auto ext = value.getDefiningOp<arith::ExtSIOp>())
    return proveBound(ext.getIn(), pid, grid);

  // scf.if：条件与 pid 无关时所有提升行都走同一分支，因此只需分别证明两个
  // 分支的 yield 值；两个分支都有范围时取并集，任一分支无范围时整条链仍
  // 单调但没有可用范围。
  // Every lifted row takes the same branch when the condition is pid-invariant,
  // so each branch only has to be proven monotone on its own.  Two bounded
  // branches yield the union of their ranges; as soon as one branch has no
  // proven range the selection is monotone but unbounded.
  if (auto ifOp = value.getDefiningOp<scf::IfOp>()) {
    if (!isPidInvariant(ifOp.getCondition(), pid))
      return BoundProof::none();
    if (ifOp.getElseRegion().empty())
      return BoundProof::none();

    unsigned index = cast<OpResult>(value).getResultNumber();
    SmallVector<Value, 2> yields;
    for (Region *region : {&ifOp.getThenRegion(), &ifOp.getElseRegion()}) {
      auto yield = dyn_cast<scf::YieldOp>(region->front().getTerminator());
      if (!yield || yield.getNumOperands() <= index)
        return BoundProof::none();
      yields.push_back(yield.getOperand(index));
    }

    auto then = proveBound(yields[0], pid, grid);
    auto otherwise = proveBound(yields[1], pid, grid);
    if (!then.ok() || !otherwise.ok())
      return BoundProof::none();
    if (then.kind == BoundKind::Bounded && otherwise.kind == BoundKind::Bounded)
      return BoundProof::bounded(
          {std::min(then.bounds.lo, otherwise.bounds.lo),
           std::max(then.bounds.hi, otherwise.bounds.hi)});
    return BoundProof::monotone();
  }

  // 二元整数运算：先递归证明两个操作数，再按运算类型推导结果范围。
  //   add —— 两个有范围的操作数逐项相加；混入与 pid 无关的项时只保留
  //           「单调但幅度未知」，因为常量平移不改变单调性；
  //   mul —— 只有两个操作数都有范围且非负时才可推导，否则乘法可能放大
  //           甚至翻转整条链；
  //   div —— 除数必须是已知正数常量，被除数的范围按常量整除收窄。
  // 任一中间运算可能回绕时退化为「单调但无范围」（checkWidth 再核对结果位宽），
  // 因为行提升的末行上界是运行期重新求值，不需要证明器给出的数值。
  Operation *op = value.getDefiningOp();
  if (op && op->getNumOperands() == 2) {
    auto lhs = proveBound(op->getOperand(0), pid, grid);
    auto rhs = proveBound(op->getOperand(1), pid, grid);
    if (lhs.ok() && rhs.ok()) {
      IntegerBounds result;
      if (isa<arith::AddIOp>(op)) {
        // 常量平移保持单调性但会隐藏和的幅度，因此只能作为自身范围使用。
        // A constant shift preserves monotonicity but hides the magnitude of
        // the sum, so the result is only usable as a bound of its own.
        if (lhs.kind != BoundKind::Bounded ||
            rhs.kind != BoundKind::Bounded)
          return BoundProof::monotone();
        if (__builtin_add_overflow(lhs.bounds.lo, rhs.bounds.lo, &result.lo) ||
            __builtin_add_overflow(lhs.bounds.hi, rhs.bounds.hi, &result.hi))
          return BoundProof::monotone();
      } else if (isa<arith::MulIOp>(op)) {
        // 未知因子可能放大或翻转整条链，含幅度未知项的乘积一律不证明。
        // An unknown factor could flip or amplify the chain, so a product with
        // an unbounded term stays unproven.
        if (lhs.kind != BoundKind::Bounded ||
            rhs.kind != BoundKind::Bounded)
          return BoundProof::none();
        if (lhs.bounds.lo < 0 || rhs.bounds.lo < 0 ||
            __builtin_mul_overflow(lhs.bounds.lo, rhs.bounds.lo, &result.lo) ||
            __builtin_mul_overflow(lhs.bounds.hi, rhs.bounds.hi, &result.hi))
          return BoundProof::none();
      } else if (isa<arith::SubIOp>(op)) {
        // 被减数随 pid 单调、减数与 pid 无关时，差值保持同一方向；减数依赖
        // pid 则可能翻转整条链。幅度未知，因此只给出单调性。
        // A monotone numerator minus a pid-invariant term keeps its direction;
        // a pid-dependent subtrahend can invert it.  The magnitude is unknown,
        // so only monotonicity is reported.
        if (!isPidInvariant(op->getOperand(1), pid))
          return BoundProof::none();
        return BoundProof::monotone();
      } else if (isa<arith::DivSIOp>(op) && rhs.kind == BoundKind::Bounded &&
                 rhs.bounds.lo == rhs.bounds.hi && rhs.bounds.lo > 0) {
        // 除以正常量保持单调性，但被除数的幅度未知时无法给出范围。
        // Dividing by a positive constant preserves monotonicity, but the
        // quotient has no usable range when the numerator is unbounded.
        if (lhs.kind != BoundKind::Bounded)
          return BoundProof::monotone();
        result = {lhs.bounds.lo / rhs.bounds.lo, lhs.bounds.hi / rhs.bounds.lo};
      } else {
        // 其余算术（例如 sub、rem）不再细分，统一放弃证明。
        return BoundProof::none();
      }
      return checkWidth(value, BoundProof::bounded(result));
    }
  }

  // 兜底：算术推导失败但值完全与 pid 无关时仍可接受——各行取同一常量，
  // 末行副本天然是最大值；tt.reduce 这类不变量源的标量结果由此通过。
  // A value the arithmetic prover cannot bound is still acceptable when it does
  // not depend on the query row: it never changes as the row advances, so the
  // last row's copy is trivially the maximum.  This is how a bound produced by a
  // tt.reduce of invariant sources is accepted.
  if (isPidInvariant(value, pid))
    return BoundProof::monotone();
  return BoundProof::none();
}

// 只接受能给出具体数值范围的上界证明，返回下界与上界。
std::optional<IntegerBounds> inferMonotonicBounds(Value value, Value pid,
                                                  int64_t grid) {
  BoundProof proof = proveBound(value, pid, grid);
  if (proof.kind != BoundKind::Bounded)
    return std::nullopt;
  return proof.bounds;
}

// 上界证明的布尔查询：只要能证明单调（含 pid 无关常量）即为 true。
bool isMonotoneInPid(Value value, Value pid, int64_t grid) {
  return proveBound(value, pid, grid).ok();
}

// True when `root` is computed from `target`, looking through the regions of
// every operation on the way.  A bound selected by an scf.if takes its value
// from a branch rather than from an operand, so the shallow test would miss the
// arithmetic that has to be re-evaluated at the last row.
// 穿透所有嵌套 region 判断 root 是否依赖 target；visited 防止重复访问。
static bool dependsOnValueThroughRegions(Value root, Value target,
                                         llvm::DenseSet<Value> &visited) {
  if (root == target)
    return true;
  if (!visited.insert(root).second)
    return false;

  Operation *op = root.getDefiningOp();
  if (!op)
    return false;

  bool found = false;
  op->walk([&](Operation *nested) {
    if (found)
      return;
    for (Value operand : nested->getOperands())
      if (dependsOnValueThroughRegions(operand, target, visited)) {
        found = true;
        return;
      }
  });
  return found;
}

// 上面的便捷重载，内部创建 visited 集合。
static bool dependsOnValueThroughRegions(Value root, Value target) {
  llvm::DenseSet<Value> visited;
  return dependsOnValueThroughRegions(root, target, visited);
}

// The part of a loop bound that can be re-evaluated at the last row: integer
// arithmetic, plus an scf.if whose branches hold nothing but that arithmetic.  A
// region that could have side effects stops the last-row copy, so the bound is
// evaluated once instead of being duplicated into a speculative path.
// 判断 op 是否属于可安全复制到末行求值的纯标量上界切片（arith 或纯算术 scf.if）。
static bool isScalarBoundSlice(Operation *op) {
  StringRef ns = op->getName().getDialectNamespace();
  if (ns == "arith")
    return true;
  if (!isa<scf::IfOp>(op))
    return false;

  bool pure = true;
  op->walk([&](Operation *nested) {
    if (nested == op || !pure)
      return;
    // Branch terminators only forward the yielded value, so they are part of
    // the slice; anything else in a branch has to be plain arithmetic.
    if (isa<scf::YieldOp>(nested))
      return;
    StringRef inner = nested->getName().getDialectNamespace();
    if (inner != "arith" && inner != "math")
      pure = false;
  });
  return pure;
}

// `tt.ptr<tensor<...>>` is a block-pointer descriptor, not a pointer tensor.
// 判断类型是否为块指针描述符 tt.ptr<tensor<...>>。
static bool isTensorPointer(Type type) {
  auto pointer = dyn_cast<triton::PointerType>(type);
  return pointer && isa<RankedTensorType>(pointer.getPointeeType());
}

// 在 2..maxRows 的 2 的幂中选每个 program 处理的行数，返回 1 表示不提升。
// 选择流程：预算换算 -> 从大到小枚举 2 的幂 -> 逐项检查网格整除、dot 行数
// 上限与单值字节预算，第一个通过的因子即结果。
int64_t selectRowsPerProgram(int64_t grid, int64_t maxDotRows,
                             int64_t maxTensorBytes,
                             const RowCountPolicy &policy) {
  // 拆分路径会物化两个半宽值，因此单值预算可以翻倍；若不翻倍，拆分路径
  // 永远不可能被收益模型选中。
  // The split path materialises two half-width values, so it may spend twice
  // the per-value budget; without the doubling a split could never be chosen.
  const int64_t budget = policy.splitWideAccumulator
                             ? 2 * policy.maxLiftedValueBytes
                             : policy.maxLiftedValueBytes;

  // 从最大因子开始逐个减半，保证优先选择收益最大的因子。
  for (int64_t factor = policy.maxRows; factor >= 2; factor /= 2) {
    // 静态网格必须整除，否则会改变 program 与查询行的映射；运行期范围
    // 用掩码处理最后一组，因此任意因子都可接受。
    // A static grid must divide exactly; a runtime extent masks the last group,
    // so any factor is admissible there.
    if (grid && grid % factor != 0)
      continue;
    // 提升后 M 维上限检查：maxDotRows * factor 不得超过 targetDotRows。
    if (policy.targetDotRows && maxDotRows > policy.targetDotRows / factor)
      continue;
    // 单值字节预算检查（乘法溢出时同样放弃该因子）。
    int64_t lifted;
    if (__builtin_mul_overflow(maxTensorBytes, factor, &lifted) ||
        lifted > budget)
      continue;
    // L0 容量是正确性约束：重写按因子放大 lhs 与累加器 tile，预算按因子
    // 均摊；预算为 0 表示目标架构容量未知，对应检查关闭。
    if ((policy.l0ABudgetBytes &&
         static_cast<uint64_t>(policy.maxL0ABytes) >
             policy.l0ABudgetBytes / factor) ||
        (policy.l0CBudgetBytes &&
         static_cast<uint64_t>(policy.maxL0CBytes) >
             policy.l0CBudgetBytes / factor))
      continue;
    return factor;
  }
  return 1;
}

// 返回单个元素的字节数；未知元素类型返回 0。
static int64_t getElementBytes(Type element) {
  if (auto integer = dyn_cast<IntegerType>(element))
    return (integer.getWidth() + 7) / 8;
  if (auto floatType = dyn_cast<FloatType>(element))
    return (floatType.getWidth() + 7) / 8;
  if (isa<triton::PointerType>(element))
    return 8;
  return 0;
}

// 返回张量总字节数；元素类型未知时返回 0。
static int64_t getTensorBytes(RankedTensorType type) {
  int64_t bytes = getElementBytes(type.getElementType());
  if (bytes == 0)
    return 0;
  for (int64_t extent : type.getShape())
    bytes *= extent;
  return bytes;
}

// 保存重写器与候选描述，能力开关从候选同步过来。
RowLiftBuilder::RowLiftBuilder(IRRewriter &rewriter, RowLiftCandidate candidate)
    : rewriter(rewriter), candidate(candidate), features(candidate.features) {}

// 克隆函数体并逐条重建，成功时用新函数体替换旧函数体并返回 true。
//
// 重建的单位是「函数级」而不是单条指令：先把原 block 的每条指令按行提升
// 语义重放一遍，全部成功后才丢弃旧 block，从而保证失败时原函数保持可用。
bool RowLiftBuilder::run(triton::FuncOp fn) {
  Block *old = &fn.getBody().front();
  // 原循环的起始上界。整组的 trip count 由末行取值决定，因此重建过程中
  // 需要知道哪些标量参与了这个上界的计算（见下方的末行切片克隆）。
  scf::ForOp loop = *old->getOps<scf::ForOp>().begin();
  Value upperBound = loop.getUpperBound();
  // 累加器拆分必须在重写任何指令之前定下来：要拆半的描述符可能位于循环
  // 之前，一旦先按整宽展开成指针张量，就无法再按半宽重建。
  // The accumulator split has to be decided before any instruction is rewritten:
  // a descriptor above the loop would otherwise be expanded at full width and
  // could no longer be rebuilt per half.
  plannedSplitIndices.clear();
  plannedSplitValues.clear();
  if (!planWideAccumulatorSplits(loop, plannedSplitIndices))
    return false;
  // 新建一个空 block 作为重建目标，先挂到函数体尾部，成功后替换旧 block。
  auto *fresh = new Block();
  fn.getBody().push_back(fresh);

  // 函数参数原样搬入新 block；valueMapping 用于普通重建，lastRowMapping
  // 额外记录「末行的映射」，使上界表达式能在末行取值上重新求值。
  for (BlockArgument arg : old->getArguments()) {
    Value newArg = fresh->addArgument(arg.getType(), arg.getLoc());
    valueMapping.map(arg, newArg);
    lastRowMapping.map(arg, newArg);
  }

  rewriter.setInsertionPointToStart(fresh);
  for (Operation &op : old->without_terminator()) {
    // 锚点就是程序号 tt.get_program_id：把单个 pid 展开成组内连续行号，
    // 不把原 pid 交给后续指令。
    if (&op == candidate.anchor) {
      // 克隆原 pid 作为组号的来源，再算组内首行 base = pid * rowsPerProgram。
      auto pid = cast<triton::GetProgramIdOp>(rewriter.clone(op));
      Value base = rewriter.create<arith::MulIOp>(
          op.getLoc(), pid.getResult(),
          rewriter.create<arith::ConstantIntOp>(
              op.getLoc(), candidate.rowsPerProgram, 32));
      // 末行 last = base + rowsPerProgram - 1，用于形成共享循环上界。
      Value last = rewriter.create<arith::AddIOp>(
          op.getLoc(), base,
          rewriter.create<arith::ConstantIntOp>(
              op.getLoc(), candidate.rowsPerProgram - 1, 32));
      // 运行期网格下组内末行可能越界，此处钳到 extent - 1，避免上界越界。
      if (candidate.runtimeGrid) {
        Value extent = fresh->getArgument(fresh->getNumArguments() - 2);
        Value lastValid = rewriter.create<arith::SubIOp>(
            op.getLoc(), extent,
            rewriter.create<arith::ConstantIntOp>(op.getLoc(), 1, 32));
        last = rewriter.create<arith::MinSIOp>(op.getLoc(), last, lastValid);
      }
      lastRowMapping.map(op.getResult(0), last);

      // pid 的提升值是一段连续行号：[base, base+1, ..., base+R-1]。
      auto t = getLiftedType(op.getResult(0).getType());
      Value lanes = rewriter.create<triton::MakeRangeOp>(
          op.getLoc(), t, 0, candidate.rowsPerProgram);
      Value start = rewriter.create<triton::SplatOp>(op.getLoc(), t, base);
      valueMapping.map(
          op.getResult(0),
          rewriter.create<arith::AddIOp>(op.getLoc(), start, lanes)
              .getResult());
      // 运行期网格下，再用 base + lanes < extent 得到逐行有效掩码，
      // 后续所有与行相关的访存都要在它之下重建。
      if (candidate.runtimeGrid) {
        Value extent = fresh->getArgument(fresh->getNumArguments() - 2);
        Value limit = rewriter.create<triton::SplatOp>(op.getLoc(), t, extent);
        validRows = rewriter.create<arith::CmpIOp>(
            op.getLoc(), arith::CmpIPredicate::slt,
            valueMapping.lookup(op.getResult(0)), limit);
      }
      continue;
    }

    // 只把「纯标量上界切片」克隆进末行映射：该切片就是参与循环上界计算的
    // 标量算术，在末行取值上重新求值得到组内最大 trip count。依赖判定必须
    // 穿透 region：scf.if 选出的上界来自分支 yield 而不是操作数，浅层判定
    // 会漏掉这段算术，导致循环重写时找不到末行上界。
    // Only clone the pure scalar bound slice for the last-row evaluation.  The
    // dependence test has to look through regions: a bound selected by an scf.if
    // takes its value from a branch yield rather than from an operand, so the
    // shallow test would leave that arithmetic unmapped and the loop rewrite
    // would then fail to find the last-row upper bound.
    if (op.getNumResults() == 1 &&
        !isa<RankedTensorType>(op.getResult(0).getType()) &&
        isScalarBoundSlice(&op) &&
        dependsOnValueThroughRegions(upperBound, op.getResult(0)))
      rewriter.clone(op, lastRowMapping);

    // 逐条重建；任一形态不被支持即整次提升失败，旧函数体保持不变。
    if (!rewrite(&op)) {
      LLVM_DEBUG(llvm::dbgs() << "row-lift: rewrite failed at " << op << "\n");
      return false;
    }

    // 记录本指令结果的映射。与行无关的值同时进入 lastRowMapping，作为末行
    // 表达式的输入；已展开的块指针没有描述符值，直接跳过。
    for (Value v : op.getResults()) {
      // 已展开的块指针没有可继续传递的描述符值。
      // An expanded block pointer has no descriptor value to carry over.
      if (!valueMapping.contains(v))
        continue;
      Value mapped = valueMapping.lookup(v);
      if (!hasRowDimension(v) && !lastRowMapping.contains(v))
        lastRowMapping.map(v, mapped);
    }
  }

  // 收尾：补 return，解除旧 block 上所有值的引用后删除旧 block。
  rewriter.create<triton::ReturnOp>(fn.getLoc());
  old->dropAllDefinedValueUses();
  old->erase();
  return true;
}

// 按操作种类分派重建；遇到尚未支持的形态返回 false，整次提升随即回退。
//
// 派发顺序带有语义，不能随意调整：
//   0) scf.for 先处理，循环重写会一次性重建所有循环进位值；
//   1) 「已被拆成两半的值」与「已提升的下标表读」必须早于通用依赖探测，
//      否则探测会去查不存在的全宽映射，克隆出非法操作数；
//   2) 依赖探测：与查询行无关的指令直接克隆，与行相关的才需要特化；
//   3) 带 region 的形态（tt.reduce / tt.make_tensor_ptr / scf.if）逐个重建；
//   4) 描述符访存、循环内行相关访存、dot / splat / expand / reshape / trans
//      走各自的形状规则；
//   5) 其余形态走通用克隆，操作数统一补行维、结果类型统一加行维。
bool RowLiftBuilder::rewrite(Operation *op) {
  // 步骤 0：循环是唯一会改变循环进位值个数的形态。
  if (auto loop = dyn_cast<scf::ForOp>(op))
    return rewriteLoop(loop);

  // 步骤 0b：已规划拆半的描述符必须在这里就拆半。这一步要早于下面的依赖
  // 探测：与行无关的描述符会被当作普通指令整宽克隆，而规划要求它产出两个
  // 半宽描述符；消费它的 load 也必须走两半 load，否则只映射出一个整宽值。
  // A descriptor planned for halving has to be split here, before the
  // dependence probe below: a row-invariant one would otherwise be cloned at
  // full width, and its load would map a single whole-width value.
  if (auto makePtr = dyn_cast<triton::MakeTensorPtrOp>(op)) {
    auto split = splitDescriptors.find(makePtr.getResult());
    if (split != splitDescriptors.end())
      return rewriteSplitTensorPointer(makePtr, split->second);
  }
  if (auto load = dyn_cast<triton::LoadOp>(op)) {
    auto split = splitDescriptors.find(load.getPtr());
    if (split != splitDescriptors.end())
      return rewriteSplitDescriptorLoad(load, split->second);
  }

  // 步骤 1a：拆分累加器没有全宽映射，消费方必须先走拆分感知路径，否则下面
  // 的依赖探测会查不到映射，克隆出带无效操作数的指令。
  // A split accumulator has no full-width mapping, so its consumers must be
  // rebuilt before the dependency probe below (which would otherwise look up
  // the absent mapping and clone the op with an invalid operand).
  if (llvm::any_of(op->getOperands(),
                   [&](Value v) { return splitValues.contains(v); }))
    return rewriteSplitOp(op);

  // 步骤 1b：下标表读已在循环外完成，循环内的列读变成列提取而不是访存。
  // 这里必须早于依赖探测，因为被服务的 load 地址本身与行相关，探测会把
  // 它当作普通指令克隆，从而留下未映射的操作数。
  // A hoisted index table has already been read before the loop, so its column
  // load becomes a column extraction instead of a memory access.  The lookup
  // runs before the dependence probe because the served load's address is
  // itself row-dependent and would otherwise be cloned as an unmapped operand.
  if (auto load = dyn_cast<triton::LoadOp>(op)) {
    auto table = columnTables.find(load.getPtr());
    if (table != columnTables.end()) {
      // 列提取需要循环归纳变量，不在循环内时无法构造，直接拒绝。
      if (!loopInductionVar)
        return false;
      valueMapping.map(load.getResult(),
                       extractTableColumn(table->second, loopInductionVar));
      return true;
    }
  }
  // 表读一旦改由循环外的整表读服务，原来的列地址计算就没有使用者了；
  // 只要它确实只有一个使用者就允许删除，返回 true 表示「本条无需重建」。
  // The column address computation is dead once its only use, the table load,
  // is served by the hoisted read.
  if (auto add = dyn_cast<triton::AddPtrOp>(op))
    if (columnTables.contains(add.getResult()))
      return add.getResult().hasOneUse();

  // 步骤 2：判断本指令是否与查询行相关。
  //   descriptor —— 操作数是已展开的块指针，必须走描述符访存路径；
  //   dependent  —— 操作数或嵌套 region 里用到了带行维的值。
  bool descriptor = llvm::any_of(
      op->getOperands(),
      [&](Value v) { return liftedDescriptors.contains(v); });
  bool dependent =
      descriptor || llvm::any_of(op->getOperands(), [&](Value v) {
        return hasRowDimension(v);
      });
  // 带 region 的指令可能只在 region 内部使用提升值，仅看顶层操作数不足
  // 以判定它与行无关，因此还要扫描所有嵌套 region。
  // An op with regions may only use lifted values in its body, so the top-level
  // operands alone do not prove it invariant.
  if (!dependent)
    op->walk([&](Operation *nested) {
      for (Value v : nested->getOperands())
        if (!dependent && valueMapping.contains(v) && hasRowDimension(v))
          dependent = true;
    });
  // 与行无关：保持原样克隆（其操作数映射到的值与原始类型相同）。
  if (!dependent) {
    rewriter.clone(*op, valueMapping);
    return true;
  }

  // 已展开的块指针只可能被 load/store 消费；其他消费者（例如 tt.advance）
  // 还没有指针张量等价物，直接拒绝。
  // An expanded block pointer is consumed by a load or store below; any other
  // consumer (for example tt.advance) has no pointer tensor equivalent yet.
  if (descriptor && !isa<triton::LoadOp, triton::StoreOp>(op))
    return false;

  // 步骤 3：两个带 region 的形态无法走通用克隆（通用克隆只复制操作数与
  // 结果类型，会留下缺失 region 的指令），必须单独重建。
  // Dispatch the two region-bearing shapes that the generic path below cannot
  // rebuild: it clones operands and result types only, so it would emit an op
  // whose region is missing.
  if (auto reduce = dyn_cast<triton::ReduceOp>(op))
    return rewriteReduce(reduce);
  if (auto makePtr = dyn_cast<triton::MakeTensorPtrOp>(op))
    return rewriteTensorPointer(makePtr);
  // 目前只有 scf.if 能按带 region 的方式重建；其他形态失败关闭，避免生成
  // 缺少 region 的操作。
  if (auto ifOp = dyn_cast<scf::IfOp>(op))
    return rewriteIfOp(ifOp);
  if (op->getNumRegions() != 0)
    return false;

  // 步骤 4a：块指针 load/store 用展开后的指针张量重建，边界检查转掩码。
  if (auto load = dyn_cast<triton::LoadOp>(op)) {
    auto it = liftedDescriptors.find(load.getPtr());
    if (it != liftedDescriptors.end())
      return rewriteDescriptorLoad(load, it->second);
  }
  if (auto store = dyn_cast<triton::StoreOp>(op)) {
    auto it = liftedDescriptors.find(store.getPtr());
    if (it != liftedDescriptors.end())
      return rewriteDescriptorStore(store, it->second);
  }

  // 步骤 4b：循环体内与行相关的普通 load。某些行的扫描可能已经结束，只有
  // 逐行有效掩码能把它们挡在内存之外，缺少该能力时保持原有的投机访存
  // 语义并拒绝提升。
  if (auto load = dyn_cast<triton::LoadOp>(op)) {
    if (op->getParentOfType<scf::ForOp>()) {
      // A row-dependent load in the body reads the addresses of rows whose
      // scan may already have ended.  Only the per-row active mask keeps those
      // rows off memory, so without the capability the load stays speculative
      // and is declined.
      if (!features.maskedLoad || !rowActiveMask)
        return false;
      Value pointer = ensureRowDimension(load.getPtr());
      if (!isa<RankedTensorType>(pointer.getType()))
        return false;
      bool ok = true;
      Value mask =
          combineMemoryMask(pointer, load.getMask(), rowActiveMask, ok);
      if (!ok || !mask)
        return false;
      // other 值（掩码为假时的填充）同样要补行维并与提升后的类型对齐。
      auto type = getLiftedType(load.getType());
      Value other =
          load.getOther()
              ? broadcastFullValue(ensureRowDimension(load.getOther()), type)
              : Value(rewriter.create<arith::ConstantOp>(
                    op->getLoc(), rewriter.getZeroAttr(type)));
      auto replacement = rewriter.create<triton::LoadOp>(
          op->getLoc(), pointer, mask, other, load.getBoundaryCheck(),
          load.getPadding(), load.getCache(), load.getEvict(),
          load.getIsVolatile());
      valueMapping.map(load.getResult(), replacement.getResult());
      return true;
    }
    // 循环外的 load 没有「扫描提前结束」的行，继续走下面通用重建，
    // 保持历史行为。
    // A load above the loop has no row whose scan could have ended early, so it
    // keeps the historical generic rebuild below.
  }
  // 步骤 4c：dot。只有 A 与查询行相关、B 与行无关时才提升：把 [R,M,K] 展平
  // 成 [R*M,K] 复用普通 dot，结果再 reshape 回 [R,M,N]。
  if (auto dot = dyn_cast<triton::DotOp>(op)) {
    if (!hasRowDimension(dot.getA()) || hasRowDimension(dot.getB()))
      return false;

    auto result = cast<RankedTensorType>(dot.getType());
    SmallVector<Value> operands{
        flattenDotRows(ensureRowDimension(dot.getA())),
        getMappedValue(dot.getB()),
        flattenDotRows(ensureRowDimension(dot.getC()))};
    Type flatType = RankedTensorType::get(
        {candidate.rowsPerProgram * result.getShape()[0],
         result.getShape()[1]},
        result.getElementType());
    Operation *d =
        rewriter.create(op->getLoc(), op->getName().getIdentifier(), operands,
                        TypeRange{flatType}, op->getAttrs());
    valueMapping.map(
        dot.getResult(),
        reshape(d->getResult(0), getLiftedType(result).getShape()));
    return true;
  }

  // 步骤 4d：标量 splat 变成「先补行维再广播到提升形状」。
  if (auto splat = dyn_cast<triton::SplatOp>(op)) {
    valueMapping.map(splat.getResult(),
                     broadcastRowValue(ensureRowDimension(splat.getSrc()),
                                       getLiftedType(splat.getType())));
    return true;
  }

  // 步骤 4e：插入维度时轴号整体后移一位，因为行维占用了 0 号轴。
  if (auto expand = dyn_cast<triton::ExpandDimsOp>(op)) {
    valueMapping.map(
        expand.getResult(),
        rewriter
            .create<triton::ExpandDimsOp>(op->getLoc(),
                                          ensureRowDimension(expand.getSrc()),
                                          expand.getAxis() + 1)
            .getResult());
    return true;
  }

  // 步骤 4f：reshape 保持每行内部元素顺序，避免把不同 program 的值混在一起。
  if (auto view = dyn_cast<triton::ReshapeOp>(op)) {
    // Preserve each row's element order. Allowing arbitrary reordering of
    // the lifted tensor could mix values belonging to different
    // programs.
    valueMapping.map(view.getResult(),
                     reshape(ensureRowDimension(view.getSrc()),
                             getLiftedType(view.getType()).getShape()));
    return true;
  }

  // 步骤 4g：转置的轴号同样整体后移一位，行维固定在 0 号轴不参与置换。
  if (auto trans = dyn_cast<triton::TransOp>(op)) {
    SmallVector<int32_t> order{0};
    for (int32_t axis : trans.getOrder())
      order.push_back(axis + 1);
    valueMapping.map(
        trans.getResult(),
        rewriter
            .create<triton::TransOp>(
                op->getLoc(), ensureRowDimension(trans.getSrc()), order)
            .getResult());
    return true;
  }

  // 步骤 4h：运行期网格下，循环外的 load 也要带有效行掩码，越界的行
  // 不能被读到。
  if (candidate.runtimeGrid) {
    if (auto load = dyn_cast<triton::LoadOp>(op)) {
      Value pointer = ensureRowDimension(load.getPtr());
      if (!isa<RankedTensorType>(pointer.getType()))
        return false;
      bool ok = true;
      Value mask = combineMemoryMask(pointer, load.getMask(), Value(), ok);
      if (!ok)
        return false;
      auto type = getLiftedType(load.getType());
      Value other = load.getOther()
                        ? ensureRowDimension(load.getOther())
                        : rewriter.create<arith::ConstantOp>(
                              op->getLoc(), rewriter.getZeroAttr(type));
      auto replacement = rewriter.create<triton::LoadOp>(
          op->getLoc(), pointer, mask, other, load.getBoundaryCheck(),
          load.getPadding(), load.getCache(), load.getEvict(),
          load.getIsVolatile());
      valueMapping.map(load.getResult(), replacement.getResult());
      return true;
    }
  }

  // 步骤 4i：带掩码的 store（O、lse 的写回）只有在 maskedStore 能力开启时
  // 才提升；否则通用克隆会构造出形状与提升后指针不匹配的 mask。
  // A masked store is lifted only when the capability is enabled; the generic
  // clone below would otherwise build a mask whose shape no longer matches the
  // lifted pointer.
  if (auto store = dyn_cast<triton::StoreOp>(op)) {
    if (candidate.runtimeGrid || (store.getMask() && features.maskedStore)) {
      Value pointer = ensureRowDimension(store.getPtr());
      if (!isa<RankedTensorType>(pointer.getType()))
        return false;
      bool ok = true;
      Value mask = combineMemoryMask(pointer, store.getMask(), Value(), ok);
      if (!ok)
        return false;
      rewriter.create<triton::StoreOp>(
          op->getLoc(), pointer, ensureRowDimension(store.getValue()), mask,
          store.getBoundaryCheck(), store.getCache(), store.getEvict());
      return true;
    }
  }

  // 步骤 5：通用克隆。操作数统一补行维，结果类型统一在首部插入 [R]。
  SmallVector<Value> operands;
  for (Value v : op->getOperands())
    operands.push_back(ensureRowDimension(v));

  SmallVector<Type> types;
  for (Type t : op->getResultTypes())
    types.push_back(getLiftedType(t));

  Operation *cloned =
      rewriter.create(op->getLoc(), op->getName().getIdentifier(), operands,
                      types, op->getAttrs());
  valueMapping.map(op->getResults(), cloned->getResults());
  return true;
}

// 合并描述符边界掩码、运行期有效行掩码与原始 mask 操作数，返回合并后的 i1 掩码。
// 合并规则：所有掩码都对齐到指针张量的形状后逐项做 arith.and；没有掩码时
// 返回空 Value（表示不限定的访存）；某一部分无法对齐时把 ok 置 false。
Value RowLiftBuilder::combineMemoryMask(Value pointer, Value originalMask,
                                        Value extraMask, bool &ok) {
  // 合并顺序固定为「描述符边界掩码 -> 有效行掩码 -> 原始 mask」。原始 mask 的
  // 行维展开必须留到第二步才生成，否则它会先于前两步的合并指令出现，导致
  // 未开启新能力时的指令序列与基线不再逐字节一致。
  // Keep the historical merge order (descriptor boundary mask, valid rows,
  // then the original mask), and delay the row expansion of the original mask to
  // the second step so an unset feature reproduces the baseline instruction
  // sequence byte for byte.
  Value mask = mergeMasks(
      pointer, Value(),
      {extraMask, candidate.runtimeGrid ? validRows : Value()}, ok);
  if (!ok)
    return Value();
  return mergeMasks(
      pointer, mask, {originalMask ? ensureRowDimension(originalMask) : Value()},
      ok);
}

// 把若干已成形（或只需尾部广播）的掩码分量合并到指针形状上。不做行维补齐，
// 因为拆半路径传入的就是半宽掩码。
// Merge mask parts that are already shaped (or only need trailing broadcast) onto
// the pointer shape.  No row dimension is inserted, because the split path hands
// in half-width masks.
Value RowLiftBuilder::mergeMasks(Value pointer, Value base, ArrayRef<Value> parts,
                                 bool &ok) {
  ok = true;
  // 掩码的基准形状 / 类型：与指针张量同形，元素为 i1。
  auto pointerType = cast<RankedTensorType>(pointer.getType());
  auto maskType =
      RankedTensorType::get(pointerType.getShape(), rewriter.getI1Type());
  Value mask = base;
  // 单个掩码分量的对齐与累积：形状不同时只能通过补尾部单位维再广播，
  // 因此已与指针同秩但形状不同的分量被拒绝。
  auto merge = [&](Value part) {
    if (!part || !ok)
      return;
    if (part.getType() != maskType) {
      auto partType = dyn_cast<RankedTensorType>(part.getType());
      // Aligning can only add trailing unit dimensions to a row vector; a
      // part that already ranks like the pointer but has another shape
      // cannot be merged safely.
      if (!partType || partType.getElementType() != rewriter.getI1Type() ||
          partType.getRank() >= maskType.getRank()) {
        ok = false;
        return;
      }
      part = broadcastRowValue(part, maskType);
    }
    mask = mask
               ? Value(rewriter.create<arith::AndIOp>(part.getLoc(), mask, part))
               : part;
  };
  for (Value part : parts)
    merge(part);
  return mask;
}

// 把原循环重建为多行循环：规划表读与累加器拆分，并用末行上界生成新循环。
//
// 重建步骤：
//   1) 循环进位值先补行维，作为新循环的初始值；
//   2) 下标表读规划必须在循环体重建之前完成，因为它会改写 load 的语义；
//   3) 累加器拆分规划也必须在建循环之前完成，因为它会改变进位值个数；
//   4) 新循环的 trip count 取「末行上界」，单调性由 matcher 已证明；
//   5) 循环体逐条重建；maskedLoad 需要先准备好逐行有效掩码；
//   6) 循环末尾用有效行掩码选择「本轮新值 / 上一轮值」；
//   7) 循环结果写回映射；拆分的累加器以成对结果继续向后传递。
bool RowLiftBuilder::rewriteLoop(scf::ForOp old) {
  // 已展开成指针张量的块指针不能作为循环进位值：展开时已删掉描述符，
  // 循环里没有它的等价物。
  // A loop carrying an expanded block pointer would need a descriptor value
  // that the expansion deliberately removed.
  for (Value arg : old.getInitArgs())
    if (liftedDescriptors.contains(arg))
      return false;

  // 每个循环从头开始，清空上一个循环遗留的归纳变量与有效行掩码。
  rowActiveMask = Value();
  loopInductionVar = Value();

  // 步骤 2：把循环体内的标量下标表读提升为循环外一次带掩码的 [R,S] 整表
  // 读，循环体只提取列、不再访存。
  // The scalar index-table reads are hoisted into one masked [R,S] vector read
  // per program, so the body only extracts its column and touches no memory.
  if (features.tableRead && !planTableRead(old))
    return false;

  // 步骤 1+3：原循环的进位值统一补上行维；哪些累加器要拆半已在 run 里定下
  // （描述符必须先于展开拆半），新循环要为每个拆半值携带两个进位值。拆半
  // 初值直接从原始进位值算，避免对已提升值再做一次行维提升。
  // Step 1+3: every carried value gains the row dimension; the accumulators
  // planned for halving in `run` become two loop-carried values.  A split
  // initial value is derived from the original one, never from its lifted copy.
  SmallVector<int64_t> splitIndices(plannedSplitIndices.begin(),
                                    plannedSplitIndices.end());

  // 建立「原进位值索引 -> 新进位值起始位置 + 个数」的映射：一个未拆分的
  // 累加器对应 1 个新进位值，拆分的累加器对应 2 个半宽值。
  // One original accumulator maps onto one or two loop-carried values.
  SmallVector<Value> loopInit;
  SmallVector<std::pair<int64_t, int64_t>> ranges;
  for (auto [index, arg] : llvm::enumerate(old.getInitArgs())) {
    if (llvm::is_contained(splitIndices, static_cast<int64_t>(index))) {
      SplitValue halves;
      if (!splitInitialValue(arg, halves))
        return false;
      ranges.push_back({loopInit.size(), 2});
      loopInit.push_back(halves.first);
      loopInit.push_back(halves.second);
    } else {
      ranges.push_back({loopInit.size(), 1});
      loopInit.push_back(ensureRowDimension(arg));
    }
  }

  // 步骤 4：末行上界必须已经物化到 lastRowMapping（run 里克隆的标量上界
  // 切片产生），否则说明上界被不支持的 region 包住，本次提升放弃。
  // The monotonic upper-bound proof means the last row supplies the maximum.
  // A bound whose last-row copy was not materialised (for example because it
  // sits behind an unsupported region) declines instead of aborting.
  if (!lastRowMapping.contains(old.getUpperBound())) {
    LLVM_DEBUG(llvm::dbgs() << "row-lift: last-row bound missing\n");
    return false;
  }
  // 下界与步长沿用原循环（它们与行无关），上界换成末行取值。
  auto loop = rewriter.create<scf::ForOp>(
      old.getLoc(), getMappedValue(old.getLowerBound()),
      lastRowMapping.lookup(old.getUpperBound()), getMappedValue(old.getStep()),
      loopInit);
  loop->setAttrs(old->getAttrs());
  loopInductionVar = loop.getInductionVar();

  // 归纳变量与各进位值的映射：拆分值登记到 splitValues，普通值登记到
  // valueMapping。
  valueMapping.map(old.getInductionVar(), loop.getInductionVar());
  for (auto [index, arg] : llvm::enumerate(old.getRegionIterArgs())) {
    auto [start, count] = ranges[index];
    if (count == 2)
      splitValues[arg] = {loop.getRegionIterArgs()[start],
                          loop.getRegionIterArgs()[start + 1]};
    else
      valueMapping.map(arg, loop.getRegionIterArgs()[start]);
  }

  rewriter.setInsertionPointToStart(loop.getBody());
  // 步骤 5：循环体内与行相关的访存需要逐行有效掩码，因此必须在重建循环体
  // 之前就准备好。
  // A row-dependent body access needs the per-row mask, so it has to be
  // available before the body is rebuilt.
  if (features.maskedLoad)
    rowActiveMask = createLoopActiveMask(old, loop);
  for (Operation &op : old.getBody()->without_terminator())
    if (!rewrite(&op))
      return false;

  auto yield = cast<scf::YieldOp>(old.getBody()->getTerminator());
  // maskedLoad 关闭时，有效行掩码仍要在循环体之后生成；复用同一个辅助
  // 函数，指令序列与基线保持一致。
  if (!rowActiveMask)
    rowActiveMask = createLoopActiveMask(old, loop);
  Value active = rowActiveMask;

  // 步骤 6：对「原循环在本轮没有执行」的行，保持上一轮的值不变（同时保留
  // NaN/Inf 语义）。掩码为假时选 previous。
  // Retain exactly the previous accumulator for rows whose original loop did
  // not execute this iteration (also preserves NaN/Inf behavior).
  auto keep = [&](Value incoming, Value previous, RankedTensorType type) {
    Value mask = broadcastRowValue(
        active, RankedTensorType::get(type.getShape(), rewriter.getI1Type()));
    return Value(rewriter.create<arith::SelectOp>(old.getLoc(), mask, incoming,
                                                  previous));
  };

  // 逐项生成新的 yield：拆分累加器产生两个半宽值，其余值补行维后逐项选择。
  SmallVector<Value> output;
  for (auto [index, value] : llvm::enumerate(yield.getOperands())) {
    auto [start, count] = ranges[index];
    if (count == 2) {
      auto it = splitValues.find(value);
      if (it == splitValues.end())
        return false;
      output.push_back(keep(
          it->second.first, loop.getRegionIterArgs()[start],
          cast<RankedTensorType>(loop.getRegionIterArgs()[start].getType())));
      output.push_back(keep(
          it->second.second, loop.getRegionIterArgs()[start + 1],
          cast<RankedTensorType>(
              loop.getRegionIterArgs()[start + 1].getType())));
      continue;
    }
    auto previous =
        cast<RankedTensorType>(loop.getRegionIterArgs()[start].getType());
    output.push_back(keep(ensureRowDimension(value),
                          loop.getRegionIterArgs()[start], previous));
  }

  // 用新生成的 yield 替换循环体末尾的占位 terminator。
  if (!loop.getBody()->empty() && isa<scf::YieldOp>(loop.getBody()->back()))
    rewriter.eraseOp(loop.getBody()->getTerminator());
  rewriter.setInsertionPointToEnd(loop.getBody());
  rewriter.create<scf::YieldOp>(old.getLoc(), output);

  // 步骤 7：循环结果回填映射。拆分的累加器以成对结果离开循环，后续 store
  // 可以按同样方式拆它的指针，而不必重建全宽值。
  for (auto [index, result] : llvm::enumerate(old.getResults())) {
    auto [start, count] = ranges[index];
    if (count == 2) {
      // The halves leave the loop as a pair so that a later store can split its
      // pointer the same way instead of rebuilding the full width.
      splitValues[result] = {loop.getResults()[start],
                             loop.getResults()[start + 1]};
      continue;
    }
    valueMapping.map(result, loop.getResults()[start]);
  }

  // 循环体之后继续发出后续指令。
  rewriter.setInsertionPointAfter(loop);
  return true;
}

// 生成 iv < ub 的逐行有效掩码：把循环归纳变量广播成 [R] 向量，再与提升后
// 的上界逐行比较。
Value RowLiftBuilder::createLoopActiveMask(scf::ForOp old, scf::ForOp loop) {
  Value ivSplat = rewriter.create<triton::SplatOp>(
      old.getLoc(), getLiftedType(old.getInductionVar().getType()),
      loop.getInductionVar());
  return rewriter.create<arith::CmpIOp>(
      old.getLoc(), arith::CmpIPredicate::slt, ivSplat,
      ensureRowDimension(old.getUpperBound()));
}

// 判断 `base` 在提升后是否会带上行维：与查询行相关的值会被提升，循环进位值
// 无论是否与行相关都会被强制提升（与 RowLiftBuilder 的 lifted 集合一致）。
// A value carries the row dimension after lifting when it depends on the query
// row; loop-carried values are lifted unconditionally, matching the matcher's
// lifted set.
static bool willCarryRowDimension(Value base, scf::ForOp loop, Value pid) {
  if (dependsOnValue(base, pid))
    return true;
  if (auto argument = dyn_cast<BlockArgument>(base))
    return argument.getOwner()->getParentOp() == loop.getOperation();
  return false;
}

// 下标表提升的完整判据。规则 matcher 与 planTableRead 都调用它：matcher 用它
// 在匹配阶段拒绝提升不了的形态，重写路径用它保证自己独立成立。
// The complete predicate for index-table hoisting, called by both the rule's
// matcher and planTableRead.
bool canHoistTableReads(scf::ForOp loop, const RowLiftCandidate &candidate) {
  SmallVector<triton::LoadOp> tableLoads;
  for (Operation &op : loop.getBody()->without_terminator()) {
    auto load = dyn_cast<triton::LoadOp>(&op);
    if (load && !isa<RankedTensorType>(load.getType()))
      tableLoads.push_back(load);
  }
  // 没有标量表读的循环不需要列宽，因此不因上界不可证而拒绝。
  // A loop without a scalar table read needs no column extent.
  if (tableLoads.empty())
    return true;

  // 列宽来自上界证明：表读能力必须与 reduce 能力同时开启，且运行期网格
  // 没有静态 trip count，无法给出列宽。
  // The column extent comes from the bound proof: the table-read capability
  // rides on `reduce`, and a runtime grid has no static trip count.
  if (!candidate.features.tableRead || !candidate.features.reduce ||
      candidate.grid <= 0)
    return false;
  auto pidOp = dyn_cast<triton::GetProgramIdOp>(candidate.anchor);
  if (!pidOp)
    return false;
  Value pid = pidOp.getResult();
  auto bound = inferMonotonicBounds(loop.getUpperBound(), pid, candidate.grid);
  if (!bound || bound->hi <= 0)
    return false;

  // 按上界 hi 向上取 2 的幂作为静态列宽；过宽的列网格放弃提升。
  int64_t columns = 1;
  while (columns < bound->hi)
    columns *= 2;
  if (columns > kMaxTableColumns)
    return false;

  // 逐个表读必须是顺序扫描形态 `load(base + iv)`：列号就是迭代下标，且该
  // 地址没有其他使用者；基址是整数元素标量指针，提升后带行维（每行一份表）。
  for (triton::LoadOp load : tableLoads) {
    auto address = load.getPtr().getDefiningOp<triton::AddPtrOp>();
    if (!address || address.getOffset() != loop.getInductionVar() ||
        !load.getPtr().hasOneUse())
      return false;
    Value base = address.getPtr();
    auto pointer = dyn_cast<triton::PointerType>(base.getType());
    if (!pointer || !isa<IntegerType>(pointer.getPointeeType()) ||
        !willCarryRowDimension(base, loop, pid))
      return false;
  }
  return true;
}

// 把循环内 load(base+iv) 形式的标量下标表读提升为循环外一次 [R,S] 掩码读。
// 判据与规则 matcher 共用：匹配阶段已经拒绝提升不了的形态，这里再判一次，
// 使重写路径不依赖 matcher 也能独立成立。
bool RowLiftBuilder::planTableRead(scf::ForOp old) {
  if (!canHoistTableReads(old, candidate))
    return false;

  // 第一步：收集可提升的标量表读；没有时整条链路无事可做。
  // Collect the hoistable scalar reads; a loop without one has nothing to do.
  SmallVector<triton::LoadOp> tableLoads;
  for (Operation &op : old.getBody()->without_terminator()) {
    auto load = dyn_cast<triton::LoadOp>(&op);
    if (load && !isa<RankedTensorType>(load.getType()))
      tableLoads.push_back(load);
  }
  if (tableLoads.empty())
    return true;

  // 第二步：列宽取自上界证明。判据已保证证明存在且为正，这里只把它换算成
  // 2 的幂：组内任何一行的 trip count 都不会超过上界 hi，用不到的列由掩码
  // 屏蔽、读取值为 0。
  // The column extent comes from the bound proof; the predicate already
  // guarantees a positive range, so it is only rounded up to a power of two.
  Value pid = cast<triton::GetProgramIdOp>(candidate.anchor).getResult();
  auto bound = inferMonotonicBounds(old.getUpperBound(), pid, candidate.grid);
  if (!bound)
    return false;
  int64_t columns = 1;
  while (columns < bound->hi)
    columns *= 2;

  Location loc = old.getLoc();
  // 第四步：逐个取出可提升的表读，在循环外构造一张 [R,S] 的指针网格并读入。
  for (triton::LoadOp load : tableLoads) {
    // 形态已由 canHoistTableReads 保证；这里只取回表基址，用于构造行基址
    // 向量。判据里的「是否会带行维」在重写期等价于 hasRowDimension。
    // The shape was already checked by canHoistTableReads; only the table base
    // is needed here to build the per-row base vector.
    auto address = load.getPtr().getDefiningOp<triton::AddPtrOp>();
    Value base = address.getPtr();

    // 构造 [R,S] 相关类型：指针网格、值网格、偏移网格，以及列下标 [S]。
    Value rowBase = ensureRowDimension(base);
    auto rowBaseType = cast<RankedTensorType>(rowBase.getType());
    int64_t rows = rowBaseType.getShape()[0];
    auto elementPointer =
        cast<triton::PointerType>(rowBaseType.getElementType());
    auto pointerGridType = RankedTensorType::get({rows, columns},
                                                 elementPointer);
    auto valueType = RankedTensorType::get({rows, columns},
                                            elementPointer.getPointeeType());
    auto offsetType =
        RankedTensorType::get({rows, columns}, rewriter.getI32Type());
    auto columnType = RankedTensorType::get({columns}, rewriter.getI32Type());

    // 列号向量 [0,1,...,S-1]，广播成 [R,S]；表基址同样广播，二者相加得到
    // 每个 (行, 列) 的地址。
    Value columnsRange = rewriter.create<triton::MakeRangeOp>(
        loc, columnType, 0, static_cast<int32_t>(columns));
    Value columnGrid = rewriter.create<triton::BroadcastOp>(
        loc, offsetType,
        rewriter.create<triton::ExpandDimsOp>(loc, columnsRange, 0));
    Value pointerGrid = rewriter.create<triton::BroadcastOp>(
        loc, pointerGridType,
        rewriter.create<triton::ExpandDimsOp>(loc, rowBase, 1));
    Value pointers = rewriter.create<triton::AddPtrOp>(
        loc, pointerGridType, pointerGrid, columnGrid);

    // 原顺序扫描的「越界即停」变成显式掩码：列号超过该行自身 trip count
    // 的位置永不读取，因此任何行都不会读到别的行的表项。
    // The boundary check of the scan becomes the explicit mask: a column past
    // the row's own trip count is never read, so no row reads another row's
    // table.
    Value boundGrid = rewriter.create<triton::BroadcastOp>(
        loc, offsetType,
        rewriter.create<triton::ExpandDimsOp>(
            loc, ensureRowDimension(old.getUpperBound()), 1));
    Value mask = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, columnGrid, boundGrid);
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(valueType));
    // 每 program 只做这一次 [R,S] 掩码读，结果按「原列地址」登记，供循环
    // 内的 load 改为列提取。
    auto table = rewriter.create<triton::LoadOp>(
        loc, pointers, mask, zero, load.getCache(), load.getEvict(), false);
    columnTables[load.getPtr()] =
        TableRead{table.getResult(), zero, columnsRange};
  }
  return true;
}

// 用 one-hot 选择把非目标列清零，再沿列求和得到第 iv 列。
Value RowLiftBuilder::extractTableColumn(const TableRead &read, Value iv) {
  Location loc = read.table.getLoc();
  auto tableType = cast<RankedTensorType>(read.table.getType());
  auto columnType = cast<RankedTensorType>(read.columns.getType());
  auto maskType =
      RankedTensorType::get(tableType.getShape(), rewriter.getI1Type());

  // 构造 one-hot 列掩码：与 iv 相等的列置 1，其余列置 0。
  // Only the column holding the iteration index survives the select, so the
  // row sum is exactly that one table entry.
  Value index = rewriter.create<triton::SplatOp>(loc, columnType, iv);
  Value onehot = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq, read.columns, index);
  // 选择后只有目标列保留原值，其余列变成 0；再沿列求和即得到该列的 [R]
  // 切片。这里不做 gather，避免引入逐行的动态索引。
  Value picked = rewriter.create<arith::SelectOp>(
      loc,
      rewriter.create<triton::BroadcastOp>(
          loc, maskType,
          rewriter.create<triton::ExpandDimsOp>(loc, onehot, 0)),
      read.table, read.zero);
  return sumAlongAxis(loc, picked, 1);
}

// 用整数加法组合子在指定轴上求和，生成 tt.reduce。
Value RowLiftBuilder::sumAlongAxis(Location loc, Value value, int64_t axis) {
  auto sourceType = cast<RankedTensorType>(value.getType());
  Type element = sourceType.getElementType();
  OpBuilder::InsertionGuard guard(rewriter);
  auto reduce = rewriter.create<triton::ReduceOp>(
      loc, ValueRange{value}, static_cast<int32_t>(axis));
  Block *combine = rewriter.createBlock(&reduce.getCombineOp());
  combine->addArgument(element, loc);
  combine->addArgument(element, loc);
  rewriter.setInsertionPointToStart(combine);
  Value sum = rewriter.create<arith::AddIOp>(
      loc, combine->getArgument(0), combine->getArgument(1));
  rewriter.create<triton::ReduceReturnOp>(loc, sum);
  return reduce->getResult(0);
}

// 前向声明：planWideAccumulatorSplits 与 canSplitWideAccumulators 都复用同
// 一份累加器岛判定（定义见下方）。
// Forward declaration: the accumulator-island check is shared by
// planWideAccumulatorSplits and canSplitWideAccumulators.
static bool isSplitFriendlyValue(Value value, Operation *loop, bool &sawDot);

// 若一次使用是该 scf.if 分支末尾的 yield，返回对应的 if 结果；否则返回空值。
// 累加器经过 scf.if 时值会变成 if 结果，岛分析必须穿过这一层，否则后半段
// 的消费者（循环后的归一化与写回）会被误判为岛外。
// If a use is the terminator of an scf.if branch, return the matching if result;
// otherwise return a null value.  An accumulator that flows through an scf.if
// turns into the if result, so the island analysis has to follow that edge or
// the post-loop consumers would look like they are outside the island.
static Value ifResultForYield(OpOperand &use) {
  auto yield = dyn_cast<scf::YieldOp>(use.getOwner());
  if (!yield)
    return Value();
  auto ifOp = dyn_cast<scf::IfOp>(yield->getParentOp());
  if (!ifOp || use.getOperandNumber() >= ifOp.getNumResults())
    return Value();
  return ifOp.getResult(use.getOperandNumber());
}

// 收集累加器岛里由该值供给的 dot 与 store。遍历规则与 isSplitFriendlyValue
// 保持一致：只跟进 dot 的累加器、store 的值、逐元素算术与 scf.if 的结果。
// `island` 非空时同时记录岛内途经的每个原始值，供 scf.if 在建结果类型之前
// 判断哪些结果要占两个半宽项。
// Collect the dots and stores fed by `root` inside its accumulator island,
// following exactly the same edges as isSplitFriendlyValue.  When `island` is
// non-null every visited value is recorded as well, so an scf.if can decide
// which results take two half-width types before it is created.
static void collectSplitIsland(Value root, SmallVectorImpl<triton::DotOp> &dots,
                               SmallVectorImpl<triton::StoreOp> &stores,
                               Operation *loop = nullptr,
                               DenseSet<Value> *island = nullptr) {
  SmallVector<Value> work{root};
  DenseSet<Value> seen;
  while (!work.empty()) {
    Value value = work.pop_back_val();
    if (!seen.insert(value).second)
      continue;
    if (island)
      island->insert(value);
    for (OpOperand &use : value.getUses()) {
      Operation *user = use.getOwner();
      if (auto dot = dyn_cast<triton::DotOp>(user)) {
        if (use.get() == dot.getC()) {
          dots.push_back(dot);
          work.push_back(dot.getResult());
        }
        continue;
      }
      if (auto store = dyn_cast<triton::StoreOp>(user)) {
        if (use.get() == store.getValue())
          stores.push_back(store);
        continue;
      }
      if (isa<scf::YieldOp>(user)) {
        // 回到本循环背边不算岛外：本循环的结果继续在岛内（循环之后的归一化
        // 与写回都在岛上），if 分支的 yield 则沿对应结果走。
        // Yielding to this loop's back edge stays inside the island: its result
        // feeds the post-loop normalisation and write-back.  A branch yield of an
        // scf.if continues on the matching if result.
        if (loop && user->getParentOp() == loop) {
          auto forOp = cast<scf::ForOp>(loop);
          unsigned index = use.getOperandNumber();
          if (index < forOp.getNumResults())
            work.push_back(forOp.getResult(index));
        } else if (Value result = ifResultForYield(use)) {
          work.push_back(result);
        }
        continue;
      }
      if (user->getNumResults() == 1 && user->getNumRegions() == 0) {
        StringRef dialect = user->getName().getDialectNamespace();
        if (dialect == "arith" || dialect == "math")
          work.push_back(user->getResult(0));
      }
    }
  }
}

// 描述符能否沿最后一维拆成两半：它必须由块指针构造，tile 的最后一维与半宽
// 对应的整宽一致且为偶数。
// True when the descriptor can be halved along its last dimension: it is built
// by a block pointer, and its last tile extent equals the full width and splits
// into two equal halves.
static bool descriptorHalvesAtWidth(Value pointer, int64_t width) {
  auto makePtr = pointer.getDefiningOp<triton::MakeTensorPtrOp>();
  if (!makePtr)
    return false;
  auto pointerType = dyn_cast<triton::PointerType>(makePtr.getType());
  auto blockType =
      pointerType ? dyn_cast<RankedTensorType>(pointerType.getPointeeType())
                  : RankedTensorType();
  return blockType && blockType.getRank() > 0 &&
         blockType.getShape().back() == width && width >= 2 && width % 2 == 0;
}

// 累加器岛能否结构拆半：每个 dot 的 B 必须来自一个最后一维同宽的块指针
// load，每个 store 的指针也必须是最后一维同宽的块指针，这样 V 的半宽由访存
// 本身产生，而不是靠 reshape/trans/split 重组值。成功时把描述符追加到
// `descriptors`。
// True when the accumulator island can be split structurally: each dot's B has
// to come from a block-pointer load whose last tile extent is the accumulator
// width, and each store's pointer has to be a block pointer of the same width,
// so the half-width V is produced by the memory access itself rather than by a
// reshape/trans/split of a value.  The descriptors are appended on success.
static bool describeSplitDescriptor(Value root, int64_t width,
                                    SmallVectorImpl<Value> &descriptors,
                                    Operation *loop = nullptr,
                                    DenseSet<Value> *island = nullptr) {
  SmallVector<triton::DotOp> dots;
  SmallVector<triton::StoreOp> stores;
  collectSplitIsland(root, dots, stores, loop, island);
  // 岛里没有任何 dot 意味着没有可拆的 PV 累加器。
  if (dots.empty())
    return false;
  for (triton::DotOp dot : dots) {
    auto load = dot.getB().getDefiningOp<triton::LoadOp>();
    if (!load || !descriptorHalvesAtWidth(load.getPtr(), width))
      return false;
    descriptors.push_back(load.getPtr());
  }
  for (triton::StoreOp store : stores) {
    if (!descriptorHalvesAtWidth(store.getPtr(), width))
      return false;
    descriptors.push_back(store.getPtr());
  }
  return true;
}

// 找出超出字节预算且使用范围可安全拆半的累加器，输出需要拆分的循环进位索引
// 与要拆半的描述符。
bool RowLiftBuilder::planWideAccumulatorSplits(
    scf::ForOp old, SmallVectorImpl<int64_t> &indices) {
  // 能力关闭时不做任何拆分规划，走历史的全宽单值路径。
  if (!features.splitWideAccumulator)
    return true;

  for (auto [index, arg] :
       llvm::enumerate(old.getRegionIterArgs())) {
    // 提升后的类型（行维在最前面），其字节数就是提升后的大小。
    auto lifted = getLiftedType(arg.getType());
    const int64_t budget = candidate.splitValueBudgetBytes;
    const int64_t bytes = getTensorBytes(lifted);
    if (bytes <= budget)
      continue;
    // 标量进位值或语义上不拆的值（rank < 2）在字节预算内本就通不过上一条，
    // 这里再判一次以保持历史行为。
    // A scalar-carried or semantically unsplit value (rank < 2) never exceeds
    // the budget above; the check is kept so the historical path is unchanged.
    if (lifted.getRank() < 2)
      continue;

    // 拆半后必须真的落回预算之内，且最后一维能均分成两个连续半宽。
    // Halving must actually bring the value under the budget, and the width
    // must be splittable into two equal contiguous halves.
    if (bytes > 2 * budget)
      return false;
    int64_t width = lifted.getShape().back();
    if (width < 2 || width % 2 != 0)
      return false;

    // 只有「累加器岛」内的值才能拆：它的所有使用者都必须能按半宽重建，且
    // 岛里的 dot/store 都必须能从最后一维同宽的描述符结构拆半。
    bool sawDot = false;
    if (!isSplitFriendlyValue(arg, old, sawDot) || !sawDot)
      return false;
    SmallVector<Value> descriptors;
    if (!describeSplitDescriptor(arg, width, descriptors, old.getOperation(),
                                 &plannedSplitValues))
      return false;
    for (Value descriptor : descriptors)
      splitDescriptors.try_emplace(descriptor, SplitDescriptor{});
    indices.push_back(static_cast<int64_t>(index));
  }
  LLVM_DEBUG(llvm::dbgs()
             << "row-lift: planned " << indices.size() << " split indices, "
             << splitDescriptors.size() << " descriptors\n");
  return true;
}

// 判断值的所有使用是否都留在累加器岛内，并用 sawDot 确认该岛确实由 dot 供给。
//
// 累加器岛的合法组成：dot 的 C 操作数、纯算术的中间结果、store 的值、以及
// 送回本循环的 yield。任何越界使用者都会导致拆分失败，因为半宽值无法以
// 全宽语义继续流动。
static bool isSplitFriendlyValue(Value value, Operation *loop,
                                 bool &sawDot) {
  for (OpOperand &use : value.getUses()) {
    Operation *user = use.getOwner();
    if (auto dot = dyn_cast<triton::DotOp>(user)) {
      // 只拆 dot 的累加器操作数；拆 A 或 B 会连带缩小收缩维，改变归约语义。
      // Only the accumulator operand of the dot is split; halving A or B would
      // also halve the contraction and change the reduction.
      if (use.get() != dot.getC())
        return false;
      sawDot = true;
      continue;
    }
    if (auto store = dyn_cast<triton::StoreOp>(user)) {
      // 只允许作为被写回的值（O / lse），不能作为指针或掩码。
      if (use.get() != store.getValue())
        return false;
      continue;
    }
    if (isa<scf::YieldOp>(user)) {
      // yield 必须回到本循环的背边；也可以是 scf.if 分支的 yield，此时值变成
      // 对应的 if 结果，继续沿该结果判定。
      // A yield has to feed this loop's back edge; an scf.if branch yield is
      // also allowed, in which case the value becomes the matching if result and
      // the check continues on that result.
      if (user->getParentOp() == loop)
        continue;
      Value result = ifResultForYield(use);
      if (!result || !isSplitFriendlyValue(result, loop, sawDot))
        return false;
      continue;
    }
    // 岛上只允许逐元素算术修改值，并且算术结果本身也必须继续留在岛上。
    // The island may only be reshaped by elementwise arithmetic, and each
    // result of that arithmetic must itself stay inside the island.
    if (user->getNumResults() != 1 || user->getNumRegions() != 0)
      return false;
    StringRef dialect = user->getName().getDialectNamespace();
    if (dialect != "arith" && dialect != "math")
      return false;
    if (!isSplitFriendlyValue(user->getResult(0), loop, sawDot))
      return false;
  }
  return true;
}

// 拆宽能力的匹配判据：把单行字节数乘以每个 program 的行数，超过单值预算的
// 进位值必须能对半拆开（不超过两倍预算、末维为偶数、累加器岛合法且可由
// 最后一维同宽的描述符结构拆半）。规则 matcher 用它提前拒绝拆不开的形态；
// planWideAccumulatorSplits 在重写前用同一组规则挑选具体索引。
// Match-time predicate for the split capability.  It multiplies a single row's
// byte count by the row count, and every value above the per-value budget must
// halve cleanly and structurally; planWideAccumulatorSplits picks the actual
// indices with the same rules before the rewrite starts.
bool canSplitWideAccumulators(scf::ForOp loop, int64_t rowsPerProgram,
                              bool enabled, int64_t budgetBytes) {
  // 能力未开启时不做拆分，也没有需要拒绝的形态。
  if (!enabled)
    return true;
  for (Value carried : loop.getRegionIterArgs()) {
    auto tensor = dyn_cast<RankedTensorType>(carried.getType());
    // 标量进位值或语义上不拆的值（rank < 2）继续走历史单值路径。
    if (!tensor || tensor.getRank() < 2)
      continue;
    int64_t perRow = getTensorBytes(tensor);
    if (perRow <= 0)
      continue;
    int64_t bytes;
    if (__builtin_mul_overflow(perRow, rowsPerProgram, &bytes) ||
        bytes <= budgetBytes)
      continue;
    // 拆半后必须真的落回预算之内，且最后一维能均分成两个连续半宽。
    // Halving must bring the value under the budget, and the width must be
    // splittable into two equal contiguous halves.
    if (bytes > 2 * budgetBytes)
      return false;
    int64_t width = tensor.getShape().back();
    if (width < 2 || width % 2 != 0)
      return false;
    // 只有「累加器岛」内的值才能拆，且该岛确实由 dot 供给、能从结构上拆半。
    bool sawDot = false;
    if (!isSplitFriendlyValue(carried, loop.getOperation(), sawDot) || !sawDot)
      return false;
    SmallVector<Value> descriptors;
    if (!describeSplitDescriptor(carried, width, descriptors, loop.getOperation()))
      return false;
  }
  return true;
}


// 沿最后一维把值拆成两个连续半宽值。
bool RowLiftBuilder::splitLastDimension(Value value, SplitValue &halves) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type || type.getRank() < 2)
    return false;
  int64_t width = type.getShape().back();
  if (width < 2 || width % 2 != 0)
    return false;

  // 半宽形状：最后一维减半。
  SmallVector<int64_t> halfShape(type.getShape().begin(),
                                 type.getShape().end() - 1);
  halfShape.push_back(width / 2);
  auto halfType = RankedTensorType::get(halfShape, type.getElementType());

  // 第一步：reshape 成 [.., 2, W/2]，使两个半宽在内存中连续。
  // [.., W] -> [.., 2, W/2] keeps both halves contiguous, and the trailing
  // transpose makes the size-2 dimension the one tt.split removes.
  SmallVector<int64_t> pairShape(halfShape);
  pairShape.pop_back();
  pairShape.push_back(2);
  pairShape.push_back(width / 2);
  Value paired = reshape(value, pairShape);

  // 第二步：转置把大小为 2 的维度移到最后一维，满足 tt.split 的格式要求。
  SmallVector<int64_t> transposedShape(halfShape);
  transposedShape.push_back(2);
  SmallVector<int32_t> order;
  for (int64_t dim = 0; dim + 1 < type.getRank(); ++dim)
    order.push_back(static_cast<int32_t>(dim));
  order.push_back(static_cast<int32_t>(type.getRank()));
  order.push_back(static_cast<int32_t>(type.getRank() - 1));
  Value transposed = rewriter.create<triton::TransOp>(
      value.getLoc(), RankedTensorType::get(transposedShape,
                                            type.getElementType()),
      paired, order);

  // 第三步：tt.split 沿最后一维切成两个半宽值。
  auto split = rewriter.create<triton::SplitOp>(
      value.getLoc(), halfType, halfType, transposed);
  halves = {split.getOutLHS(), split.getOutRHS()};
  return true;
}

// 判断 `value` 能否尾部广播到 `target`，且沿最后一维的取值不变：源的秩更小
// （末维是补齐出来的单位维），或源末维就是 1。只有这种形态才能用同一个半宽
// 值同时代表两半。
// True when `value` can broadcast into `target` by trailing dimensions and its
// value is identical along the last axis: either the source ranks lower (the
// last axis comes from a padded unit dimension) or its last extent is one. Only
// such shapes may be represented by one half-width value shared by both halves.
static bool collapseAcrossLastDim(Value value, RankedTensorType target) {
  auto src = dyn_cast<RankedTensorType>(value.getType());
  if (!src || src.getRank() > target.getRank())
    return false;
  if (!(src.getRank() < target.getRank() || src.getShape().back() == 1))
    return false;
  for (int64_t dim = 0; dim < src.getRank(); ++dim)
    if (src.getShape()[dim] != 1 &&
        src.getShape()[dim] != target.getShape()[dim])
      return false;
  return true;
}

// 拆进位初值：splat 常量直接生成半宽常量，其余形态退回 reshape + split。
// Splits an initial value: a splat constant yields a half-width constant directly
// and anything else falls back to reshape + split.
bool RowLiftBuilder::splitInitialValue(Value value, SplitValue &halves) {
  auto tensor = dyn_cast<RankedTensorType>(value.getType());
  if (!tensor || tensor.getRank() < 2)
    return false;
  auto lifted = getLiftedType(value.getType());
  int64_t width = lifted.getShape().back();
  if (width < 2 || width % 2 != 0)
    return false;
  SmallVector<int64_t> halfShape(lifted.getShape());
  halfShape.back() /= 2;
  auto halfType = RankedTensorType::get(halfShape, lifted.getElementType());

  // 取值在各列相同的初值可以结构折叠成同一个半宽值。
  if (Value half = rebuildHalfOperand(value, halfType)) {
    halves = {half, half};
    return true;
  }
  // 其余形态没有半宽来源，只能先补行维再 reshape + split。
  return splitLastDimension(ensureRowDimension(value), halves);
}

// 把一个整宽操作数按半宽重建。只有「各列取值相同」的形态（splat 常量、标量
// splat、沿最后一维不变的广播、只插入单位维的 expand_dims）可以这样折叠，重建
// 后同一个半宽值同时供两半使用。返回空 Value 表示无法结构折叠。
// Rebuild a full-width operand at half width.  Only shapes whose value is
// identical across the columns (a splat constant, a scalar splat, a broadcast
// that is invariant along the last axis, an expand_dims that only inserts unit
// dimensions) collapse this way, and the single result is shared by both halves.
// A null Value means the operand cannot be collapsed.
Value RowLiftBuilder::rebuildHalfOperand(Value original,
                                         RankedTensorType halfType) {
  // 已在拆分表里的值由调用方提供两半，这里不猜。
  if (splitValues.contains(original))
    return Value();
  // 映射值本身已是半宽（同一操作数已被重建过）时直接复用。
  if (valueMapping.contains(original) &&
      valueMapping.lookup(original).getType() == halfType)
    return valueMapping.lookup(original);

  Location loc = original.getLoc();
  // splat 常量：按半宽形状重新生成同一个常量。
  if (auto cst = original.getDefiningOp<arith::ConstantOp>()) {
    auto dense = dyn_cast<DenseElementsAttr>(cst.getValue());
    if (!dense || !dense.isSplat())
      return Value();
    Type element = halfType.getElementType();
    if (isa<FloatType>(element))
      return rewriter.create<arith::ConstantOp>(
          loc, DenseElementsAttr::get(halfType, dense.getSplatValue<APFloat>()));
    if (isa<IntegerType>(element))
      return rewriter.create<arith::ConstantOp>(
          loc, DenseElementsAttr::get(halfType, dense.getSplatValue<APInt>()));
    return Value();
  }
  // 标量 splat：提升后源是 [R] 向量，可以广播到任意宽度。
  if (auto splat = original.getDefiningOp<triton::SplatOp>()) {
    Value src = ensureRowDimension(splat.getSrc());
    return collapseAcrossLastDim(src, halfType)
               ? broadcastRowValue(src, halfType)
               : Value();
  }
  // 广播：源自身必须沿最后一维退化，两半才能共享同一个值。
  if (auto bcast = original.getDefiningOp<triton::BroadcastOp>()) {
    auto srcType = dyn_cast<RankedTensorType>(bcast.getSrc().getType());
    auto dstType = dyn_cast<RankedTensorType>(bcast.getType());
    if (!srcType || !dstType)
      return Value();
    if (!(srcType.getRank() < dstType.getRank() ||
          srcType.getShape().back() == 1))
      return Value();
    Value src = ensureRowDimension(bcast.getSrc());
    return collapseAcrossLastDim(src, halfType)
               ? broadcastRowValue(src, halfType)
               : Value();
  }
  // expand_dims 只插入单位维，取值沿任何轴都不变。
  if (auto expand = original.getDefiningOp<triton::ExpandDimsOp>()) {
    Value src = ensureRowDimension(expand.getSrc());
    return collapseAcrossLastDim(src, halfType)
               ? broadcastRowValue(src, halfType)
               : Value();
  }
  return Value();
}

// 按操作种类分派拆分累加器的重建：dot、store 或逐元素算术。
bool RowLiftBuilder::rewriteSplitOp(Operation *op) {
  // 只有该能力开启时才可能存在半宽值，否则拒绝以免误用。
  // The capability is the only reason a value is ever represented by halves.
  if (!features.splitWideAccumulator)
    return false;
  if (auto dot = dyn_cast<triton::DotOp>(op))
    return rewriteSplitDot(dot);
  if (auto store = dyn_cast<triton::StoreOp>(op))
    return rewriteSplitStore(store);
  return rewriteSplitElementwise(op);
}

// 把 B 与累加器各拆成两半，分别重建一次 dot。
bool RowLiftBuilder::rewriteSplitDot(triton::DotOp dot) {
  auto accumulator = splitValues.find(dot.getC());
  if (accumulator == splitValues.end())
    return false;
  // A 若已被拆半，会连带缩小收缩维，语义不再等价。
  // A split A operand would also halve the contraction dimension.
  if (splitValues.contains(dot.getA()))
    return false;
  // B 必须是被拆半的访存结果；结构拆半不在值层面重组，因此没有两半就没有
  // 半宽来源。
  // B has to be a split memory result.  The structural split does not restructure
  // values, so without two halves there is no half-width source.
  auto b = splitValues.find(dot.getB());
  if (b == splitValues.end())
    return false;
  if (!hasRowDimension(dot.getA()) || hasRowDimension(dot.getB()))
    return false;

  // 累加器的两个半宽必须是同类型的 [R,G,W/2] 三维值。
  auto accumulatorType =
      dyn_cast<RankedTensorType>(accumulator->second.first.getType());
  if (!accumulatorType || accumulatorType.getRank() != 3)
    return false;
  if (accumulator->second.second.getType() != accumulatorType)
    return false;

  // A 展平成 [R*G,K]；B 的两半必须是同形的 [K,W/2]，末维与累加器半宽一致。
  Value a = flattenDotRows(ensureRowDimension(dot.getA()));
  auto bType = dyn_cast<RankedTensorType>(b->second.first.getType());
  if (!bType || bType.getRank() != 2 || b->second.second.getType() != bType ||
      bType.getShape().back() != accumulatorType.getShape().back())
    return false;

  int64_t rows = accumulatorType.getShape()[0];
  int64_t flatRows = rows * accumulatorType.getShape()[1];
  auto flatType = RankedTensorType::get(
      {flatRows, accumulatorType.getShape()[2]},
      accumulatorType.getElementType());

  // 对每个半宽各发一次 dot：A 与 B 的对应半宽、以及累加器的对应半宽。
  SmallVector<Value> operands0{a, b->second.first,
                               flattenDotRows(accumulator->second.first)};
  SmallVector<Value> operands1{a, b->second.second,
                               flattenDotRows(accumulator->second.second)};
  SmallVector<Value> halves;
  for (const SmallVector<Value> &operands : {operands0, operands1}) {
    Operation *created = rewriter.create(
        dot.getLoc(), dot->getName().getIdentifier(), operands,
        TypeRange{flatType}, dot->getAttrs());
    halves.push_back(
        reshape(created->getResult(0), accumulatorType.getShape()));
  }
  // 结果仍以半宽对的形式登记，后续消费者继续走拆分感知路径。
  splitValues[dot.getResult()] = {halves[0], halves[1]};
  return true;
}

// 把纯逐元素算术按半宽重建两次，所有张量操作数都拆成对应的半宽。
bool RowLiftBuilder::rewriteSplitElementwise(Operation *op) {
  // 只接受单结果、无 region 的纯算术指令。
  if (op->getNumResults() != 1 || op->getNumRegions() != 0)
    return false;
  StringRef dialect = op->getName().getDialectNamespace();
  if (dialect != "arith" && dialect != "math")
    return false;
  auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!resultType || resultType.getRank() < 2)
    return false;

  // 结果的提升类型必须能均分，半宽形状即最后一维减半。
  auto lifted = getLiftedType(resultType);
  int64_t width = lifted.getShape().back();
  if (width < 2 || width % 2 != 0)
    return false;
  SmallVector<int64_t> halfShape(lifted.getShape());
  halfShape.back() /= 2;
  auto halfType = RankedTensorType::get(halfShape, lifted.getElementType());

  // 逐个操作数准备两个半宽副本：标量两侧共用，已拆半的值直接取用，整宽
  // 张量则折叠成半宽值（各列相同的形态才允许折叠）。元素类型逐操作数计算
  // （例如 truncf 的输入是 f32、输出是 f16），不能只按结果类型判断。
  // Every tensor operand must already be split or collapse to the half width;
  // scalars are shared by both halves.  The element type is derived per operand
  // (a truncf takes f32 and produces f16), so the result type alone cannot decide
  // the operand half type.
  SmallVector<std::pair<Value, Value>> operands;
  for (Value operand : op->getOperands()) {
    if (!isa<RankedTensorType>(operand.getType())) {
      operands.push_back({operand, operand});
      continue;
    }
    auto operandLifted = getLiftedType(cast<RankedTensorType>(operand.getType()));
    if (operandLifted.getRank() < 2 ||
        operandLifted.getShape().back() % 2 != 0)
      return false;
    SmallVector<int64_t> operandHalfShape(operandLifted.getShape());
    operandHalfShape.back() /= 2;
    auto operandHalfType =
        RankedTensorType::get(operandHalfShape, operandLifted.getElementType());
    auto split = splitValues.find(operand);
    if (split != splitValues.end()) {
      if (split->second.first.getType() != operandHalfType ||
          split->second.second.getType() != operandHalfType)
        return false;
      operands.push_back({split->second.first, split->second.second});
      continue;
    }
    // 整宽操作数只有在「各列取值相同」时才能折叠成一个半宽值；否则两半
    // 取值不同，结构拆分无法表达。
    Value half = rebuildHalfOperand(operand, operandHalfType);
    if (!half)
      return false;
    operands.push_back({half, half});
  }

  // 对两个半宽各重建一次同一条算术指令。
  SmallVector<Value> results;
  for (bool first : {true, false}) {
    SmallVector<Value> halfOperands;
    for (auto &pair : operands)
      halfOperands.push_back(first ? pair.first : pair.second);
    results.push_back(rewriter.create(op->getLoc(),
                                      op->getName().getIdentifier(),
                                      halfOperands, TypeRange{halfType},
                                      op->getAttrs())
                          ->getResult(0));
  }
  splitValues[op->getResult(0)] = {results[0], results[1]};
  return true;
}

// 把半宽指针、半宽值与掩码各发出一次写回。结构拆半不重组值：两个半宽值
// 分别写进各自半宽的指针，没有 join/trans/reshape。
// Emit one half-width store per half.  The structural split never restructures
// values: each half is written through its own half-width pointer, with no
// join/trans/reshape.
bool RowLiftBuilder::rewriteSplitStore(triton::StoreOp store) {
  auto fail = [](const char *why) {
    LLVM_DEBUG(llvm::dbgs() << "split-store: fail " << why << "\n");
    return false;
  };
  auto valueIt = splitValues.find(store.getValue());
  if (valueIt == splitValues.end())
    return fail("value not split");
  auto descriptorIt = splitDescriptors.find(store.getPtr());
  if (descriptorIt == splitDescriptors.end())
    return fail("descriptor not planned");
  SplitDescriptor &descriptor = descriptorIt->second;

  // 两个半宽值必须同形；后续再按半宽指针逐个校验。
  auto valueType = dyn_cast<RankedTensorType>(valueIt->second.first.getType());
  if (!valueType || valueIt->second.second.getType() != valueType)
    return fail("value halves not same type");

  Location loc = store.getLoc();
  // 带掩码写回需要 maskedStore 能力，与全宽路径一致。
  if (store.getMask() && !features.maskedStore)
    return fail("masked store unsupported");

  if (descriptor.expanded) {
    // 展开档：两半各有一份指针张量，边界检查转成各自的半宽掩码，写回不再
    // 携带边界检查。
    ArrayRef<int32_t> boundaryCheck = store.getBoundaryCheck();
    const LiftedDescriptor *halves[] = {&descriptor.first, &descriptor.second};
    for (const LiftedDescriptor *half : halves) {
      if (!half->pointer)
        return fail("expanded half pointer missing");
      for (int32_t axis : boundaryCheck)
        if (axis < 0 || axis >= static_cast<int32_t>(half->tileShape.size()))
          return fail("boundary axis out of tile rank");
    }
    for (auto [index, half] : llvm::enumerate(halves)) {
      auto pointer = half->pointer;
      auto pointerType = cast<RankedTensorType>(pointer.getType());
      if (pointerType.getShape() != valueType.getShape())
        return fail("expanded pointer shape != value shape");
      auto element = dyn_cast<triton::PointerType>(pointerType.getElementType());
      if (!element || element.getPointeeType() != valueType.getElementType())
        return fail("expanded pointer element mismatch");
      auto indexType =
          RankedTensorType::get(pointerType.getShape(), rewriter.getI64Type());
      Value boundary =
          buildDescriptorMask(*half, store.getBoundaryCheck(), indexType);
      // 原始掩码是整宽的，先按半宽折叠，再与边界掩码、有效行掩码合并。
      Value original;
      if (store.getMask()) {
        auto maskType =
            RankedTensorType::get(pointerType.getShape(), rewriter.getI1Type());
        original = rebuildHalfOperand(store.getMask(), maskType);
        if (!original)
          return fail("mask cannot fold to half width");
      }
      bool ok = true;
      Value mask = mergeMasks(
          pointer, Value(),
          {boundary, candidate.runtimeGrid ? validRows : Value(), original},
          ok);
      if (!ok)
        return fail("mask merge failed");
      Value value =
          index == 0 ? valueIt->second.first : valueIt->second.second;
      rewriter.create<triton::StoreOp>(loc, pointer, value, mask,
                                       ArrayRef<int32_t>{}, store.getCache(),
                                       store.getEvict());
    }
    return true;
  }

  // 未展开档：两半是半宽块指针，边界检查仍由块指针自己承担，因此不接受
  // 掩码操作数（整宽掩码没有对应的半宽来源）。
  if (!descriptor.firstPointer || !descriptor.secondPointer)
    return fail("block pointers missing");
  if (store.getMask())
    return fail("masked store unsupported");
  Value pointers[] = {descriptor.firstPointer, descriptor.secondPointer};
  for (auto [index, pointer] : llvm::enumerate(pointers)) {
    auto pointerType = dyn_cast<triton::PointerType>(pointer.getType());
    if (!pointerType || pointerType.getPointeeType() != valueType)
      return fail("block pointer pointee != value type");
    Value value = index == 0 ? valueIt->second.first : valueIt->second.second;
    rewriter.create<triton::StoreOp>(loc, pointer, value, Value(),
                                     store.getBoundaryCheck(), store.getCache(),
                                     store.getEvict());
  }
  return true;
}

// 在类型最前面插入行维 [R]，标量则变成 [R] 张量。
//   tensor<M,N> -> tensor<R,M,N>
//   i32         -> tensor<Rxi32>
// 行提升后的所有值都要经过这里，行维始终是 0 号轴。
RankedTensorType RowLiftBuilder::getLiftedType(Type old) {
  SmallVector<int64_t> shape{candidate.rowsPerProgram};
  Type elem = old;
  if (auto tensor = dyn_cast<RankedTensorType>(old)) {
    shape.append(tensor.getShape().begin(), tensor.getShape().end());
    elem = tensor.getElementType();
  }
  return RankedTensorType::get(shape, elem);
}

// 查原始值对应的已重建值。
Value RowLiftBuilder::getMappedValue(Value old) {
  return valueMapping.lookup(old);
}

// 判断值是否已提升：重建后的类型与原始类型不同即带行维。
bool RowLiftBuilder::hasRowDimension(Value old) {
  // 已展开的块指针故意没有描述符映射，因此按「未提升」处理其描述符值。
  // An expanded block pointer intentionally has no mapped descriptor.
  if (!valueMapping.contains(old))
    return false;
  return valueMapping.lookup(old).getType() != old.getType();
}

// 返回带行维的值：标量做 splat，张量扩维后广播，块指针保持原样。
Value RowLiftBuilder::ensureRowDimension(Value old) {
  Value value = valueMapping.contains(old) ? valueMapping.lookup(old) : Value();
  // 块指针要么在使用点被展开，要么整体保留，绝不能 splat 成描述符张量。
  // A block pointer is either expanded at its use site or kept whole, never
  // splatted into a tensor of descriptors.
  if (isTensorPointer(old.getType()))
    return value;
  // 已经是提升值：直接返回，避免重复扩维。
  if (hasRowDimension(old))
    return value;

  // 标量：splat 成 [R] 张量。
  if (!isa<RankedTensorType>(old.getType()))
    return rewriter.create<triton::SplatOp>(
        old.getLoc(), getLiftedType(old.getType()), value);

  // 张量：先在 0 号轴插入单位维，再广播到 [R]+原形状。
  Value expanded =
      rewriter.create<triton::ExpandDimsOp>(old.getLoc(), value, 0);
  return rewriter.create<triton::BroadcastOp>(
      old.getLoc(), getLiftedType(old.getType()), expanded);
}

// 生成一次保持元素顺序的 reshape。
Value RowLiftBuilder::reshape(Value v, ArrayRef<int64_t> shape) {
  auto t = cast<RankedTensorType>(v.getType());
  return rewriter.create<triton::ReshapeOp>(
      v.getLoc(), RankedTensorType::get(shape, t.getElementType()), v, false);
}

// 把 [R,M,K] 展平成 [R*M,K]，让普通 tt.dot 可以处理多行。
Value RowLiftBuilder::flattenDotRows(Value v) {
  auto t = cast<RankedTensorType>(v.getType());
  return reshape(v, {t.getShape()[0] * t.getShape()[1], t.getShape()[2]});
}

// 补齐尾部单位维后广播到目标形状。
Value RowLiftBuilder::broadcastRowValue(Value v, RankedTensorType target) {
  while (cast<RankedTensorType>(v.getType()).getRank() < target.getRank())
    v = rewriter.create<triton::ExpandDimsOp>(
        v.getLoc(), v, cast<RankedTensorType>(v.getType()).getRank());
  return rewriter.create<triton::BroadcastOp>(v.getLoc(), target, v);
}

// 在 axis+1 上重建 tt.reduce，并原样复用 combine region。
// 行维插在最前面后，原 axis 对应的轴号全部后移一位。
bool RowLiftBuilder::rewriteReduce(triton::ReduceOp reduce) {
  // 能力未开启或没有输入时保持失败关闭。
  if (!features.reduce || reduce.getSrcs().empty())
    return false;

  // 所有源值补行维。
  SmallVector<Value> sources;
  for (Value src : reduce.getSrcs())
    sources.push_back(ensureRowDimension(src));

  // 归约轴从 axis 变为 axis+1，逐行独立归约。
  auto replacement = rewriter.create<triton::ReduceOp>(
      reduce.getLoc(), sources, reduce.getAxis() + 1);
  // combine region 只关心元素类型，与形状无关，因此可以整段原样复用。
  // The combine region sees element types only, so it is reused verbatim.
  rewriter.cloneRegionBefore(reduce.getCombineOp(),
                             replacement.getCombineOp(),
                             replacement.getCombineOp().end());
  valueMapping.map(reduce.getResults(), replacement.getResults());
  return true;
}

// 把值广播到目标形状：标量 splat，行提升值补齐缺失的单位维。
Value RowLiftBuilder::broadcastFullValue(Value v, RankedTensorType target) {
  // 标量直接 splat；已带行维的 [R,...] 值先补 tile 方向的单位维再广播。
  // A scalar becomes a splat; a row-lifted [R x ...] value gains the tile
  // dimensions behind its row dimension before broadcasting.
  if (!isa<RankedTensorType>(v.getType()))
    return rewriter.create<triton::SplatOp>(v.getLoc(), target, v);
  if (v.getType() == target)
    return v;
  return broadcastRowValue(v, target);
}

// 生成沿指定 tile 轴递增的 i64 索引网格并广播到目标形状。
Value RowLiftBuilder::expandTileRange(Location loc, int64_t axis,
                                      int64_t extent, RankedTensorType target) {
  // 先生成 i32 的 [0, extent) 区间，再扩展到 i64（地址运算统一用 i64）。
  auto rangeType =
      RankedTensorType::get({extent}, rewriter.getI32Type());
  Value range = rewriter.create<triton::MakeRangeOp>(
      loc, rangeType, 0, static_cast<int32_t>(extent));
  range = rewriter.create<arith::ExtSIOp>(
      loc, RankedTensorType::get({extent}, rewriter.getI64Type()), range);

  // 除目标轴外，每个位置插入单位维，使该轴落在行维之后（轴号 = axis+1）；
  // 按升序插入可以保持各维 extent 与原 block 形状一致。
  // Insert a unit dimension at every position except the tile axis, which sits
  // directly behind the leading row dimension. Inserting in ascending order
  // keeps the extents the same as the original block shape.
  int64_t kept = axis + 1;
  for (int64_t position = 0; position < target.getRank(); ++position)
    if (position != kept)
      range = rewriter.create<triton::ExpandDimsOp>(loc, range, position);
  return rewriter.create<triton::BroadcastOp>(loc, target, range);
}

// 块指针能否展开成显式指针张量的完整判据。规则的 matcher 与
// rewriteTensorPointer 共用，避免「匹配通过、展开失败」的两处漂移。
// The complete predicate for descriptor expansion, shared by the rule's matcher
// and rewriteTensorPointer.
bool canExpandTensorPointer(triton::MakeTensorPtrOp makePtr, bool enabled) {
  // 能力未开启时保持原有失败关闭路径。
  if (!enabled)
    return false;
  auto pointerType = dyn_cast<triton::PointerType>(makePtr.getType());
  if (!pointerType)
    return false;
  auto blockType = dyn_cast<RankedTensorType>(pointerType.getPointeeType());
  if (!blockType || blockType.getRank() == 0)
    return false;
  // 显式指针张量要求每个 tile 维都有静态长度。
  // The explicit pointer tensor needs a static extent per tile dimension.
  if (llvm::any_of(blockType.getShape(),
                   [](int64_t extent) { return extent <= 0; }))
    return false;
  // offset/shape/stride 都要能扩到 i64 参与地址运算；非整数操作数无法展开。
  // Every index operand must widen to i64 for the address arithmetic.
  auto isInteger = [](Value value) {
    Type type = value.getType();
    if (auto tensor = dyn_cast<RankedTensorType>(type))
      type = tensor.getElementType();
    return isa<IntegerType>(type);
  };
  return llvm::all_of(makePtr.getOffsets(), isInteger) &&
         llvm::all_of(makePtr.getShape(), isInteger) &&
         llvm::all_of(makePtr.getStrides(), isInteger);
}

// 把块指针描述符展开成显式 [R]+tile 指针张量，并记录形状与偏移供掩码使用。
//
// 展开过程就是按原语义显式计算每个 tile 元素的地址：
//   addr[i0..ik] = base + Σ (offset[d] + i_d) * stride[d]
// 其中 i_d 是该 tile 维的下标。展开后原来的 boundaryCheck 变成使用点的
// 显式掩码，块指针本身不再出现在结果 IR 中。
bool RowLiftBuilder::rewriteTensorPointer(triton::MakeTensorPtrOp makePtr) {
  // 能力未开启时保持原有失败关闭路径。
  if (!features.tensorPointer)
    return false;

  // 规划为拆半的描述符先按半宽重建两个描述符（或两个半宽块指针）。
  auto split = splitDescriptors.find(makePtr.getResult());
  if (split != splitDescriptors.end())
    return rewriteSplitTensorPointer(makePtr, split->second);

  auto pointerType = cast<triton::PointerType>(makePtr.getType());
  auto blockType = dyn_cast<RankedTensorType>(pointerType.getPointeeType());
  if (!blockType || blockType.getRank() == 0)
    return false;

  LiftedDescriptor descriptor;
  if (!expandTensorPointer(makePtr, blockType.getShape().back(), Value(),
                           descriptor))
    return false;
  liftedDescriptors[makePtr.getResult()] = descriptor;
  return true;
}

// 展开一个描述符。`lastExtent` 是最后一维的 tile 宽度（拆半时为半宽），
// `lastOffsetOverride` 非空时替换最后一维的偏移（第二半整体后移半宽）。
// Expand one descriptor.  `lastExtent` is the trailing tile extent (half the
// width when splitting) and a non-null `lastOffsetOverride` replaces the trailing
// offset, which is how the second half is shifted by one half width.
bool RowLiftBuilder::expandTensorPointer(triton::MakeTensorPtrOp makePtr,
                                         int64_t lastExtent,
                                         Value lastOffsetOverride,
                                         LiftedDescriptor &descriptor) {
  auto pointerType = cast<triton::PointerType>(makePtr.getType());
  auto blockType = dyn_cast<RankedTensorType>(pointerType.getPointeeType());
  if (!blockType || blockType.getRank() == 0)
    return false;
  SmallVector<int64_t> tileShape(blockType.getShape());
  if (lastExtent > 0)
    tileShape.back() = lastExtent;
  // 显式指针张量要求每个 tile 维都有静态长度。
  // The explicit pointer tensor needs a static extent per tile dimension.
  if (llvm::any_of(tileShape, [](int64_t extent) { return extent <= 0; }))
    return false;

  // 提升后的形状 = [R] + 原 tile 形状；索引类型与指针张量类型都按它构造。
  Location loc = makePtr.getLoc();
  auto i64Type = rewriter.getI64Type();
  auto elementPointer =
      triton::PointerType::get(blockType.getElementType(),
                               pointerType.getAddressSpace());
  SmallVector<int64_t> fullShape{candidate.rowsPerProgram};
  fullShape.append(tileShape.begin(), tileShape.end());
  auto indexType = RankedTensorType::get(fullShape, i64Type);
  auto pointerTensorType = RankedTensorType::get(fullShape, elementPointer);

  // shape/stride 是 i64、offset 可能是 i32，地址运算前统一扩到 i64；
  // 遇到非整数类型时返回空 Value，由调用方拒绝。
  // shape and strides are i64 while offsets are i32, so the address
  // arithmetic widens every index to i64 first.
  auto widen = [&](Value v) -> Value {
    Type type = v.getType();
    Type element = type;
    if (auto tensor = dyn_cast<RankedTensorType>(type))
      element = tensor.getElementType();
    auto integer = dyn_cast<IntegerType>(element);
    if (!integer)
      return Value();
    if (integer.getWidth() == 64)
      return v;
    if (auto tensor = dyn_cast<RankedTensorType>(type))
      return rewriter.create<arith::ExtSIOp>(
          loc, RankedTensorType::get(tensor.getShape(), i64Type), v);
    return rewriter.create<arith::ExtSIOp>(loc, i64Type, v);
  };

  // 逐维取出 offset/shape/stride，形状与偏移都保存下来，后面把 boundaryCheck
  // 转掩码时要用。每个值都先扩到 i64，地址运算统一位宽。
  SmallVector<Value> offsets, shapes, strides;
  offsets.reserve(tileShape.size());
  shapes.reserve(tileShape.size());
  strides.reserve(tileShape.size());
  const int64_t lastDim = static_cast<int64_t>(tileShape.size()) - 1;
  for (int64_t dim = 0; dim < static_cast<int64_t>(tileShape.size()); ++dim) {
    Value offset =
        (lastOffsetOverride && dim == lastDim)
            ? widen(lastOffsetOverride)
            : widen(getMappedValue(makePtr.getOffsets()[dim]));
    Value shape = widen(getMappedValue(makePtr.getShape()[dim]));
    Value stride = widen(getMappedValue(makePtr.getStrides()[dim]));
    if (!offset || !shape || !stride)
      return false;
    offsets.push_back(offset);
    shapes.push_back(shape);
    strides.push_back(stride);
  }

  // 地址分两段构造，与手工参考实现同构：除最后一维外的贡献
  // (offset + range) * stride 先累加成一个「末维为 1」的基址张量，最后一维的
  // 区间再在完整形状上补加。只有这种形态才会被适配器识别成 strided 子视图；
  // 原来的「一次 addptr + 全秩下标」会被判成逐元素间接访问
  // (indirect_load/store)，后端无法对其切分 UB，直接 ub overflow。
  // The address is built in two stages exactly like the handwritten reference:
  // every axis but the last accumulates (offset + range) * stride into a base
  // tensor whose trailing dim is 1, and the trailing range is added at full
  // rank.  Only this shape is recognized as a strided subview; the flat form
  // degrades to per-element indirect access, which the backend cannot tile.
  SmallVector<int64_t> baseShape(fullShape.begin(), fullShape.end());
  baseShape.back() = 1;
  auto baseIndexType = RankedTensorType::get(baseShape, i64Type);
  auto basePointerType = RankedTensorType::get(baseShape, elementPointer);
  Value baseAddress;
  for (int64_t dim = 0; dim + 1 < static_cast<int64_t>(tileShape.size()); ++dim) {
    Value index = rewriter.create<arith::AddIOp>(
        loc, broadcastFullValue(offsets[dim], baseIndexType),
        expandTileRange(loc, dim, tileShape[dim], baseIndexType));
    Value term = rewriter.create<arith::MulIOp>(
        loc, index, broadcastFullValue(strides[dim], baseIndexType));
    baseAddress =
        baseAddress
            ? Value(rewriter.create<arith::AddIOp>(loc, baseAddress, term))
            : term;
  }

  // 基址先落到「末维为 1」的形状，与该形状的地址相加，再广播到指针张量形状。
  Value base =
      broadcastFullValue(getMappedValue(makePtr.getBase()), basePointerType);
  Value basePointer = baseAddress
                          ? Value(rewriter.create<triton::AddPtrOp>(
                                loc, basePointerType, base, baseAddress))
                          : base;
  Value pointer = broadcastFullValue(basePointer, pointerTensorType);

  // 最后一维的区间（含该轴偏移）在完整形状上直接加在指针上。
  Value lastIndex = rewriter.create<arith::AddIOp>(
      loc, broadcastFullValue(offsets.back(), indexType),
      expandTileRange(loc, lastDim, tileShape.back(), indexType));
  Value lastTerm = rewriter.create<arith::MulIOp>(
      loc, lastIndex, broadcastFullValue(strides.back(), indexType));

  // 登记描述符；最终指针张量供使用点的 load/store 取用。
  descriptor.pointer = rewriter.create<triton::AddPtrOp>(
      loc, pointerTensorType, pointer, lastTerm);
  descriptor.shapes = shapes;
  descriptor.offsets = offsets;
  descriptor.tileShape = tileShape;
  return true;
}

// 把一个整数值整体加一个常量；张量形式先生成同形 splat 常量再相加。
// Shift an integer value by a constant, splatting the constant for a tensor.
static Value shiftIntegerValue(IRRewriter &rewriter, Value base,
                               int64_t amount) {
  if (!base)
    return Value();
  Location loc = base.getLoc();
  Type type = base.getType();
  Type element = type;
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    element = tensor.getElementType();
  auto integer = dyn_cast<IntegerType>(element);
  if (!integer)
    return Value();
  Value shift = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getIntegerAttr(integer, amount));
  if (isa<RankedTensorType>(type))
    shift = rewriter.create<triton::SplatOp>(loc, type, shift);
  return rewriter.create<arith::AddIOp>(loc, base, shift);
}

// 把一个已规划拆半的描述符重建为两个半宽描述符。与行无关的块指针留在原始
// 层级，只产出两个半宽块指针（边界检查仍由它们承担）；与行相关的描述符则
// 两半各展开成 [R]+半宽 tile 的指针张量，第二半的最后一维偏移后移半宽。
// Rebuild a descriptor planned for halving.  A row-invariant block pointer stays
// at its original level and simply yields two half-width block pointers (their
// boundary checks stay on the pointer); a row-dependent descriptor is expanded
// once per half into an [R]+half-tile pointer tensor, with the trailing offset of
// the second half shifted by one half width.
bool RowLiftBuilder::rewriteSplitTensorPointer(
    triton::MakeTensorPtrOp makePtr, SplitDescriptor &descriptor) {
  if (!features.tensorPointer)
    return false;
  auto pointerType = dyn_cast<triton::PointerType>(makePtr.getType());
  auto blockType = pointerType
                       ? dyn_cast<RankedTensorType>(pointerType.getPointeeType())
                       : RankedTensorType();
  if (!blockType || blockType.getRank() == 0)
    return false;
  auto tileShape = blockType.getShape();
  int64_t width = tileShape.back();
  if (width < 2 || width % 2 != 0)
    return false;
  int64_t halfWidth = width / 2;

  // 最后一维的偏移是唯一要移动的部分：第二半整体后移半个 tile 宽。
  Value lastOffset = getMappedValue(makePtr.getOffsets().back());
  Value shifted = shiftIntegerValue(rewriter, lastOffset, halfWidth);
  if (!shifted)
    return false;

  // 与行相关与否决定两半是「展开」还是「一对半宽块指针」。
  bool rowDependent = llvm::any_of(makePtr->getOperands(),
                                   [&](Value v) { return hasRowDimension(v); });
  if (rowDependent) {
    if (!expandTensorPointer(makePtr, halfWidth, Value(), descriptor.first) ||
        !expandTensorPointer(makePtr, halfWidth, shifted, descriptor.second))
      return false;
    descriptor.expanded = true;
    return true;
  }

  SmallVector<int64_t> halfTile(tileShape.begin(), tileShape.end());
  halfTile.back() = halfWidth;
  auto halfBlock = RankedTensorType::get(halfTile, blockType.getElementType());
  auto halfPointerType = triton::PointerType::get(
      halfBlock, pointerType.getAddressSpace());

  SmallVector<Value> operands;
  for (Value v : makePtr->getOperands())
    operands.push_back(getMappedValue(v));
  auto build = [&](ArrayRef<Value> ops) {
    return rewriter
        .create(makePtr.getLoc(), makePtr->getName().getIdentifier(), ops,
                TypeRange{halfPointerType}, makePtr->getAttrs())
        ->getResult(0);
  };
  descriptor.firstPointer = build(operands);
  // 第二半必须换成后移半宽的偏移，否则两半描述符完全相同，会被 CSE 合并成
  // 同一份 load，高半累加器就拿到了低半的 V。
  // The second half must use the shifted offset; otherwise both descriptors are
  // identical, CSE folds them into one load, and the high accumulator reads the
  // low V half.
  operands.back() = shifted;
  descriptor.secondPointer = build(operands);
  descriptor.expanded = false;
  return true;
}

// 把 boundaryCheck 指定的各轴越界判断合并成显式 i1 掩码。
// 对每个轴计算 0 <= offset + range(0, extent) < shape，再把各轴结果按位与；
// 没有指定边界检查时返回空 Value。
Value RowLiftBuilder::buildDescriptorMask(
    const LiftedDescriptor &descriptor, ArrayRef<int32_t> boundaryCheck,
    RankedTensorType indexType) {
  if (boundaryCheck.empty())
    return Value();

  Location loc = descriptor.pointer.getLoc();
  Value mask;
  for (int32_t axis : boundaryCheck) {
    // 该轴的逐元素下标网格。
    Value index = rewriter.create<arith::AddIOp>(
        loc, broadcastFullValue(descriptor.offsets[axis], indexType),
        expandTileRange(loc, axis, descriptor.tileShape[axis], indexType));
    Value zero = rewriter.create<triton::SplatOp>(
        loc, indexType,
        rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(0)));
    // 下界：index >= 0；上界：index < shape（shape 来自描述符展开时保存的值）。
    Value lower = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sge, index, zero);
    Value upper = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, index,
        broadcastFullValue(descriptor.shapes[axis], indexType));
    Value axisMask = rewriter.create<arith::AndIOp>(loc, lower, upper);
    // 各轴的越界条件同时成立才算有效。
    mask = mask ? Value(rewriter.create<arith::AndIOp>(loc, mask, axisMask))
                : axisMask;
  }
  return mask;
}

// 用展开后的指针张量重建描述符 load，掩码由边界检查与原始 mask 合并而来。
bool RowLiftBuilder::rewriteDescriptorLoad(
    triton::LoadOp load, const LiftedDescriptor &descriptor) {
  Value result;
  if (!emitDescriptorLoad(load, descriptor, result))
    return false;
  valueMapping.map(load.getResult(), result);
  return true;
}

// 用给定描述符发出一次 load 并把结果交回；结果形状取自指针张量，因此拆半
// 路径按两个半宽描述符各调用一次即可得到两个半宽值。
// Emit one load for the given descriptor and hand back its result.  The result
// shape follows the pointer tensor, so the split path calls this once per half.
bool RowLiftBuilder::emitDescriptorLoad(triton::LoadOp load,
                                        const LiftedDescriptor &descriptor,
                                        Value &result) {
  // 块指针上的 mask 操作数需要与 store 侧相同的行维展开，因此同样受
  // maskedStore 能力控制。
  // A mask operand on a block pointer needs the same row expansion as the
  // store side, so it stays gated behind maskedStore.
  if (load.getMask() && !features.maskedStore)
    return false;
  // 边界检查轴号必须落在 tile 维度内。
  ArrayRef<int32_t> boundaryCheck = load.getBoundaryCheck();
  for (int32_t axis : boundaryCheck)
    if (axis < 0 || axis >= static_cast<int32_t>(descriptor.tileShape.size()))
      return false;

  Location loc = load.getLoc();
  auto pointerType = cast<RankedTensorType>(descriptor.pointer.getType());
  auto resultType = RankedTensorType::get(
      pointerType.getShape(),
      cast<RankedTensorType>(load.getType()).getElementType());
  auto indexType =
      RankedTensorType::get(pointerType.getShape(), rewriter.getI64Type());

  // boundaryCheck 转成显式掩码，并与原始 mask（及运行期有效行掩码）合并。
  bool ok = true;
  Value mask = combineMemoryMask(
      descriptor.pointer, load.getMask(),
      buildDescriptorMask(descriptor, boundaryCheck, indexType), ok);
  if (!ok)
    return false;

  // 有掩码时必须给出 other 值：优先用原 other，其次按 padding 选项生成
  // NaN 或 0，并广播到提升后的结果形状。拆半描述符的结果比提升后的原值窄，
  // 此时只能结构折叠（各列相同的 other），全宽 other 无法表达。
  Value other;
  if (mask) {
    if (load.getOther()) {
      Value lifted = ensureRowDimension(load.getOther());
      if (lifted.getType() == resultType)
        other = lifted;
      else if (getLiftedType(load.getOther().getType()) == resultType)
        other = broadcastRowValue(lifted, resultType);
      else
        other = rebuildHalfOperand(load.getOther(), resultType);
      if (!other)
        return false;
    } else {
      Value padding;
      if (load.getPadding() == triton::PaddingOption::PAD_NAN) {
        auto floatType = dyn_cast<FloatType>(resultType.getElementType());
        if (!floatType)
          return false;
        padding = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getFloatAttr(
                     floatType, APFloat::getNaN(floatType.getFloatSemantics())));
      } else {
        padding = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getZeroAttr(resultType.getElementType()));
      }
      other = rewriter.create<triton::SplatOp>(loc, resultType, padding);
    }
  }

  // 用指针张量发出普通 load；边界检查已转成掩码，因此不再携带。
  result = rewriter
               .create<triton::LoadOp>(
                   loc, descriptor.pointer, mask, other, ArrayRef<int32_t>{},
                   load.getPadding(), load.getCache(), load.getEvict(),
                   load.getIsVolatile())
               .getResult();
  return true;
}

// 用拆半描述符发出两次半宽 load。展开档按两个半宽指针张量各发一次；与行无关
// 的档直接用两个半宽块指针，边界检查仍由块指针承担。
// Emit one half-width load per half of a split descriptor.  The expanded form
// uses the two half-width pointer tensors; the row-invariant form uses the two
// half-width block pointers, whose boundary checks stay on the pointer.
bool RowLiftBuilder::rewriteSplitDescriptorLoad(triton::LoadOp load,
                                                SplitDescriptor &descriptor) {
  if (descriptor.expanded) {
    Value first, second;
    if (!emitDescriptorLoad(load, descriptor.first, first) ||
        !emitDescriptorLoad(load, descriptor.second, second))
      return false;
    splitValues[load.getResult()] = {first, second};
    return true;
  }

  // 半宽块指针自带边界检查，整宽 mask 没有对应的半宽来源，因此直接拒绝。
  if (!descriptor.firstPointer || !descriptor.secondPointer || load.getMask())
    return false;
  SmallVector<Value> halves;
  for (Value pointer : {descriptor.firstPointer, descriptor.secondPointer}) {
    auto pointerType = dyn_cast<triton::PointerType>(pointer.getType());
    auto resultType =
        pointerType ? dyn_cast<RankedTensorType>(pointerType.getPointeeType())
                    : RankedTensorType();
    if (!resultType)
      return false;
    halves.push_back(rewriter
                         .create<triton::LoadOp>(
                             load.getLoc(), pointer, Value(), Value(),
                             load.getBoundaryCheck(), load.getPadding(),
                             load.getCache(), load.getEvict(),
                             load.getIsVolatile())
                         .getResult());
  }
  splitValues[load.getResult()] = {halves[0], halves[1]};
  return true;
}

// 用展开后的指针张量重建描述符 store，并合并边界掩码与原始 mask。
bool RowLiftBuilder::rewriteDescriptorStore(
    triton::StoreOp store, const LiftedDescriptor &descriptor) {
  // O 与 lse 的带掩码写回：mask 操作数属于行维展开的一部分，受
  // maskedStore 能力控制。
  // O and lse are written back under a mask, so the mask operand is part of
  // the row expansion when the capability is enabled.
  if (store.getMask() && !features.maskedStore)
    return false;
  // 边界检查轴号必须落在 tile 维度内。
  ArrayRef<int32_t> boundaryCheck = store.getBoundaryCheck();
  for (int32_t axis : boundaryCheck)
    if (axis < 0 || axis >= static_cast<int32_t>(descriptor.tileShape.size()))
      return false;

  Location loc = store.getLoc();
  auto pointerType = cast<RankedTensorType>(descriptor.pointer.getType());
  auto indexType =
      RankedTensorType::get(pointerType.getShape(), rewriter.getI64Type());

  // 边界掩码 + 原始 mask 合并为一次写回的掩码。
  bool ok = true;
  Value mask = combineMemoryMask(
      descriptor.pointer, store.getMask(),
      buildDescriptorMask(descriptor, boundaryCheck, indexType), ok);
  if (!ok)
    return false;

  // 被写入的值必须与指针张量同形（已补上行维）。
  Value value = ensureRowDimension(store.getValue());
  auto valueType = dyn_cast<RankedTensorType>(value.getType());
  if (!valueType || valueType.getShape() != pointerType.getShape())
    return false;

  // 用指针张量发出 store；边界检查已转成掩码，因此不再携带。
  rewriter.create<triton::StoreOp>(
      loc, descriptor.pointer, value, mask,
      rewriter.getDenseI32ArrayAttr({}), store.getCache(), store.getEvict());
  return true;
}

// scf.if 能否按「所有提升行走同一分支」重建的完整判据：条件必须是标量与行
// 无关，分支各只有一个 block，结果不能是块指针。规则 matcher 与 rewriteIfOp
// 共用，匹配阶段就能拒绝逐行走不同分支的形态。
// The complete predicate for a row-invariant scf.if, shared by the rule's
// matcher and rewriteIfOp.
bool liftsRowInvariantIf(scf::IfOp ifOp, Value pid) {
  // 条件必须是标量 i1，且不依赖查询行；否则各行可能走不同分支。
  // Every lifted row takes the same branch, so the condition stays a scalar and
  // must not depend on the query row.
  Value condition = ifOp.getCondition();
  if (!condition.getType().isInteger(1) || dependsOnValue(condition, pid))
    return false;
  // 每个分支目前只支持单 block 形态。
  if (!llvm::hasSingleElement(ifOp.getThenRegion().getBlocks()) ||
      (!ifOp.getElseRegion().empty() &&
       !llvm::hasSingleElement(ifOp.getElseRegion().getBlocks())))
    return false;
  // 块指针不能作为 scf.if 的结果值。
  return llvm::none_of(ifOp.getResultTypes(), isTensorPointer);
}

// 重建条件与查询行无关的 scf.if，分支结果按提升后的类型重建。
// 因为所有提升行都走同一分支，条件保持标量、不补行维。
bool RowLiftBuilder::rewriteIfOp(scf::IfOp old) {
  // 能力未开启时保持原有失败关闭路径。
  if (!features.loopInvariantIf)
    return false;
  // 条件必须是标量 i1 且未带行维，否则各行可能走不同分支。
  // Every lifted row takes the same branch, so the condition stays a scalar.
  Value condition = old.getCondition();
  if (!condition.getType().isInteger(1) || hasRowDimension(condition))
    return false;
  // 每个分支目前只支持单 block 形态。
  if (!llvm::hasSingleElement(old.getThenRegion().getBlocks()) ||
      (!old.getElseRegion().empty() &&
       !llvm::hasSingleElement(old.getElseRegion().getBlocks())))
    return false;

  // 结果类型统一补行维；块指针不能作为 scf.if 的结果值。被规划为拆半的
  // 结果占两项半宽类型，因此两个分支都必须以半宽对 yield。
  // Every result gains the row dimension, and a result planned for halving
  // occupies two half-width slots, so both branches have to yield its two halves.
  SmallVector<Type> types;
  SmallVector<int64_t> starts;
  SmallVector<bool> split;
  for (auto [index, t] : llvm::enumerate(old.getResultTypes())) {
    // A block pointer cannot be carried by an scf.if result.
    if (isTensorPointer(t))
      return false;
    starts.push_back(static_cast<int64_t>(types.size()));
    auto lifted = getLiftedType(t);
    bool halve = features.splitWideAccumulator &&
                 plannedSplitValues.contains(old.getResult(index));
    if (halve) {
      int64_t width = lifted.getShape().back();
      if (lifted.getRank() < 2 || width < 2 || width % 2 != 0)
        return false;
      SmallVector<int64_t> halfShape(lifted.getShape());
      halfShape.back() /= 2;
      auto halfType =
          RankedTensorType::get(halfShape, lifted.getElementType());
      types.push_back(halfType);
      types.push_back(halfType);
    } else {
      types.push_back(lifted);
    }
    split.push_back(halve);
  }

  // 先建出带空分支的 scf.if，再逐分支重建。
  auto replacement = rewriter.create<scf::IfOp>(
      old.getLoc(), types, valueMapping.lookupOrDefault(condition),
      /*withElseRegion=*/!old.getElseRegion().empty());
  if (!rewriteIfBody(old.getThenRegion(), replacement.getThenRegion(), split) ||
      !rewriteIfBody(old.getElseRegion(), replacement.getElseRegion(), split))
    return false;

  // 分支重建完成后插入点会留在分支 block 内，这里移回 if 之后。
  // Body rewriting leaves the insertion point inside the branch blocks.
  rewriter.setInsertionPointAfter(replacement);
  for (auto [index, result] : llvm::enumerate(old.getResults())) {
    int64_t start = starts[index];
    if (split[index]) {
      splitValues[result] = {replacement.getResult(start),
                             replacement.getResult(start + 1)};
      continue;
    }
    valueMapping.map(result, replacement.getResult(start));
  }
  return true;
}

// 重建 scf.if 的一个分支 region，并生成提升后的 yield。
bool RowLiftBuilder::rewriteIfBody(Region &from, Region &to,
                                   ArrayRef<bool> split) {
  if (from.empty())
    return true;

  // 新 scf.if 已经为每个分支建好一个 block，这里清空其占位内容后重放指令。
  // The new scf.if already owns one block per branch, possibly with a
  // placeholder terminator that this rebuild replaces.
  Block *body = &to.front();
  body->clear();
  rewriter.setInsertionPointToStart(body);
  for (Operation &op : from.front().without_terminator())
    if (!rewrite(&op))
      return false;

  // 重建分支 yield：每个值补行维；已展开的块指针无法作为 yield 值。被规划
  // 拆半的值压入两个半宽值，与 if 结果占用的两项类型一致。
  auto yield = cast<scf::YieldOp>(from.front().getTerminator());
  SmallVector<Value> values;
  for (auto [index, v] : llvm::enumerate(yield.getOperands())) {
    // An expanded block pointer has no value that can be yielded.
    if (liftedDescriptors.contains(v))
      return false;
    if (index < split.size() && split[index]) {
      auto it = splitValues.find(v);
      if (it == splitValues.end())
        return false;
      values.push_back(it->second.first);
      values.push_back(it->second.second);
      continue;
    }
    values.push_back(ensureRowDimension(v));
  }
  rewriter.create<scf::YieldOp>(yield.getLoc(), values);
  return true;
}

} // namespace cfg
} // namespace triton
} // namespace mlir
