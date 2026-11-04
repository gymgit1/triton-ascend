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

// 行提升公共引擎（从 DotRowCoalescing 中抽出的行提升实现，供该规则复用）。
//
// 本文件把「一个 program 只处理一行查询」的函数体重建为「一个 program
// 处理 R 行查询」：所有与查询行相关的值在类型最前面插入行维 [R]，循环用
// 末行的上界作为组内共享 trip count，组内各行的差异由逐行有效掩码表达。
// 文件内函数与既定需求条目的对应关系：
//   1. 公共组件：本文件即从 DotRowCoalescing 抽出的行提升引擎，各规则
//      各自维护 matcher，重建逻辑只保留这一份。
//   2. tt.reduce 与条件与行号无关的 scf.if：rewriteReduce / rewriteIfOp。
//   3. Q、O 的 tensor pointer 展开成指针张量：rewriteTensorPointer，以及
//      描述符 load/store 的重建。
//   8. 「入口生成合并版与回退版」由规则层（Rule / ProgramGridTransform）
//      负责，不属于本文件。
//
// 约定：RowLiftFeatures 中的能力开关全部默认关闭，未置位的能力走原有的
// 失败关闭路径；DotRow 规则不置位任何新开关，因此其输出与基线指令序列
// 逐字节一致。

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

// 读取整数常量；不是常量时返回 nullopt。规则 matcher 与重写路径共用。
std::optional<int64_t> getConstantIntValue(Value v) {
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
//   Bounded   —— 给出了具体数值范围，末行取值可作循环上界；需要按上界算出
//                具体宽度的调用方只接受此态。
// 行提升只需要「末行取值是组内最大」这一单调性，末行上界是在运行期用原表达式
// 重新求值得到的，不依赖证明器算出的数值，所以 Monotone 已经够用。
// A pid-independent term only shifts the chain by a constant, which keeps the
// monotonicity but hides its magnitude (the scalar result of a tt.reduce over a
// loaded tile, for instance).  That case and a range that does not fit the
// expression's bit width are both reported as `Monotone`: row lifting only needs
// the last row's value to be the group maximum, which is re-evaluated from the
// original expression at run time.  Only callers that have to size something
// from the proven `hi` (DotRowCoalescing) ask for `Bounded`.
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
// to `Monotone` instead of failing: row lifting only needs monotonicity, and
// only callers that size something from `hi` (DotRowCoalescing) need the range.
// 把有界证明收窄到表达式位宽；范围放不进该位宽时退化为「单调但无范围」，
// 供行提升使用；需要按上界算出具体宽度的调用方自行要求 Bounded。
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

// 由资源约束反推出每个 program 能处理的查询行数，返回 1 表示不提升。
// R is the smallest upper bound implied by the resource constraints, floored to a
// power of two because the rewrite and its retry paths walk factors by halving.
// A constraint whose input is unknown (zero budget, zero demand, runtime grid)
// does not bound R; when no constraint bounds it, no factor is proven safe and
// lifting is declined.
int64_t selectRowsPerProgram(int64_t grid, int64_t maxDotRows,
                             int64_t maxTensorBytes, int64_t peakLiveBytes,
                             const RowCountPolicy &policy) {
  // 单个提升值的字节预算。
  // Per-value byte budget.
  const int64_t budget = policy.maxLiftedValueBytes;

  // 每条约束都形如 demand * factor <= budget，等价于 factor <= budget / demand，
  // 因此逐因子比较可以折叠成取所有上界的最小值。
  // Every constraint has the form demand * factor <= budget, which is the same
  // as factor <= budget / demand, so the per-factor checks collapse into one
  // minimum over the implied upper bounds.
  std::optional<int64_t> limit;
  auto tighten = [&](int64_t bound) {
    limit = limit ? std::min(*limit, bound) : bound;
  };

  // L0A/L0C 容量：重写按因子放大 lhs 与累加器 tile，预算被因子均摊。预算为 0
  // 表示目标架构容量未知，此时该约束不参与定界。
  // L0A/L0C capacity: the rewrite widens the lhs and accumulator tiles by the
  // factor, so the budget is shared across the factor.  A zero budget means the
  // target capacities are unknown and the constraint does not bound R.
  if (policy.maxL0ABytes > 0 && policy.l0ABudgetBytes > 0)
    tighten(static_cast<int64_t>(policy.l0ABudgetBytes /
                                 static_cast<uint64_t>(policy.maxL0ABytes)));
  if (policy.maxL0CBytes > 0 && policy.l0CBudgetBytes > 0)
    tighten(static_cast<int64_t>(policy.l0CBudgetBytes /
                                 static_cast<uint64_t>(policy.maxL0CBytes)));
  // 单个提升值的字节预算。
  // Per-value byte budget.
  if (maxTensorBytes > 0)
    tighten(budget / maxTensorBytes);
  // UB 预算：提升后与查询行相关的值随 R 线性增长，因此以原始函数的峰值活值
  // 作为每因子需求，即 R × estimatePeakLiveBytes <= UB。预算是 ResourceSnapshot
  // 的真实可用 UB（物理容量减预留后打安全系数），它与 L0A/L0C、网格一样只在 R
  // 取值的阈值里生效，不再作为事后的合法性判据。
  // UB budget: query-row-dependent values grow linearly with R, so the un-lifted
  // peak live bytes are the per-factor demand, i.e. R × estimatePeakLiveBytes <=
  // UB.  The budget is the real usable UB from ResourceSnapshot (physical
  // capacity minus reserved, scaled by the safety percent), and it bounds the R
  // value together with L0A/L0C and the grid instead of acting as a post-hoc
  // legality check.
  if (policy.ubBudgetBytes > 0 && peakLiveBytes > 0)
    tighten(static_cast<int64_t>(
        policy.ubBudgetBytes / static_cast<uint64_t>(peakLiveBytes)));
  // 提升后 M 维上限；0 表示调用方不限制。
  // Cap on the lifted M dimension; zero means the caller does not cap it.
  if (policy.targetDotRows > 0 && maxDotRows > 0)
    tighten(policy.targetDotRows / maxDotRows);
  // 合并后轴 0 仍要铺满一轮物理核；只有静态网格可判定，运行期网格 grid 为 0。
  // Axis 0 must still fill one full wave of physical cores.  Only a static
  // extent can be judged; a runtime grid reports zero.
  if (grid > 0 && policy.coreNum > 0)
    tighten(grid / static_cast<int64_t>(policy.coreNum));

  if (!limit)
    return 1;

  // 向下取 2 的幂：重写与重试都按 2 的幂收缩因子。
  // Floor to a power of two: the rewrite and its retries shrink the factor by
  // halving.
  int64_t factor = 1;
  while (factor <= *limit / 2)
    factor *= 2;

  // 静态网格必须整除，否则会改变 program 与查询行的映射；运行期网格用掩码
  // 处理最后一组。整除不是单调条件，所以仍需向下回退。
  // A static grid must divide exactly, otherwise the program-to-row mapping
  // changes; a runtime extent masks the last group.  Divisibility is not
  // monotone, so it still needs the downward walk.
  while (factor >= 2 && grid && grid % factor != 0)
    factor /= 2;

  return factor >= 2 ? factor : 1;
}

// 返回单个元素的字节数；未知元素类型返回 0。
int64_t getElementStorageBytes(Type element) {
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
  int64_t bytes = getElementStorageBytes(type.getElementType());
  if (bytes == 0)
    return 0;
  for (int64_t extent : type.getShape())
    bytes *= extent;
  return bytes;
}

// 保存重写器与候选描述，能力开关从候选同步过来。
RowLiftBuilder::RowLiftBuilder(IRRewriter &rewriter, RowLiftCandidate candidate)
    : rewriter(rewriter), candidate(candidate), features(candidate.features) {}

// 运行期网格把 x 轴 extent 作为倒数第二个隐藏参数追加在函数参数尾部
// （末尾一个是 y 轴 extent）。参数布局由 ProgramGridTransform 定义，这里集中
// 一处取用，避免布局变化时两处静默错位。
// A runtime grid appends the x extent as the second-to-last hidden argument (the
// last one is the y extent).  ProgramGridTransform owns that layout; fetching it
// in one place keeps a layout change from silently mis-indexing both call sites.
static Value getRuntimeGridExtent(Block *body) {
  return body->getArgument(body->getNumArguments() - 2);
}

// 克隆函数体并逐条重建，成功时用新函数体替换旧函数体并返回 true。
//
// 重建的单位是「函数级」而不是单条指令：先把原 block 的每条指令按行提升
// 语义重放一遍，全部成功后才丢弃旧 block，从而保证失败时原函数保持可用。
bool RowLiftBuilder::run(triton::FuncOp fn) {
  Block *old = &fn.getBody().front();
  // 原循环的起始上界。整组的 trip count 由末行取值决定，因此重建过程中
  // 需要知道哪些标量参与了这个上界的计算（见下方的末行切片克隆）。
  // 公共引擎要求函数体内恰好有一个循环；没有循环时直接拒绝，避免解引用
  // 空迭代器（调用方的 matcher 虽然已经保证 loops==1，但这里仍按 API 契约判空）。
  auto loops = old->getOps<scf::ForOp>();
  if (loops.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "row-lift: no loop to lift\n");
    return false;
  }
  scf::ForOp loop = *loops.begin();
  Value upperBound = loop.getUpperBound();
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
        Value extent = getRuntimeGridExtent(fresh);
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
      // 运行期网格下，再用「组内行号 < 剩余行数」得到逐行有效掩码，
      // 后续所有与行相关的访存都要在它之下重建。
      if (candidate.runtimeGrid) {
        Value extent = getRuntimeGridExtent(fresh);
        // 逐行有效位写成「纯组内行号 vs 标量上界」：base + lanes < extent 等价于
        // lanes < extent - base，这样比较的左侧是纯 make_range（静态偏移），
        // 右侧是单值，掩码分析可以把它折叠成一个矩形区域，而不是一条带动态
        // 基址的算术链。行号本身（valueMapping）仍然是 base + lanes，作为逐行
        // 访存的行下标保留。
        // Spell the row-validity as "pure in-group lane range vs a scalar bound":
        // base + lanes < extent is the same as lanes < extent - base, and the
        // left-hand side becomes a plain make_range with a static offset so the
        // mask analysis can fold it into a rectangle instead of an arithmetic
        // chain with a dynamic base.  The lifted row index itself keeps base.
        Value rowsLeft = rewriter.create<arith::SubIOp>(op.getLoc(), extent,
                                                        base);
        Value limit = rewriter.create<triton::SplatOp>(op.getLoc(), t, rowsLeft);
        validRows = rewriter.create<arith::CmpIOp>(
            op.getLoc(), arith::CmpIPredicate::slt, lanes, limit);
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

// 掩码访存的填充值：PAD_NAN 要求浮点元素类型，用 NaN 常量再广播成结果形状，
// 与描述符访存路径保持同一语义；其余 padding 选项沿用零值。非浮点元素上
// 请求 PAD_NAN 时返回空值，由调用方拒绝。
// Padding value of a masked access: PAD_NAN needs a floating element type and
// splats a NaN, matching the descriptor path; every other padding option keeps
// zero.  PAD_NAN on a non-float element returns null and the caller declines.
static Value buildMaskPadding(IRRewriter &rewriter, Location loc, Type type,
                              std::optional<triton::PaddingOption> padding) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  Type element = tensor ? tensor.getElementType() : type;
  if (padding != triton::PaddingOption::PAD_NAN)
    return rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(type));
  auto floatType = dyn_cast<FloatType>(element);
  if (!floatType)
    return Value();
  Value nan = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getFloatAttr(
               floatType, APFloat::getNaN(floatType.getFloatSemantics())));
  return tensor ? Value(rewriter.create<triton::SplatOp>(loc, type, nan)) : nan;
}

// 按操作种类分派重建；遇到尚未支持的形态返回 false，整次提升随即回退。
//
// 派发顺序带有语义，不能随意调整：
//   0) scf.for 先处理，循环重写会一次性重建所有循环进位值；
//   1) 「已被拆成两半的值」必须早于通用依赖探测，否则探测会去查不存在的
//      全宽映射，克隆出非法操作数；
//   2) 依赖探测：与查询行无关的指令直接克隆，与行相关的才需要特化；
//   3) 带 region 的形态（tt.reduce / tt.make_tensor_ptr / scf.if）逐个重建；
//   4) 描述符访存、循环内行相关访存、dot / splat / expand / reshape / trans
//      走各自的形状规则；
//   5) 其余形态走通用克隆，操作数统一补行维、结果类型统一加行维。
bool RowLiftBuilder::rewrite(Operation *op) {
  // 步骤 0：循环是唯一会改变循环进位值个数的形态。
  if (auto loop = dyn_cast<scf::ForOp>(op))
    return rewriteLoop(loop);

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
                        : buildMaskPadding(rewriter, op->getLoc(), type,
                                           load.getPadding());
      if (!other)
        return false;
      auto replacement = rewriter.create<triton::LoadOp>(
          op->getLoc(), pointer, mask, other, load.getBoundaryCheck(),
          load.getPadding(), load.getCache(), load.getEvict(),
          load.getIsVolatile());
      valueMapping.map(load.getResult(), replacement.getResult());
      return true;
    }
  }

  // 步骤 4i：运行期网格下的 store 需要逐行有效掩码，因此单独重建。带掩码的
  // 写回能力尚未开启，带掩码 store 暂时失败关闭。
  if (auto store = dyn_cast<triton::StoreOp>(op)) {
    if (candidate.runtimeGrid) {
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
    // 循环体内没有掩码的 store 必须失败关闭：提升后的循环跑到组内末行上界，
    // 扫描已提前结束的行在本轮仍会执行到这条 store，通用克隆会把整段
    // [R, ...] 写回，而它们的结果本应保持旧值。
    // An unmasked store inside the loop has to fail closed: the lifted loop runs
    // to the group's last-row bound, so rows whose scan already ended still reach
    // this store, and the generic clone would write the whole [R, ...] block back.
    if (op->getParentOfType<scf::ForOp>() && !store.getMask())
      return false;
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

// 把原循环重建为多行循环：先规划累加器拆分，再用末行上界生成新循环。
//
// 重建步骤：
//   1) 循环进位值先补行维，作为新循环的初始值；
//   2) 累加器拆分规划必须在建循环之前完成，因为它会改变进位值个数；
//   3) 新循环的 trip count 取「末行上界」，单调性由 matcher 已证明；
//   4) 循环体逐条重建；maskedLoad 需要先准备好逐行有效掩码；
//   5) 循环末尾用有效行掩码选择「本轮新值 / 上一轮值」；
//   6) 循环结果写回映射；拆分的累加器以成对结果继续向后传递。
bool RowLiftBuilder::rewriteLoop(scf::ForOp old) {
  // 已展开成指针张量的块指针不能作为循环进位值：展开时已删掉描述符，
  // 循环里没有它的等价物。
  // A loop carrying an expanded block pointer would need a descriptor value
  // that the expansion deliberately removed.
  for (Value arg : old.getInitArgs())
    if (liftedDescriptors.contains(arg))
      return false;

  // 每个循环从头开始，清空上一个循环遗留的有效行掩码。
  rowActiveMask = Value();

  // 步骤 1+2：原循环的进位值统一补上行维。
  // Step 1+2: every carried value gains the row dimension.
  SmallVector<Value> loopInit;
  for (Value arg : old.getInitArgs())
    loopInit.push_back(ensureRowDimension(arg));

  // 步骤 3：末行上界必须已经物化到 lastRowMapping（run 里克隆的标量上界
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

  // 归纳变量与各进位值的映射。
  valueMapping.map(old.getInductionVar(), loop.getInductionVar());
  for (auto [index, arg] : llvm::enumerate(old.getRegionIterArgs()))
    valueMapping.map(arg, loop.getRegionIterArgs()[index]);

  rewriter.setInsertionPointToStart(loop.getBody());
  for (Operation &op : old.getBody()->without_terminator())
    if (!rewrite(&op))
      return false;

  auto yield = cast<scf::YieldOp>(old.getBody()->getTerminator());
  // maskedLoad 关闭时，有效行掩码仍要在循环体之后生成；复用同一个辅助
  // 函数，指令序列与基线保持一致。
  if (!rowActiveMask)
    rowActiveMask = createLoopActiveMask(old, loop);
  Value active = rowActiveMask;

  // 步骤 5：对「原循环在本轮没有执行」的行，保持上一轮的值不变（同时保留
  // NaN/Inf 语义）。掩码为假时选 previous。
  // Retain exactly the previous accumulator for rows whose original loop did
  // not execute this iteration (also preserves NaN/Inf behavior).
  auto keep = [&](Value incoming, Value previous, RankedTensorType type) {
    Value mask = broadcastRowValue(
        active, RankedTensorType::get(type.getShape(), rewriter.getI1Type()));
    return Value(rewriter.create<arith::SelectOp>(old.getLoc(), mask, incoming,
                                                  previous));
  };

  // 逐项生成新的 yield：原进位值补行维后逐项选择。
  SmallVector<Value> output;
  for (auto [index, value] : llvm::enumerate(yield.getOperands())) {
    auto previous =
        cast<RankedTensorType>(loop.getRegionIterArgs()[index].getType());
    output.push_back(keep(ensureRowDimension(value),
                          loop.getRegionIterArgs()[index], previous));
  }

  // 用新生成的 yield 替换循环体末尾的占位 terminator。
  if (!loop.getBody()->empty() && isa<scf::YieldOp>(loop.getBody()->back()))
    rewriter.eraseOp(loop.getBody()->getTerminator());
  rewriter.setInsertionPointToEnd(loop.getBody());
  rewriter.create<scf::YieldOp>(old.getLoc(), output);

  // 步骤 6：循环结果回填映射。
  for (auto [index, result] : llvm::enumerate(old.getResults()))
    valueMapping.map(result, loop.getResults()[index]);

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

// 生成沿指定 tile 轴递增的索引网格并广播到目标形状，元素宽度跟随目标类型。
// 地址运算统一用 i32（见 expandTensorPointer 的说明），因此只有目标类型是
// i64 时才把 [0, extent) 区间扩宽；这样展开路径不再额外产生 i64 中间量。
// Build the ascending index grid for one tile axis and broadcast it to the target
// shape, with the element width following the target type.  The address
// arithmetic uses i32 (see expandTensorPointer), so the [0, extent) range is only
// widened when the target is i64, which keeps the expansion path free of extra
// i64 intermediates.
Value RowLiftBuilder::expandTileRange(Location loc, int64_t axis,
                                      int64_t extent, RankedTensorType target) {
  auto elementType = target.getElementType();
  auto integer = dyn_cast<IntegerType>(elementType);
  if (!integer)
    return Value();
  auto rangeType =
      RankedTensorType::get({extent}, rewriter.getI32Type());
  Value range = rewriter.create<triton::MakeRangeOp>(
      loc, rangeType, 0, static_cast<int32_t>(extent));
  if (integer.getWidth() > 32)
    range = rewriter.create<arith::ExtSIOp>(
        loc, RankedTensorType::get({extent}, elementType), range);

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
  // 地址运算统一用 i32（与手写 kernel 一致）：!tt.ptr 地址向量的元素宽度决定
  // 它在 UB 里的常驻体积，[R]+tile 的 i64 地址向量在 D128 上就有 128 KiB，
  // 还会连带把边界掩码的比较链也拉到 i64 域。手写版读到表值后立刻 trunci 到
  // i32 再算地址，这里改成同样口径。
  // Address arithmetic uses i32, like the handwritten kernel: the element width of
  // the !tt.ptr address vector decides how much UB it occupies resident, and a
  // [R]+tile i64 vector already costs 128 KiB on D128 while also dragging the
  // boundary-mask comparisons into the i64 domain.  The handwritten kernel truncates
  // table values to i32 right after loading them; this path now does the same.
  auto addressType = rewriter.getI32Type();
  auto elementPointer =
      triton::PointerType::get(blockType.getElementType(),
                               pointerType.getAddressSpace());
  SmallVector<int64_t> fullShape{candidate.rowsPerProgram};
  fullShape.append(tileShape.begin(), tileShape.end());
  auto indexType = RankedTensorType::get(fullShape, addressType);
  auto pointerTensorType = RankedTensorType::get(fullShape, elementPointer);

  // shape/stride 在 make_tensor_ptr 里是 i64、offset 可能是 i32，地址运算前
  // 统一收敛到 i32：宽于 32 位的截断，窄于 32 位的符号扩展。
  // 遇到非整数类型时返回空 Value，由调用方拒绝。
  // shape/strides are i64 in make_tensor_ptr and offsets may be i32, so every
  // index is normalised to i32 before the address arithmetic: wider values are
  // truncated, narrower ones sign-extended.
  auto toAddressType = [&](Value v) -> Value {
    Type type = v.getType();
    Type element = type;
    auto tensor = dyn_cast<RankedTensorType>(type);
    if (tensor)
      element = tensor.getElementType();
    auto integer = dyn_cast<IntegerType>(element);
    if (!integer)
      return Value();
    if (integer.getWidth() == 32)
      return v;
    Type converted = tensor ? Type(RankedTensorType::get(tensor.getShape(),
                                                         addressType))
                            : Type(addressType);
    if (integer.getWidth() > 32)
      return rewriter.create<arith::TruncIOp>(loc, converted, v);
    return rewriter.create<arith::ExtSIOp>(loc, converted, v);
  };

  // 逐维取出 offset/shape/stride，形状与偏移都保存下来，后面把 boundaryCheck
  // 转掩码时要用。每个值都先收敛到 i32，地址运算统一位宽。
  SmallVector<Value> offsets, shapes, strides;
  offsets.reserve(tileShape.size());
  shapes.reserve(tileShape.size());
  strides.reserve(tileShape.size());
  const int64_t lastDim = static_cast<int64_t>(tileShape.size()) - 1;
  for (int64_t dim = 0; dim < static_cast<int64_t>(tileShape.size()); ++dim) {
    Value offset =
        (lastOffsetOverride && dim == lastDim)
            ? toAddressType(lastOffsetOverride)
            : toAddressType(getMappedValue(makePtr.getOffsets()[dim]));
    Value shape = toAddressType(getMappedValue(makePtr.getShape()[dim]));
    Value stride = toAddressType(getMappedValue(makePtr.getStrides()[dim]));
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
  auto baseIndexType = RankedTensorType::get(baseShape, addressType);
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
    // 逐轴先在「只沿该轴变化」的窄形状上做越界比较，最后再把 i1 谓词广播成整块
    // 掩码。整块形状的索引网格是 UB 里最大的一类临时量（[R, G, 128] 的 i32 网格
    // 就有 64 KiB），而越界判定本来只沿一个 tile 轴变化；先窄后宽与先宽后窄语义
    // 完全相同，却省掉每个轴一份整块索引网格和一份整块比较结果。手工优化版正是
    // 这个形态：在 1x1x64 上比较，只 broadcast 一次 i1。
    // Compare along a narrow single-axis shape and broadcast the predicate once.
    // A full-shape index grid ([R, G, 128] of i32 is 64 KiB) is one of the largest
    // UB temporaries, while an out-of-bounds test only varies along one tile axis.
    // Narrowing first is exactly equivalent and drops one full-shape grid and one
    // full-shape comparison result per axis.  The hand-written kernel has this
    // form: compare on 1x1x64 and broadcast the i1 once.
    const int64_t rank = static_cast<int64_t>(indexType.getRank());
    if (axis + 1 >= rank)
      return Value();
    // 比较要落在「至多一个非 1 维」的形状上，行维也必须压成 1：离散掩码判定只
    // 接受单轴变化的 cmpi，凡是出现两个及以上非 1 维的比较一律判为离散掩码，
    // 访存随后被迫降级成逐元素的 indirect_load（i64 索引网格 + 逐元素谓词）。
    // 手工优化版的三片谓词都只有一个非 1 维，最后才 andi/broadcast 回整块。
    // A comparison has to live on a shape with at most one non-unit dimension, so
    // the row dimension must narrow to 1 as well: discrete-mask detection only
    // accepts single-axis comparisons, and any comparison with two or more
    // non-unit dimensions is classified as a discrete mask, which downgrades the
    // access to per-element indirect_load (i64 index grid plus element predicates).
    // All three predicates of the hand-written kernel have exactly one non-unit
    // dimension, and are andi/broadcast to the full shape only afterwards.
    // 例外：该轴的偏移或形状本身逐行变化时谓词必须逐行变化，压成 1 既无法广播
    // 也表达不出逐行越界，这种轴退回保留行维的旧形态。
    // Exception: when the offset or shape of this axis itself varies per row the
    // predicate must vary per row too, so such an axis keeps the old form.
    auto narrowable = [&](Value v) {
      auto tensor = dyn_cast<RankedTensorType>(v.getType());
      if (!tensor)
        return true;
      for (int64_t dim = 0; dim < tensor.getRank(); ++dim)
        if (dim != axis + 1 && tensor.getShape()[dim] != 1)
          return false;
      return true;
    };
    SmallVector<int64_t> narrow(indexType.getShape().begin(),
                                indexType.getShape().end());
    if (descriptor.tileShape[axis] > 1 &&
        narrowable(descriptor.offsets[axis]) &&
        narrowable(descriptor.shapes[axis])) {
      narrow.assign(rank, 1);
    } else {
      for (int64_t dim = 1; dim < rank; ++dim)
        narrow[dim] = 1;
    }
    narrow[axis + 1] = descriptor.tileShape[axis];
    auto narrowType = RankedTensorType::get(narrow, indexType.getElementType());

    // 该轴的逐元素下标网格（窄形状）。
    Value index = rewriter.create<arith::AddIOp>(
        loc, broadcastFullValue(descriptor.offsets[axis], narrowType),
        expandTileRange(loc, axis, descriptor.tileShape[axis], narrowType));
    Value zero = rewriter.create<triton::SplatOp>(
        loc, narrowType,
        rewriter.create<arith::ConstantOp>(
            loc, rewriter.getIntegerAttr(
                     cast<IntegerType>(indexType.getElementType()), 0)));
    // 下界：index >= 0；上界：index < shape（shape 来自描述符展开时保存的值）。
    Value lower = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sge, index, zero);
    Value upper = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, index,
        broadcastFullValue(descriptor.shapes[axis], narrowType));
    Value axisNarrow = rewriter.create<arith::AndIOp>(loc, lower, upper);
    auto maskType =
        RankedTensorType::get(indexType.getShape(), rewriter.getI1Type());
    Value axisMask =
        rewriter.create<triton::BroadcastOp>(loc, maskType, axisNarrow);
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
  if (load.getMask())
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
      RankedTensorType::get(pointerType.getShape(), rewriter.getI32Type());

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
      // 先按类型判断再生成 IR：拆半路径的结果比提升后的整宽值窄，此时直接按
      // 半宽折叠，不能先无条件提升出一个全宽临时值。
      // Decide from the types before emitting anything: on the split path the
      // result is narrower than the lifted full width, so fold straight to half
      // width instead of materialising a full-width temporary first.
      if (getLiftedType(load.getOther().getType()) == resultType) {
        other = broadcastRowValue(ensureRowDimension(load.getOther()),
                                  resultType);
      } else {
        other = rebuildHalfOperand(load.getOther(), resultType);
        if (!other) {
          Value lifted = ensureRowDimension(load.getOther());
          if (lifted.getType() == resultType)
            other = lifted;
        }
      }
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


// 用展开后的指针张量重建描述符 store，并合并边界掩码与原始 mask。
bool RowLiftBuilder::rewriteDescriptorStore(
    triton::StoreOp store, const LiftedDescriptor &descriptor) {
  // O 与 lse 的带掩码写回：mask 操作数属于行维展开的一部分，受
  // maskedStore 能力控制。
  // O and lse are written back under a mask, so the mask operand is part of
  // the row expansion when the capability is enabled.
  if (store.getMask())
    return false;
  // 边界检查轴号必须落在 tile 维度内。
  ArrayRef<int32_t> boundaryCheck = store.getBoundaryCheck();
  for (int32_t axis : boundaryCheck)
    if (axis < 0 || axis >= static_cast<int32_t>(descriptor.tileShape.size()))
      return false;

  Location loc = store.getLoc();
  auto pointerType = cast<RankedTensorType>(descriptor.pointer.getType());
  auto indexType =
      RankedTensorType::get(pointerType.getShape(), rewriter.getI32Type());

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
  // 条件必须是标量 i1，且不依赖查询行；否则各行可能走不同分支。这里用会
  // 穿透 region 的 isPidInvariant：条件可能由某个 scf.if 的分支产出，浅层
  // 判据会把这种情况误判为行无关。
  // Every lifted row takes the same branch, so the condition stays a scalar and
  // must not depend on the query row.  isPidInvariant looks through regions: the
  // condition may come from a branch of another scf.if, which the shallow probe
  // would wrongly report as row-invariant.
  Value condition = ifOp.getCondition();
  if (!condition.getType().isInteger(1) || !isPidInvariant(condition, pid))
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

  // 结果类型统一补行维；块指针不能作为 scf.if 的结果值。
  // Every result gains the row dimension.
  SmallVector<Type> types;
  for (Type t : old.getResultTypes()) {
    // A block pointer cannot be carried by an scf.if result.
    if (isTensorPointer(t))
      return false;
    types.push_back(getLiftedType(t));
  }

  // 先建出带空分支的 scf.if，再逐分支重建。
  auto replacement = rewriter.create<scf::IfOp>(
      old.getLoc(), types, valueMapping.lookupOrDefault(condition),
      /*withElseRegion=*/!old.getElseRegion().empty());
  if (!rewriteIfBody(old.getThenRegion(), replacement.getThenRegion()) ||
      !rewriteIfBody(old.getElseRegion(), replacement.getElseRegion()))
    return false;

  // 分支重建完成后插入点会留在分支 block 内，这里移回 if 之后。
  // Body rewriting leaves the insertion point inside the branch blocks.
  rewriter.setInsertionPointAfter(replacement);
  valueMapping.map(old.getResults(), replacement.getResults());
  return true;
}

// 重建 scf.if 的一个分支 region，并生成提升后的 yield。
bool RowLiftBuilder::rewriteIfBody(Region &from, Region &to) {
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

  // 重建分支 yield：每个值补行维；已展开的块指针无法作为 yield 值。
  auto yield = cast<scf::YieldOp>(from.front().getTerminator());
  SmallVector<Value> values;
  for (Value v : yield.getOperands()) {
    // An expanded block pointer has no value that can be yielded.
    if (liftedDescriptors.contains(v))
      return false;
    values.push_back(ensureRowDimension(v));
  }
  rewriter.create<scf::YieldOp>(yield.getLoc(), values);
  return true;
}

} // namespace cfg
} // namespace triton
} // namespace mlir
