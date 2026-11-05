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
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <cstdlib>
#include <utility>

#define DEBUG_TYPE "graph-optimize"

using namespace mlir;
using namespace triton;

namespace mlir {
namespace triton {
namespace cfg {

// 本文件把「一个 program 只处理一行查询」的函数体重建为「一个 program
// 处理 R 行查询」：与查询行相关的值在类型最前面插入行维 [R]，循环用组内
// 最后一个有效行的上界作为共享 trip count，各行的差异由逐行有效掩码表达。
//
// 公共类型、依赖分析与形状/掩码工具之后，扩展能力按需求顺序组织：
//   1. tt.reduce 与条件和行号无关的 scf.if；
//   2. Q、O 的 tensor pointer 展开成显式指针张量；
//   3. 下标表等行相关数据的逐行读取（也支持规则 A 提到循环外的表读取）；
//   4. O/lse 的带掩码写回，以及循环状态和计算链的压平；
//   5. 过宽输出累加器与配套描述符沿末维拆半。
// 最后是引擎入口与指令分派；函数的排列不改变分派顺序。
//
// DotRowCoalescing 与规则 B 共用本引擎，各自维护 matcher。RowLiftFeatures
// 默认全部关闭；DotRow 不启用这些扩展开关，两个规则都使用共享上界策略。
// 循环压平针对计算链，地址、掩码和布局边界仍可保留 reshape。启动契约由规则层发布；双版本
// 生成与运行期选择不属于本组件，网格契约本身也不提供双版本分派。

//===----------------------------------------------------------------------===//
// Common analysis, shape and memory-mask helpers
//===----------------------------------------------------------------------===//

// 判断 v 是否等于 pid 或由 pid 的顶层计算链产生；不深入 region。
bool dependsOnValue(Value v, Value pid) {
  if (v == pid)
    return true;

  Operation *op = v.getDefiningOp();
  return op && llvm::any_of(op->getOperands(),
                            [&](Value a) { return dependsOnValue(a, pid); });
}

// 块参数本身不会引入对其它值的依赖；检查定义操作的操作数与嵌套 region。
// scf.if 的结果来自分支 yield，tt.reduce 的结果来自输入组合，不能只看顶层。
// Block arguments do not introduce a dependence by themselves. Inspect the
// defining operation's operands and regions, including branch/reduction bodies.
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

// 判断 value 是否完全与 pid 无关，判定会穿透 scf.if / tt.reduce 的 region。
bool isPidInvariant(Value value, Value pid) {
  return !dependsOnValueThroughRegions(value, pid);
}

// 读取整数常量；不是常量时返回 nullopt。规则 matcher 与重写路径共用。
std::optional<int64_t> getConstantIntValue(Value v) {
  APInt value;
  if (matchPattern(v, m_ConstantInt(&value)) && value.isSignedIntN(64))
    return value.getSExtValue();

  return std::nullopt;
}

// 标量表达式在 pid 范围内的有符号区间，仅供本文件的上界和等迭代数分析使用。
struct IntegerBounds {
  int64_t lo;
  int64_t hi;
};

// 证明上界随 pid 非降，并保留区间供乘数符号和等迭代数分析使用。
// 输入约定：有效行的上界算术按普通有符号整数计算，不发生回绕。这里不再要求
// 静态范围完全落在位宽内；例如运行期 pid + 共享偏移可证明非降，但独立区间
// 无法表达 grid 与序列长度的关联。此时仍返回单调，调用方取最后一个有效行。
// 该约定只放宽上界的数值范围证明，不放行负乘数、非正除数或未知条件选择。
// Valid inputs must not wrap in row-dependent bound arithmetic. The analysis
// proves order under that contract; it does not require independent intervals
// to prove the runtime grid/sequence-length relation. Generated IR keeps its
// original integer width. Unsupported or non-monotone forms decline the lift.
static std::optional<IntegerBounds> inferMonotonicBounds(Value value, Value pid,
                                                        int64_t grid) {
  auto integer = dyn_cast<IntegerType>(value.getType());
  if (!integer || integer.getWidth() > 64)
    return std::nullopt;
  unsigned width = integer.getWidth();
  IntegerBounds domain{APInt::getSignedMinValue(width).getSExtValue(),
                       APInt::getSignedMaxValue(width).getSExtValue()};
  // 128 位仅用于编译器内部的区间运算，避免分析器自身溢出；按上述有效输入约定
  // 将区间限于结果类型的范围。这不是在生成的 kernel 中插入饱和算术。
  auto finish = [&](__int128 lo, __int128 hi) -> IntegerBounds {
    return {static_cast<int64_t>(std::clamp<__int128>(lo, domain.lo, domain.hi)),
            static_cast<int64_t>(std::clamp<__int128>(hi, domain.lo, domain.hi))};
  };
  if (value == pid)
    return finish(0, (grid > 0 ? grid : INT32_MAX) - 1);
  if (auto constant = getConstantIntValue(value))
    return IntegerBounds{*constant, *constant};
  if (isPidInvariant(value, pid))
    return domain;
  if (auto ext = value.getDefiningOp<arith::ExtSIOp>())
    return inferMonotonicBounds(ext.getIn(), pid, grid);

  // 规则 A 的 NSA 扫描上界：row >= 0 ? row / BN + 1 : 0。
  // row 非降且 BN > 0 时，两段各自非降，在 row=0 处由 0 跳到 1；不需要
  // 通用分支区间推导。接受 canonicalize 后的反向条件，其余 select 仍拒绝。
  if (auto select = value.getDefiningOp<arith::SelectOp>()) {
    auto cmp = select.getCondition().getDefiningOp<arith::CmpIOp>();
    if (!cmp || getConstantIntValue(cmp.getRhs()) != 0)
      return std::nullopt;
    Value positive;
    if (cmp.getPredicate() == arith::CmpIPredicate::sge &&
        getConstantIntValue(select.getFalseValue()) == 0)
      positive = select.getTrueValue();
    else if (cmp.getPredicate() == arith::CmpIPredicate::slt &&
             getConstantIntValue(select.getTrueValue()) == 0)
      positive = select.getFalseValue();
    auto add = positive ? positive.getDefiningOp<arith::AddIOp>()
                        : arith::AddIOp();
    if (!add)
      return std::nullopt;
    Value quotient = getConstantIntValue(add.getRhs()) == 1 ? add.getLhs()
                     : getConstantIntValue(add.getLhs()) == 1 ? add.getRhs()
                                                             : Value();
    auto div = quotient ? quotient.getDefiningOp<arith::DivSIOp>()
                        : arith::DivSIOp();
    if (!div || div.getLhs() != cmp.getLhs())
      return std::nullopt;
    auto divisor = getConstantIntValue(div.getRhs());
    auto bounds = inferMonotonicBounds(div.getLhs(), pid, grid);
    if (!divisor || *divisor <= 0 || !bounds)
      return std::nullopt;
    return finish(0, std::max<__int128>(
                         0, static_cast<__int128>(bounds->hi) / *divisor + 1));
  }

  Operation *op = value.getDefiningOp();
  if (!op || op->getNumOperands() != 2)
    return std::nullopt;
  auto lhs = inferMonotonicBounds(op->getOperand(0), pid, grid);
  auto rhs = inferMonotonicBounds(op->getOperand(1), pid, grid);
  if (!lhs || !rhs)
    return std::nullopt;
  if (isa<arith::AddIOp>(op))
    return finish(static_cast<__int128>(lhs->lo) + rhs->lo,
                  static_cast<__int128>(lhs->hi) + rhs->hi);
  if (isa<arith::SubIOp>(op) && isPidInvariant(op->getOperand(1), pid))
    return finish(static_cast<__int128>(lhs->lo) - rhs->hi,
                  static_cast<__int128>(lhs->hi) - rhs->lo);
  if (isa<arith::MulIOp>(op)) {
    // 乘非负常量保持次序，不要求另一个操作数非负；两个变化量则需都非负。
    if (rhs->lo == rhs->hi && rhs->lo >= 0)
      return finish(static_cast<__int128>(lhs->lo) * rhs->lo,
                    static_cast<__int128>(lhs->hi) * rhs->lo);
    if (lhs->lo == lhs->hi && lhs->lo >= 0)
      return finish(static_cast<__int128>(lhs->lo) * rhs->lo,
                    static_cast<__int128>(lhs->lo) * rhs->hi);
    if (lhs->lo >= 0 && rhs->lo >= 0)
      return finish(static_cast<__int128>(lhs->lo) * rhs->lo,
                    static_cast<__int128>(lhs->hi) * rhs->hi);
  }
  if (isa<arith::DivSIOp>(op) && rhs->lo == rhs->hi && rhs->lo > 0)
    return finish(lhs->lo / rhs->lo, lhs->hi / rhs->lo);
  return std::nullopt;
}

// 在不回绕的输入约定下证明上界非降，并检查合并带来的额外求值是否安全。
// 通过后统一取最后一个有效行；无法证明的形态直接拒绝，不再生成组内 max。
bool canUseLastRowLoopBound(scf::ForOp loop, Value pid, int64_t grid) {
  Value bound = loop.getUpperBound();
  auto unsafeArithmetic = [](Operation *op) {
    StringRef ns = op->getName().getDialectNamespace();
    return (ns == "arith" || ns == "math") && !isSpeculatable(op);
  };
  if (grid <= 0) {
    // Padded tail rows also evaluate address arithmetic outside the bound's
    // slice. Masking a load does not make an earlier integer division safe.
    auto function = loop->getParentOfType<triton::FuncOp>();
    if (!function)
      return false;
    auto result = function.walk([&](Operation *op) {
      bool unsafe = unsafeArithmetic(op) &&
                    llvm::any_of(op->getOperands(), [&](Value operand) {
                      return !isPidInvariant(operand, pid);
                    });
      return unsafe ? WalkResult::interrupt() : WalkResult::advance();
    });
    if (result.wasInterrupted())
      return false;
  }
  if (!isPidInvariant(bound, pid)) {
    // A shared loop also evaluates the body for rows which have stopped. A
    // select on the yielded accumulator cannot repair a trapping operation.
    auto result = loop.getBody()->walk([&](Operation *op) {
      return unsafeArithmetic(op) ? WalkResult::interrupt()
                                  : WalkResult::advance();
    });
    if (result.wasInterrupted())
      return false;
  }
  return inferMonotonicBounds(bound, pid, grid).has_value();
}

// NSA 和原有 DotRow 的上界只重建无 region 的标量算术；其他形态由调用方回退。
static bool isScalarBoundSlice(Operation *op) {
  StringRef ns = op->getName().getDialectNamespace();
  return (ns == "arith" || ns == "math") && op->getNumRegions() == 0;
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
  // 拆分路径会物化两个半宽值，因此单值预算可以翻倍；若不翻倍，拆分路径
  // 永远不可能被收益模型选中。
  // The split path materialises two half-width values, so it may spend twice
  // the per-value budget; without the doubling a split could never be chosen.
  const int64_t budget = policy.splitWideAccumulator
                             ? 2 * policy.maxLiftedValueBytes
                             : policy.maxLiftedValueBytes;

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
  // 作为每因子需求，即 R × peak × 复制份数 <= UB。静态峰值按一份统计，而后端
  // 会多缓冲循环携带的 UB 张量（见 kUBBufferReplication 的实测数据），不乘这份
  // 复制就会放行装不进 UB 的形态（NSA bwd-dq 4 行提升真机 ub overflow）。预算
  // 是 ResourceSnapshot 的真实可用 UB（物理容量减预留后打安全系数），它与
  // L0A/L0C、网格一样只在 R 取值的阈值里生效，不再作为事后的合法性判据。
  // UB budget: query-row-dependent values grow linearly with R, so the un-lifted
  // peak live bytes times the backend's buffer replication are the per-factor
  // demand, i.e. R × peak × replication <= UB.  The static peak counts one copy
  // while the backend multi-buffers loop-carried UB tensors (see the measured
  // numbers on kUBBufferReplication); without that factor a form that cannot fit
  // UB is admitted (the NSA bwd-dq 4-row lift overflows on device).  The budget is
  // the real usable UB from ResourceSnapshot (physical capacity minus reserved,
  // scaled by the safety percent), and it bounds the R value together with
  // L0A/L0C and the grid instead of acting as a post-hoc legality check.
  if (policy.ubBudgetBytes > 0 && peakLiveBytes > 0) {
    const uint64_t perFactorDemand =
        static_cast<uint64_t>(peakLiveBytes) * kUBBufferReplication;
    tighten(static_cast<int64_t>(policy.ubBudgetBytes / perFactorDemand));
  }
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

  if (::getenv("TRITON_GRAPH_UB_DEBUG"))
    llvm::errs() << "graph-ub[rowcount]: unLiftedPeak=" << peakLiveBytes
                 << " ubBudget=" << policy.ubBudgetBytes << " limit=" << *limit
                 << " R=" << factor << "\n";

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

// 运行期网格把 x 轴 extent 作为倒数第二个隐藏参数追加在函数参数尾部
// （末尾一个是 y 轴 extent）。参数布局由 ProgramGridTransform 定义，这里集中
// 一处取用，避免布局变化时两处静默错位。
// A runtime grid appends the x extent as the second-to-last hidden argument (the
// last one is the y extent).  ProgramGridTransform owns that layout; fetching it
// in one place keeps a layout change from silently mis-indexing both call sites.
static Value getRuntimeGridExtent(Block *body) {
  return body->getArgument(body->getNumArguments() - 2);
}

// 末维减半后的张量类型：末维必须能均分成两个连续半宽。
// Tensor type with its trailing extent halved; the trailing extent must split
// into two equal contiguous halves.
static RankedTensorType halvedType(RankedTensorType full) {
  if (!full || full.getRank() < 2)
    return RankedTensorType();
  int64_t width = full.getShape().back();
  if (width < 2 || width % 2 != 0)
    return RankedTensorType();
  SmallVector<int64_t> shape(full.getShape());
  shape.back() /= 2;
  return RankedTensorType::get(shape, full.getElementType());
}

// 边界检查轴号必须落在 tile 维度内。
// Every boundary-check axis has to be a tile dimension.
static bool boundaryAxesInTile(ArrayRef<int32_t> boundaryCheck,
                               ArrayRef<int64_t> tileShape) {
  return llvm::all_of(boundaryCheck, [&](int32_t axis) {
    return axis >= 0 && axis < static_cast<int32_t>(tileShape.size());
  });
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
  Value mask =
      mergeMasks(pointer, Value(),
                 {extraMask, candidate.runtimeGrid ? validRows : Value()}, ok);
  if (!ok)
    return Value();
  return mergeMasks(pointer, mask,
                    {originalMask ? ensureRowDimension(originalMask) : Value()},
                    ok);
}

// 把若干已成形（或只需尾部广播）的掩码分量合并到指针形状上。不做行维补齐，
// 因为拆半路径传入的就是半宽掩码。
// Merge mask parts that are already shaped (or only need trailing broadcast) onto
// the pointer shape.  No row dimension is inserted, because the split path hands
// in half-width masks.
Value RowLiftBuilder::mergeMasks(Value pointer, Value base,
                                 ArrayRef<Value> parts, bool &ok) {
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
    mask =
        mask ? Value(rewriter.create<arith::AndIOp>(part.getLoc(), mask, part))
             : part;
  };
  for (Value part : parts)
    merge(part);
  return mask;
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

//===----------------------------------------------------------------------===//
// 1. tt.reduce and row-invariant scf.if
//===----------------------------------------------------------------------===//

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
  auto replacement = rewriter.create<triton::ReduceOp>(reduce.getLoc(), sources,
                                                       reduce.getAxis() + 1);
  // combine region 只关心元素类型，与形状无关，因此可以整段原样复用。
  // The combine region sees element types only, so it is reused verbatim.
  rewriter.cloneRegionBefore(reduce.getCombineOp(), replacement.getCombineOp(),
                             replacement.getCombineOp().end());
  valueMapping.map(reduce.getResults(), replacement.getResults());
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
                 plannedHalfWidths.contains(old.getResult(index));
    if (halve) {
      auto halfType = halvedType(lifted);
      if (!halfType)
        return false;
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

//===----------------------------------------------------------------------===//
// 2. Q/O tensor pointers expanded into pointer tensors
//===----------------------------------------------------------------------===//

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
  auto rangeType = RankedTensorType::get({extent}, rewriter.getI32Type());
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
  // offset/shape/stride 必须为整数；展开路径把它们转换为 i32 地址偏移。
  // Every index operand must be integral; expansion converts it to i32.
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
  auto elementPointer = triton::PointerType::get(blockType.getElementType(),
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
    Type converted =
        tensor ? Type(RankedTensorType::get(tensor.getShape(), addressType))
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
  for (int64_t dim = 0; dim + 1 < static_cast<int64_t>(tileShape.size());
       ++dim) {
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
  descriptor.pointer = rewriter.create<triton::AddPtrOp>(loc, pointerTensorType,
                                                         pointer, lastTerm);
  descriptor.shapes = shapes;
  descriptor.offsets = offsets;
  descriptor.tileShape = tileShape;
  return true;
}

// 把 boundaryCheck 指定的各轴越界判断合并成显式 i1 掩码。
// 对每个轴计算 0 <= offset + range(0, extent) < shape，再把各轴结果按位与；
// 没有指定边界检查时返回空 Value。
Value RowLiftBuilder::buildDescriptorMask(const LiftedDescriptor &descriptor,
                                          ArrayRef<int32_t> boundaryCheck,
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
    Value lower = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge,
                                                 index, zero);
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
bool RowLiftBuilder::rewriteDescriptorLoad(triton::LoadOp load,
                                           const LiftedDescriptor &descriptor) {
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
  if (!boundaryAxesInTile(boundaryCheck, descriptor.tileShape))
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
        other =
            broadcastRowValue(ensureRowDimension(load.getOther()), resultType);
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
            loc,
            rewriter.getFloatAttr(
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
               .create<triton::LoadOp>(loc, descriptor.pointer, mask, other,
                                       ArrayRef<int32_t>{}, load.getPadding(),
                                       load.getCache(), load.getEvict(),
                                       load.getIsVolatile())
               .getResult();
  return true;
}

//===----------------------------------------------------------------------===//
// 3. Per-row reads, including the selected-block index table
//===----------------------------------------------------------------------===//

// 下标表的逐行读取复用普通行相关 load 的重建。规则 A 把表读取提到循环外
// 时，表地址仍按 [R,...] 提升；循环内的读取额外受 iv < ub 的活动掩码保护。
// Read each row's own index table through the ordinary row-dependent load path.
// Rule A may hoist the table above the loop; its address still gains [R,...].
// In-loop reads additionally require the row's scan to be active.
bool RowLiftBuilder::rewriteRowLoad(triton::LoadOp load) {
  bool insideLoop = static_cast<bool>(load->getParentOfType<scf::ForOp>());
  // A row-dependent load in the body reads the addresses of rows whose scan
  // may already have ended. Without a per-row active mask, decline the lift.
  if (insideLoop && (!features.maskedLoad || !rowActiveMask))
    return false;
  Value pointer = ensureRowDimension(load.getPtr());
  if (!isa<RankedTensorType>(pointer.getType()))
    return false;
  bool ok = true;
  Value mask = combineMemoryMask(pointer, load.getMask(),
                                 insideLoop ? rowActiveMask : Value(), ok);
  if (!ok || (insideLoop && !mask))
    return false;

  // other 也补行维；循环内再广播到 load 结果形状，保留原有填充语义。
  // Lift the fill value too. Keep the historical in-loop broadcast and the
  // runtime-grid path's direct mapped value.
  auto type = getLiftedType(load.getType());
  Value other;
  if (load.getOther()) {
    other = ensureRowDimension(load.getOther());
    if (insideLoop)
      other = broadcastFullValue(other, type);
  } else {
    other = buildMaskPadding(rewriter, load.getLoc(), type, load.getPadding());
  }
  if (!other)
    return false;
  auto replacement = rewriter.create<triton::LoadOp>(
      load.getLoc(), pointer, mask, other, load.getBoundaryCheck(),
      load.getPadding(), load.getCache(), load.getEvict(),
      load.getIsVolatile());
  valueMapping.map(load.getResult(), replacement.getResult());
  return true;
}

//===----------------------------------------------------------------------===//
// 4. Masked O/lse stores and flat loop state/computation
//===----------------------------------------------------------------------===//

// O/lse 的普通指针张量写回：合并原始 mask 与运行期末组有效行掩码。
// Merge the original output mask and the runtime grid's valid-row mask before
// storing O/lse through an ordinary pointer tensor.
bool RowLiftBuilder::rewriteMaskedStore(triton::StoreOp store) {
  Value pointer = ensureRowDimension(store.getPtr());
  if (!isa<RankedTensorType>(pointer.getType()))
    return false;
  bool ok = true;
  Value mask = combineMemoryMask(pointer, store.getMask(), Value(), ok);
  if (!ok)
    return false;
  rewriter.create<triton::StoreOp>(
      store.getLoc(), pointer, ensureRowDimension(store.getValue()), mask,
      store.getBoundaryCheck(), store.getCache(), store.getEvict());
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
  if (!boundaryAxesInTile(boundaryCheck, descriptor.tileShape))
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
  rewriter.create<triton::StoreOp>(loc, descriptor.pointer, value, mask,
                                   rewriter.getDenseI32ArrayAttr({}),
                                   store.getCache(), store.getEvict());
  return true;
}

// 依照上界不回绕的输入约定，判断对齐行组内的迭代数是否相同；仅用于省去
// 多余的逐行 active 选择，不替代上界单调性检查。
// Prove equal trip counts inside an aligned row group. For
// (pid + offset) / divisor + shift, offset and divisor must be multiples of R;
// the outer constant shift does not affect equality. This also recognizes the
// pid / BN + 1 emitted by rule A. This uses the same no-wrap input contract as
// the monotonicity analysis; alignment alone does not establish monotonicity.
static bool hasUniformGroupedTripCount(scf::ForOp loop, Value pid,
                                       int64_t rows, int64_t grid) {
  Value bound = loop.getUpperBound();
  if (isPidInvariant(bound, pid))
    return true;
  if (rows <= 0 || (rows & (rows - 1)) ||
      getConstantIntValue(loop.getLowerBound()) != std::optional<int64_t>(0))
    return false;
  auto step = getConstantIntValue(loop.getStep());
  if (!step || *step <= 0)
    return false;
  auto stripExtension = [](Value value) {
    while (auto ext = value.getDefiningOp<arith::ExtSIOp>())
      value = ext.getIn();
    return value;
  };
  bound = stripExtension(bound);
  if (auto add = bound.getDefiningOp<arith::AddIOp>()) {
    if (getConstantIntValue(add.getRhs()))
      bound = stripExtension(add.getLhs());
    else if (getConstantIntValue(add.getLhs()))
      bound = stripExtension(add.getRhs());
    else
      return false;
  }
  auto div = bound.getDefiningOp<arith::DivSIOp>();
  if (!div)
    return false;
  auto divisor = getConstantIntValue(div.getRhs());
  if (!divisor || *divisor <= 0 || *divisor % rows)
    return false;
  Value numerator = div.getLhs();
  // Signed division truncates toward zero. Even an aligned pair [-32, -31]
  // divided by 32 gives [-1, 0]; an outer +1 makes their trip counts differ.
  // Require the numerator to stay nonnegative before using the alignment test.
  auto bounds = inferMonotonicBounds(numerator, pid, grid);
  if (!bounds || bounds->lo < 0)
    return false;
  if (auto add = numerator.getDefiningOp<arith::AddIOp>()) {
    Value source = add.getLhs();
    auto offset = getConstantIntValue(add.getRhs());
    if (!offset) {
      source = add.getRhs();
      offset = getConstantIntValue(add.getLhs());
    }
    if (!offset || *offset % rows)
      return false;
    numerator = source;
  }
  return stripExtension(numerator) == pid;
}

// 把原循环重建为多行循环：先规划累加器拆分，再用共享上界生成新循环。
//
// 重建步骤：
//   1) 循环进位值先补行维，作为新循环的初始值；
//   2) 累加器拆分规划必须在建循环之前完成，因为它会改变进位值个数；
//   3) 非降性已由匹配期确认，取最后一个有效行的上界；
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
  const bool uniformTripCount =
      features.groupedLoopBounds &&
      hasUniformGroupedTripCount(old, candidate.anchor->getResult(0),
                                  candidate.rowsPerProgram, candidate.grid);

  // 步骤 1+2：原循环的进位值统一补上行维；哪些累加器要拆半已在 run 里定下
  // （描述符必须先于展开拆半），新循环要为每个拆半值携带两个进位值。拆半
  // 初值直接从原始进位值算，避免对已提升值再做一次行维提升。
  // Step 1+2: every carried value gains the row dimension; the accumulators
  // planned for halving in `run` become two loop-carried values.  A split
  // initial value is derived from the original one, never from its lifted copy.
  // 建立「原进位值索引 -> 新进位值起始位置 + 个数」的映射：一个未拆分的
  // 累加器对应 1 个新进位值，拆分的累加器对应 2 个半宽值。
  // One original accumulator maps onto one or two loop-carried values.
  SmallVector<Value> loopInit;
  SmallVector<std::pair<int64_t, int64_t>> ranges;
  for (auto [index, arg] : llvm::enumerate(old.getInitArgs())) {
    if (llvm::is_contained(plannedSplitIndices, static_cast<int64_t>(index))) {
      SplitValue halves;
      if (!ensureRowHalves(arg, halves))
        return false;
      ranges.push_back({loopInit.size(), 2});
      loopInit.push_back(halves.first);
      loopInit.push_back(halves.second);
    } else {
      ranges.push_back({loopInit.size(), 1});
      loopInit.push_back(ensureRowDimension(arg));
    }
  }

  // 步骤 3：共享上界在循环外只计算一次；无法物化时丢弃沙箱。
  Value sharedBound = lastRowMapping.lookupOrNull(old.getUpperBound());
  if (!sharedBound)
    return false;
  // 下界与步长沿用原循环（它们与行无关）。
  auto loop = rewriter.create<scf::ForOp>(
      old.getLoc(), getMappedValue(old.getLowerBound()),
      sharedBound, getMappedValue(old.getStep()), loopInit);
  loop->setAttrs(old->getAttrs());

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
  // 步骤 4：循环体内与行相关的访存需要逐行有效掩码，因此必须在重建循环体
  // 之前就准备好。
  // A row-dependent body access needs the per-row mask, so it has to be
  // available before the body is rebuilt.
  if (uniformTripCount) {
    auto type = getLiftedType(rewriter.getI1Type());
    rowActiveMask = rewriter.create<arith::ConstantOp>(
        old.getLoc(), DenseElementsAttr::get(type, true));
    LLVM_DEBUG(llvm::dbgs() << "row-lift: uniform grouped trip count\n");
  } else if (features.maskedLoad)
    rowActiveMask = createLoopActiveMask(old, loop);
  for (Operation &op : old.getBody()->without_terminator())
    if (!rewrite(&op))
      return false;

  auto yield = cast<scf::YieldOp>(old.getBody()->getTerminator());
  // maskedLoad 关闭时，有效行掩码仍要在循环体之后生成；复用同一个辅助
  // 函数，指令序列与基线保持一致。
  if (!rowActiveMask)
    rowActiveMask = createLoopActiveMask(old, loop);
  // 步骤 5：对「原循环在本轮没有执行」的行，保持上一轮的值不变（同时保留
  // NaN/Inf 语义）。掩码为假时选 previous。
  // Retain exactly the previous accumulator for rows whose original loop did
  // not execute this iteration (also preserves NaN/Inf behavior).
  auto keep = [&](Value incoming, Value previous, RankedTensorType type) {
    if (uniformTripCount)
      return incoming;
    Value mask = broadcastRowValue(
        rowActiveMask,
        RankedTensorType::get(type.getShape(), rewriter.getI1Type()));
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
      output.push_back(
          keep(it->second.second, loop.getRegionIterArgs()[start + 1],
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

  // 步骤 6：循环结果回填映射。拆分的累加器以成对结果离开循环，后续 store
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
  Value active = rewriter.create<arith::CmpIOp>(
      old.getLoc(), arith::CmpIPredicate::slt, ivSplat,
      ensureRowDimension(old.getUpperBound()));
  // 尾组的访存由 combineMemoryMask 等路径单独合并 validRows；逐行计算
  // 不跨查询行归约，因此未写回的补齐行不必在每轮冻结累加器。
  // tt.assert 可能观察这些状态，含断言时保留原来的有效行约束。
  // Memory masks already protect padded rows. Only assertions can observe
  // their otherwise discarded accumulator state; keep the old mask for them.
  bool hasAssert = candidate.function
                       .walk([](triton::AssertOp) {
                         return WalkResult::interrupt();
                       })
                       .wasInterrupted();
  if (validRows && hasAssert)
    active = rewriter.create<arith::AndIOp>(old.getLoc(), active, validRows);
  return active;
}

namespace {

// 行提升构建器用逻辑布局 [R,M,...] 构造地址、掩码和拆半累加器。该布局只在
// 访存边界保留，循环体内的计算区域统一把前两维折叠成 R*M；否则每个 dot
// 前后的 reshape 会打断向量融合，并实打实地生成一份完整的 P 临时量。
// The row-lifting builder uses logical [R,M,...] types to construct addresses,
// masks and split accumulators. Keep that representation at the memory boundary,
// but collapse the first two axes throughout the loop's compute region. Views
// around every dot interrupt vector fusion and materialise full P temporaries.
// 本文件私有的循环重写上下文：类内集中声明接口和状态，函数实现统一放在类外。
class LoopRowFlattener {
public:
  LoopRowFlattener(IRRewriter &rewriter, scf::ForOp loop, int64_t rows,
                   int64_t dotRows);
  void run();

private:
  Type flatType(Type type) const;
  Value view(Value value, Type type);
  Value get(Value value, Type type);
  Value flattenCapture(Value value, Type type);
  void rewriteBlock(Block *from, Block *to, TypeRange yieldTypes);
  void rewrite(Operation *op);

  IRRewriter &rewriter;
  scf::ForOp original;            // 待扁平化的原循环
  scf::ForOp replacement;         // 重建中的新循环
  int64_t rows;                   // 提升行数 R
  int64_t dotRows;                // dot 的 M 维
  IRMapping mapping;              // 旧值 -> 新布局值
  DenseMap<Value, Value> captures;// 循环外不变量的边界视图缓存
};

} // namespace

LoopRowFlattener::LoopRowFlattener(IRRewriter &rewriter, scf::ForOp loop,
                                 int64_t rows, int64_t dotRows)
    : rewriter(rewriter), original(loop), rows(rows), dotRows(dotRows) {}

// 在旧循环之前新建一个循环：初值、循环参数和结果都换成扁平布局，逐句重写
// 循环体后把结果 view 回逻辑布局，最后删除旧循环。
// Builds a replacement loop whose initial values, iter args and results use the
// flattened layout, rewrites the body statement by statement, views the results
// back to the logical layout and erases the original loop.
void LoopRowFlattener::run() {
  rewriter.setInsertionPoint(original);
  SmallVector<Value> initial;
  for (Value value : original.getInitArgs())
    initial.push_back(get(value, flatType(value.getType())));
  replacement = rewriter.create<scf::ForOp>(
      original.getLoc(), original.getLowerBound(), original.getUpperBound(),
      original.getStep(), initial);
  replacement->setAttrs(original->getAttrs());
  mapping.map(original.getInductionVar(), replacement.getInductionVar());
  mapping.map(original.getRegionIterArgs(), replacement.getRegionIterArgs());
  rewriteBlock(original.getBody(), replacement.getBody(),
               replacement.getResultTypes());

  rewriter.setInsertionPointAfter(replacement);
  for (auto [oldValue, newValue] :
       llvm::zip(original.getResults(), replacement.getResults())) {
    Value result = view(newValue, oldValue.getType());
    oldValue.replaceAllUsesWith(result);
  }
  rewriter.eraseOp(original);
  rewriter.setInsertionPointAfter(replacement);
}

// 只有前两维恰好是 [rows, dotRows]、没有 encoding、且元素为 float/int 的张量
// 才折叠成 [rows*dotRows, ...]；其它类型原样返回，保持边界布局不变。
// Collapse [rows, dotRows, ...] into [rows*dotRows, ...] only for plain
// float/integer tensors whose leading two extents match the loop shape; every
// other type is returned unchanged and keeps its boundary layout.
Type LoopRowFlattener::flatType(Type type) const {
  auto tensor = dyn_cast<RankedTensorType>(type);
  if (!tensor || tensor.getRank() < 2 || tensor.getEncoding() ||
      tensor.getShape()[0] != rows || tensor.getShape()[1] != dotRows ||
      !isa<FloatType, IntegerType>(tensor.getElementType()))
    return type;
  SmallVector<int64_t> shape{rows * dotRows};
  llvm::append_range(shape, tensor.getShape().drop_front(2));
  return RankedTensorType::get(shape, tensor.getElementType());
}

// 把 `value` 转成目标类型：类型已一致时零开销返回，否则插入一条逻辑上
// 等价的 tt.reshape 作为布局边界。
// Converts `value` to `type`: returns it directly when the type already
// matches, otherwise inserts an equivalent tt.reshape as the layout boundary.
Value LoopRowFlattener::view(Value value, Type type) {
  if (value.getType() == type)
    return value;
  return rewriter.create<triton::ReshapeOp>(
      value.getLoc(), cast<RankedTensorType>(type), value, false);
}

// 取 `value` 在目标布局下的表示：类型已一致直接用映射结果；循环外定义的
// 不变量（含 softmax 递推用到的常量）在循环外建立并缓存边界视图；循环内
// 的值则就地插入视图。
// Returns `value` in the requested layout: a matching mapped value is used
// as-is, an invariant defined outside the loop gets its boundary view built
// and cached above the loop, and a loop-local value is viewed in place.
Value LoopRowFlattener::get(Value value, Type type) {
  Value mapped = mapping.lookupOrDefault(value);
  if (mapped.getType() == type)
    return mapped;
  // 循环外捕获值和初值都是不变量，它们的边界视图应建在循环之外，softmax
  // 递推用到的常量也一样。
  // Captures and initial values are invariant: their boundary views belong
  // above the loop, including constants used by the softmax recurrence.
  if (original.isDefinedOutsideOfLoop(value)) {
    auto found = captures.find(value);
    if (found != captures.end() && found->second.getType() == type)
      return found->second;
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(replacement ? replacement : original);
    Value result = flattenCapture(value, type);
    if (!result)
      result = view(mapped, type);
    captures[value] = result;
    return result;
  }
  return view(mapped, type);
}

// 让循环外的不变量计算直接产出与循环内消费者一致的布局。如果先按 [R,M,N]
// 算完再 view，bufferization 可能同时保留两种布局（Q 缩放和 lse 尤其明显）。
// 只有单结果、结果类型可折叠，且操作是 expand_dims / elementwise / broadcast
// / splat 时才原位重算；否则返回空，交由调用方退回普通 view。
// Keep invariant computation in the same layout as its loop consumers.
// Viewing a fully broadcast [R,M,N] result only after computing it can make
// bufferization materialize both layouts, notably for Q scaling and lse.
Value LoopRowFlattener::flattenCapture(Value value, Type type) {
  Operation *op = value.getDefiningOp();
  if (!op || op->getNumResults() != 1 || flatType(value.getType()) != type)
    return {};
  // 可折叠的 expand_dims 在循环外重算：轴号左移一位，源先在扁平布局下取到。
  // A collapsible expand_dims is rebuilt outside the loop with its axis
  // shifted left by one and its source fetched in the flat layout.
  if (auto expand = dyn_cast<triton::ExpandDimsOp>(op)) {
    if (expand.getAxis() >= 2 &&
        flatType(expand.getSrc().getType()) != expand.getSrc().getType())
      return rewriter.create<triton::ExpandDimsOp>(
          expand.getLoc(),
          get(expand.getSrc(), flatType(expand.getSrc().getType())),
          expand.getAxis() - 1);
    return {};
  }
  bool elementwise =
      op->hasTrait<OpTrait::Elementwise>() && op->getNumRegions() == 0;
  if (!elementwise && !isa<triton::BroadcastOp, triton::SplatOp>(op))
    return {};
  if (!llvm::all_of(op->getOperandTypes(), [&](Type operandType) {
        return !isa<RankedTensorType>(operandType) ||
               flatType(operandType) != operandType;
      }))
    return {};
  IRMapping operands;
  for (Value operand : op->getOperands())
    operands.map(operand, get(operand, flatType(operand.getType())));
  Operation *next = rewriter.clone(*op, operands);
  next->getResult(0).setType(type);
  return next->getResult(0);
}

// 把 `from` 中的非终结指令逐条重写到 `to`，最后按 `yieldTypes` 把原 yield
// 的操作数转成新布局，生成新的 scf.yield。
// Rewrites every non-terminator of `from` into `to`, then emits a new
// scf.yield whose operands are converted to `yieldTypes`.
void LoopRowFlattener::rewriteBlock(Block *from, Block *to,
                                   TypeRange yieldTypes) {
  rewriter.setInsertionPointToEnd(to);
  if (!to->empty() && isa<scf::YieldOp>(to->back()))
    rewriter.eraseOp(to->getTerminator());
  for (Operation &op : from->without_terminator())
    rewrite(&op);
  auto yield = cast<scf::YieldOp>(from->getTerminator());
  SmallVector<Value> values;
  for (auto [value, type] : llvm::zip(yield.getOperands(), yieldTypes))
    values.push_back(get(value, type));
  rewriter.create<scf::YieldOp>(yield.getLoc(), values);
}

// 单条指令的布局重写分派：scf.if / tt.reduce / expand_dims / reshape 需要
// 结构性重建，其余 elementwise、broadcast、splat 只克隆后改结果类型。
// Per-op dispatch: scf.if, tt.reduce, expand_dims and reshape need structural
// rebuilding; other elementwise, broadcast and splat ops are cloned and have
// their result types updated.
void LoopRowFlattener::rewrite(Operation *op) {
  // scf.if 的结果按扁平类型重建，两个分支用同一组结果类型重写；条件通常
  // 是标量，也统一走 get 以保持映射一致。
  // Rebuild scf.if with flattened result types and rewrite both branches
  // against them; the condition is fetched through `get` as well.
  if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    SmallVector<Type> types;
    for (Type type : ifOp.getResultTypes())
      types.push_back(flatType(type));
    auto next = rewriter.create<scf::IfOp>(
        ifOp.getLoc(), types,
        get(ifOp.getCondition(), ifOp.getCondition().getType()),
        !ifOp.getElseRegion().empty());
    next->setAttrs(ifOp->getAttrs());
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriteBlock(ifOp.thenBlock(), next.thenBlock(), types);
      if (!ifOp.getElseRegion().empty())
        rewriteBlock(ifOp.elseBlock(), next.elseBlock(), types);
    }
    mapping.map(ifOp.getResults(), next.getResults());
    return;
  }

  if (auto reduce = dyn_cast<triton::ReduceOp>(op)) {
    // 折叠 R 和 M 不能改变任何一维上的归约语义，因此归约轴 >= 2 时才把轴号
    // 前移一位并在扁平布局上重建，否则留在逻辑布局边界上。
    // Collapsing R and M must never change a reduction over either axis.
    // Such a reduction stays at the logical-layout boundary instead.
    if (reduce.getAxis() >= 2 &&
        llvm::all_of(reduce.getSrcs(), [&](Value value) {
          return flatType(value.getType()) != value.getType();
        })) {
      SmallVector<Value> sources;
      for (Value value : reduce.getSrcs())
        sources.push_back(get(value, flatType(value.getType())));
      auto next = rewriter.create<triton::ReduceOp>(reduce.getLoc(), sources,
                                                    reduce.getAxis() - 1);
      rewriter.cloneRegionBefore(reduce.getCombineOp(), next.getCombineOp(),
                                 next.getCombineOp().end());
      mapping.map(reduce.getResults(), next.getResults());
      return;
    }
  }

  // 折叠前两维后轴号左移一位；源不可折叠时保持原样，不在这里改写。
  // The axis shifts left by one after the collapse; an unsplittable source is
  // left untouched here.
  if (auto expand = dyn_cast<triton::ExpandDimsOp>(op)) {
    if (expand.getAxis() >= 2 &&
        flatType(expand.getSrc().getType()) != expand.getSrc().getType()) {
      auto next = rewriter.create<triton::ExpandDimsOp>(
          expand.getLoc(),
          get(expand.getSrc(), flatType(expand.getSrc().getType())),
          expand.getAxis() - 1);
      mapping.map(expand.getResult(), next.getResult());
      return;
    }
  }

  if (auto reshape = dyn_cast<triton::ReshapeOp>(op)) {
    // 这里也覆盖 dot 进出计算区域的视图：两侧都已是计算布局时，视图会直接
    // 消失，而不是每轮迭代重建一次。
    // This includes the dot's entry/exit views. Once both sides have the
    // compute layout, the view disappears instead of being recreated per trip.
    Value source = mapping.lookupOrDefault(reshape.getSrc());
    Type type = flatType(reshape.getType());
    if (original.isDefinedOutsideOfLoop(reshape.getSrc()))
      source = get(reshape.getSrc(), flatType(reshape.getSrc().getType()));
    mapping.map(reshape.getResult(), view(source, type));
    return;
  }

  // 普通指令：elementwise、broadcast、splat 在结果可折叠时也进入扁平布局，
  // 否则按原类型克隆。broadcast 只在第 0 个操作数可折叠（源本身就按扁平
  // 布局广播）时才算可折叠。
  // Plain ops: elementwise, broadcast and splat also join the flat layout when
  // their results can be collapsed; a broadcast additionally requires operand
  // 0 to be collapsible, i.e. the source itself broadcasts in the flat layout.
  bool elementwise =
      op->hasTrait<OpTrait::Elementwise>() && op->getNumRegions() == 0;
  bool broadcast =
      isa<triton::BroadcastOp>(op) &&
      flatType(op->getOperand(0).getType()) != op->getOperand(0).getType();
  bool flatten = (elementwise || broadcast || isa<triton::SplatOp>(op)) &&
                 llvm::all_of(op->getResultTypes(), [&](Type type) {
                   return flatType(type) != type;
                 });
  // 被折叠的 elementwise / broadcast 指令，其每个张量操作数都必须带上两个
  // 逻辑行轴。逐查询的掩码或权重先在逻辑布局里广播，再在边界处 view 进
  // 计算区域。只有 splat 的源是标量，不受此限。
  // Every tensor operand of a collapsed elementwise/broadcast op must carry
  // both logical row axes. A per-query mask or weight is broadcast in its
  // logical layout first, then viewed at the boundary into the compute region.
  if (flatten && !isa<triton::SplatOp>(op))
    flatten = llvm::all_of(op->getOperandTypes(), [&](Type type) {
      return !isa<RankedTensorType>(type) || flatType(type) != type;
    });

  IRMapping operands = mapping;
  for (Value operand : op->getOperands())
    operands.map(operand, get(operand, flatten ? flatType(operand.getType())
                                               : operand.getType()));
  Operation *next = rewriter.clone(*op, operands);
  // 克隆复用同一操作，结果类型需手工改成折叠后的形状。
  // The cloned results still carry the original logical shape.
  if (flatten)
    for (Value result : next->getResults())
      result.setType(flatType(result.getType()));
  mapping.map(op->getResults(), next->getResults());
}

// 收集函数内所有循环，逐个找出 dot 结果被 reshape 成 [rows, M, ...] 的 M；
// 同一循环内各 dot 给出的 M 必须一致，确认后对该循环做计算区域扁平化。
// Walks every loop in the function, derives the dot M from the reshape of a dot
// result to [rows, M, ...] and requires one consistent M per loop before
// flattening that loop's compute region.
static void flattenLoopCompute(IRRewriter &rewriter, triton::FuncOp fn,
                               int64_t rows) {
  SmallVector<scf::ForOp> loops;
  fn.walk([&](scf::ForOp loop) { loops.push_back(loop); });
  for (scf::ForOp loop : loops) {
    std::optional<int64_t> dotRows;
    bool consistent = true;
    loop.walk([&](triton::DotOp dot) {
      for (Operation *user : dot.getResult().getUsers()) {
        auto reshape = dyn_cast<triton::ReshapeOp>(user);
        if (!reshape)
          continue;
        auto type = cast<RankedTensorType>(reshape.getType());
        if (type.getRank() != 3 || type.getShape()[0] != rows)
          continue;
        int64_t m = type.getShape()[1];
        consistent &= !dotRows || *dotRows == m;
        dotRows = m;
      }
    });
    if (dotRows && consistent)
      LoopRowFlattener(rewriter, loop, rows, *dotRows).run();
  }
}

//===----------------------------------------------------------------------===//
// 5. Split wide output accumulators and descriptors along the width
//===----------------------------------------------------------------------===//

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
  if (!ifOp)
    return Value();
  return ifOp.getResult(use.getOperandNumber());
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

// 同时检查并收集一个完整的累加器岛：dot 的 C、store 的 value、逐元素算术、
// 本循环结果和 scf.if 的结果。岛内每个值只访问一次，岛外使用立即拒绝。
// dot 的 B 与输出 store 必须来自同宽块指针，半宽值由访存结构直接产生，
// 不靠 reshape/trans/split 重新拼装。记录原始值供 if/循环在建类型前决定拆半。
//
// 原来的合法性检查在 dot 和本循环 yield 处停止，收集路径却继续向后访问。
// 这里也校验这些后继，避免 matcher 放行、到重建时才发现消费者不能处理半宽值。
// 必须在循环体内遇到供给该累加器的 dot；仅在循环外出现 dot 不构成拆分收益。
// Inspect and collect the entire island once, including dot and loop results.
// Reject unsupported uses and incompatible descriptors before rewriting starts.
// The dot must feed this loop's accumulator, rather than only a post-loop value.
static bool inspectSplitIsland(Value root, scf::ForOp loop, int64_t width,
                               SmallVectorImpl<Value> &descriptors,
                               DenseSet<Value> &island) {
  SmallVector<Value> work{root};
  bool sawLoopDot = false;
  while (!work.empty()) {
    Value value = work.pop_back_val();
    if (!island.insert(value).second)
      continue;
    for (OpOperand &use : value.getUses()) {
      Operation *user = use.getOwner();
      if (auto dot = dyn_cast<triton::DotOp>(user)) {
        // 只沿 C 拆输出宽度；A/B 是矩阵乘的输入，不能按这条链处理。
        auto load = dot.getB().getDefiningOp<triton::LoadOp>();
        if (use.get() != dot.getC() || !load ||
            !descriptorHalvesAtWidth(load.getPtr(), width))
          return false;
        descriptors.push_back(load.getPtr());
        sawLoopDot |= loop->isProperAncestor(dot.getOperation());
        work.push_back(dot.getResult());
        continue;
      }
      if (auto store = dyn_cast<triton::StoreOp>(user)) {
        if (use.get() != store.getValue() ||
            !descriptorHalvesAtWidth(store.getPtr(), width))
          return false;
        descriptors.push_back(store.getPtr());
        continue;
      }
      if (isa<scf::YieldOp>(user)) {
        // 背边沿对应的循环结果走；if 分支沿对应的 if 结果走。
        Value result = user->getParentOp() == loop.getOperation()
                           ? loop.getResult(use.getOperandNumber())
                           : ifResultForYield(use);
        if (!result)
          return false;
        work.push_back(result);
        continue;
      }
      StringRef dialect = user->getName().getDialectNamespace();
      if (user->getNumResults() != 1 || user->getNumRegions() != 0 ||
          (dialect != "arith" && dialect != "math"))
        return false;
      work.push_back(user->getResult(0));
    }
  }
  return sawLoopDot;
}

// 不生成 IR 的拆分计划。匹配期只读这一结果的成败，重写期一次性接收索引、
// 半宽映射和描述符。初值也登记半宽，避免先生成整宽提升值再拆。
// A read-only analysis product shared by matcher preflight and rewrite planning.
struct WideAccumulatorSplitPlan {
  SmallVector<int64_t> indices;
  DenseMap<Value, int64_t> halfWidths;
  SmallVector<Value> descriptors;
};

// 两个阶段共用字节计算、预算、偶数宽、岛与描述符检查，以及重叠岛半宽一致性。
// 对原始张量逐维做受检乘法，再乘 R，等价于计算 [R]+shape 的总字节数，且
// 无需创建提升类型。每一步均保留溢出保护，不能假定公共入口的 R 永远不超过 8。
// bytes 已超过非负 budget 时，用 bytes-budget > budget 检查两半是否够小，
// 避免 2*budget 本身溢出。标量和原始 rank<2 的值保持不拆的历史路径。
// Compute the lifted size with checked multiplication in one place. Check
// halving without multiplying the budget, which may itself approach INT64_MAX.
static bool analyzeWideAccumulatorSplits(scf::ForOp loop, int64_t rows,
                                         int64_t budget,
                                         WideAccumulatorSplitPlan &plan) {
  if (rows <= 0 || budget < 0)
    return false;
  for (auto [index, arg] : llvm::enumerate(loop.getRegionIterArgs())) {
    auto tensor = dyn_cast<RankedTensorType>(arg.getType());
    if (!tensor || tensor.getRank() < 2)
      continue;
    int64_t bytes = getElementStorageBytes(tensor.getElementType());
    if (!tensor.hasStaticShape() || bytes <= 0)
      return false;
    for (int64_t extent : tensor.getShape())
      if (__builtin_mul_overflow(bytes, extent, &bytes))
        return false;
    if (__builtin_mul_overflow(bytes, rows, &bytes))
      return false;
    if (bytes <= budget)
      continue;
    int64_t width = tensor.getShape().back();
    if (bytes - budget > budget || width < 2 || width % 2 != 0)
      return false;
    DenseSet<Value> island;
    if (!inspectSplitIsland(arg, loop, width, plan.descriptors, island))
      return false;
    // scf.for 已保证初值与 iter arg 同型；把初值一并登记，不再重复插入逻辑。
    island.insert(loop.getInitArgs()[index]);
    for (Value value : island) {
      auto [it, inserted] = plan.halfWidths.try_emplace(value, width / 2);
      if (!inserted && it->second != width / 2)
        return false;
    }
    plan.indices.push_back(static_cast<int64_t>(index));
  }
  return true;
}

// 匹配阶段复用完整分析，不再维护另一份大小和累加器岛判据。
bool canSplitWideAccumulators(scf::ForOp loop, int64_t rowsPerProgram,
                              bool enabled, int64_t budgetBytes) {
  WideAccumulatorSplitPlan plan;
  return !enabled ||
         analyzeWideAccumulatorSplits(loop, rowsPerProgram, budgetBytes, plan);
}

// 在生成任何 IR 前接收分析结果，描述符此时尚未按整宽展开。
bool RowLiftBuilder::planWideAccumulatorSplits(scf::ForOp loop) {
  if (!features.splitWideAccumulator)
    return true;
  WideAccumulatorSplitPlan plan;
  if (!analyzeWideAccumulatorSplits(loop, candidate.rowsPerProgram,
                                    candidate.splitValueBudgetBytes, plan))
    return false;
  plannedSplitIndices = std::move(plan.indices);
  plannedHalfWidths = std::move(plan.halfWidths);
  for (Value descriptor : plan.descriptors)
    splitDescriptors.try_emplace(descriptor, SplitDescriptor{});
  LLVM_DEBUG(llvm::dbgs() << "row-lift: planned " << plannedSplitIndices.size()
                          << " split indices, " << splitDescriptors.size()
                          << " descriptors\n");
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

// 取一个已规划拆半值的两个半宽提升值。半宽来自规划阶段登记的 plannedHalfWidths，
// 因此类型不必依赖重写期的 splitValues；已经登记过两半的值直接复用。
// Return the two half-width lifted values of a value planned for halving: the
// half width comes from `plannedHalfWidths`, filled before any IR was generated,
// and an already registered pair is reused as-is.
bool RowLiftBuilder::ensureRowHalves(Value value, SplitValue &halves) {
  if (auto it = splitValues.find(value); it != splitValues.end()) {
    halves = it->second;
    return true;
  }
  auto plan = plannedHalfWidths.find(value);
  if (plan == plannedHalfWidths.end())
    return false;
  auto original = dyn_cast<RankedTensorType>(value.getType());
  if (!original || original.getRank() < 2)
    return false;
  // 半宽提升类型：行维 + 原形状，末维换成规划登记的半宽。
  auto halfType = halvedType(getLiftedType(original));
  if (!halfType || halfType.getShape().back() != plan->second)
    return false;

  // 取值在各列相同的初值可以结构折叠成同一个半宽值，不产生全宽中间值。
  if (Value half = rebuildHalfOperand(value, halfType)) {
    halves = {half, half};
    return true;
  }

  // 当前 NSA 的初值可由常量/广播直接折叠。没有结构化半宽来源就拒绝该候选，
  // 不先物化整宽张量，再通过 reshape/trans/split 拆分。
  return false;
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
          loc,
          DenseElementsAttr::get(halfType, dense.getSplatValue<APFloat>()));
    if (isa<IntegerType>(element))
      return rewriter.create<arith::ConstantOp>(
          loc, DenseElementsAttr::get(halfType, dense.getSplatValue<APInt>()));
    return Value();
  }
  // splat / broadcast / expand_dims 的源补上行维后，只要沿最后一维退化
  // 就能由同一个半宽值代表两半；该退化条件统一由 collapseAcrossLastDim 判定。
  // A splat, broadcast or expand_dims source can be shared by both halves once
  // its row dimension is added and it degenerates along the last axis.
  Operation *source = original.getDefiningOp();
  if (!isa_and_nonnull<triton::SplatOp, triton::BroadcastOp,
                       triton::ExpandDimsOp>(source))
    return Value();
  Value src = source->getOperand(0);
  src = ensureRowDimension(src);
  return collapseAcrossLastDim(src, halfType) ? broadcastRowValue(src, halfType)
                                              : Value();
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
bool RowLiftBuilder::rewriteSplitTensorPointer(triton::MakeTensorPtrOp makePtr,
                                               SplitDescriptor &descriptor) {
  if (!features.tensorPointer)
    return false;
  auto pointerType = dyn_cast<triton::PointerType>(makePtr.getType());
  auto blockType =
      pointerType ? dyn_cast<RankedTensorType>(pointerType.getPointeeType())
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
  auto halfPointerType =
      triton::PointerType::get(halfBlock, pointerType.getAddressSpace());

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
    halves.push_back(
        rewriter
            .create<triton::LoadOp>(load.getLoc(), pointer, Value(), Value(),
                                    load.getBoundaryCheck(), load.getPadding(),
                                    load.getCache(), load.getEvict(),
                                    load.getIsVolatile())
            .getResult());
  }
  splitValues[load.getResult()] = {halves[0], halves[1]};
  return true;
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
  auto flatType =
      RankedTensorType::get({flatRows, accumulatorType.getShape()[2]},
                            accumulatorType.getElementType());

  // 对每个半宽各发一次 dot：A 与 B 的对应半宽、以及累加器的对应半宽。
  SmallVector<Value> operands0{a, b->second.first,
                               flattenDotRows(accumulator->second.first)};
  SmallVector<Value> operands1{a, b->second.second,
                               flattenDotRows(accumulator->second.second)};
  SmallVector<Value> halves;
  for (const SmallVector<Value> &operands : {operands0, operands1}) {
    Operation *created =
        rewriter.create(dot.getLoc(), dot->getName().getIdentifier(), operands,
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
  auto halfType = halvedType(getLiftedType(resultType));
  if (!halfType)
    return false;

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
    auto operandHalfType =
        halvedType(getLiftedType(cast<RankedTensorType>(operand.getType())));
    if (!operandHalfType)
      return false;
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
    results.push_back(rewriter
                          .create(op->getLoc(), op->getName().getIdentifier(),
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
  const Value halfValues[] = {valueIt->second.first, valueIt->second.second};

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
      if (!boundaryAxesInTile(boundaryCheck, half->tileShape))
        return fail("boundary axis out of tile rank");
    }
    for (auto [index, half] : llvm::enumerate(halves)) {
      auto pointer = half->pointer;
      auto pointerType = cast<RankedTensorType>(pointer.getType());
      if (pointerType.getShape() != valueType.getShape())
        return fail("expanded pointer shape != value shape");
      auto element =
          dyn_cast<triton::PointerType>(pointerType.getElementType());
      if (!element || element.getPointeeType() != valueType.getElementType())
        return fail("expanded pointer element mismatch");
      auto indexType =
          RankedTensorType::get(pointerType.getShape(), rewriter.getI32Type());
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
      Value value = halfValues[index];
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
    Value value = halfValues[index];
    rewriter.create<triton::StoreOp>(loc, pointer, value, Value(),
                                     store.getBoundaryCheck(), store.getCache(),
                                     store.getEvict());
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Builder entry and operation dispatch
//===----------------------------------------------------------------------===//

// 保存重写器与候选描述，能力开关从候选同步过来。
RowLiftBuilder::RowLiftBuilder(IRRewriter &rewriter, RowLiftCandidate candidate)
    : rewriter(rewriter), candidate(candidate), features(candidate.features) {}

// namespace

// 克隆函数体并逐条重建，成功时用新函数体替换旧函数体并返回 true。
//
// 重建的单位是「函数级」而不是单条指令：先把原 block 的每条指令按行提升
// 语义重放一遍，全部成功后才丢弃旧 block。失败时可能留下部分新 block，
// 调用方必须丢弃私有沙箱，不能把失败后的 fn 当作未修改的原始 IR。
bool RowLiftBuilder::run(triton::FuncOp fn) {
  Block *old = &fn.getBody().front();
  // 原循环的上界。在输入不回绕的约定下证明非降，重算最后一个有效行。
  // 公共引擎要求函数体内恰好有一个循环；没有循环时直接拒绝，避免解引用
  // 空迭代器（调用方的 matcher 虽然已经保证 loops==1，但这里仍按 API 契约判空）。
  auto loops = old->getOps<scf::ForOp>();
  if (loops.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "row-lift: no loop to lift\n");
    return false;
  }
  scf::ForOp loop = *loops.begin();
  Value upperBound = loop.getUpperBound();
  if (!canUseLastRowLoopBound(loop, candidate.anchor->getResult(0),
                              candidate.grid))
    return false;
  // 累加器拆分必须在重写任何指令之前定下来：要拆半的描述符可能位于循环
  // 之前，一旦先按整宽展开成指针张量，就无法再按半宽重建。
  // The accumulator split has to be decided before any instruction is rewritten:
  // a descriptor above the loop would otherwise be expanded at full width and
  // could no longer be rebuilt per half.
  plannedSplitIndices.clear();
  plannedHalfWidths.clear();
  if (!planWideAccumulatorSplits(loop))
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
          rewriter.create<arith::ConstantIntOp>(op.getLoc(),
                                                candidate.rowsPerProgram, 32));
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
      valueMapping.map(op.getResult(0),
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
        Value rowsLeft =
            rewriter.create<arith::SubIOp>(op.getLoc(), extent, base);
        Value limit =
            rewriter.create<triton::SplatOp>(op.getLoc(), t, rowsLeft);
        validRows = rewriter.create<arith::CmpIOp>(
            op.getLoc(), arith::CmpIPredicate::slt, lanes, limit);
      }
      continue;
    }

    // 只把无 region 的纯标量上界算术复制到末行映射；与行无关的结果
    // 在普通重写后加入映射。不能按这条路径重建的上界由策略选择或调用方回退。
    if (op.getNumResults() == 1 &&
        !isa<RankedTensorType>(op.getResult(0).getType()) &&
        isScalarBoundSlice(&op) &&
        dependsOnValueThroughRegions(upperBound, op.getResult(0)))
      rewriter.clone(op, lastRowMapping);

    // 逐条重建；任一形态不被支持即失败，由调用方丢弃整个沙箱。
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
  if (features.flatCompute)
    flattenLoopCompute(rewriter, fn, candidate.rowsPerProgram);
  return true;
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

  // 步骤 2：判断本指令是否与查询行相关。
  //   descriptor —— 操作数是已展开的块指针，必须走描述符访存路径；
  //   dependent  —— 操作数或嵌套 region 里用到了带行维的值。
  bool descriptor = llvm::any_of(op->getOperands(), [&](Value v) {
    return liftedDescriptors.contains(v);
  });
  bool dependent = descriptor || llvm::any_of(op->getOperands(), [&](Value v) {
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

  // 步骤 4b：循环内逐行 load 需要活动掩码；运行期网格的循环外 load
  // 需要末组有效行掩码。静态网格的循环外 load 保留通用重建路径。
  if (auto load = dyn_cast<triton::LoadOp>(op))
    if (op->getParentOfType<scf::ForOp>() || candidate.runtimeGrid)
      return rewriteRowLoad(load);

  // 步骤 4c：dot。只有 A 与查询行相关、B 与行无关时才提升：把 [R,M,K] 展平
  // 成 [R*M,K] 复用普通 dot，结果再 reshape 回 [R,M,N]。
  if (auto dot = dyn_cast<triton::DotOp>(op)) {
    if (!hasRowDimension(dot.getA()) || hasRowDimension(dot.getB()))
      return false;

    auto result = cast<RankedTensorType>(dot.getType());
    SmallVector<Value> operands{flattenDotRows(ensureRowDimension(dot.getA())),
                                getMappedValue(dot.getB()),
                                flattenDotRows(ensureRowDimension(dot.getC()))};
    Type flatType = RankedTensorType::get(
        {candidate.rowsPerProgram * result.getShape()[0], result.getShape()[1]},
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
    valueMapping.map(expand.getResult(),
                     rewriter
                         .create<triton::ExpandDimsOp>(
                             op->getLoc(), ensureRowDimension(expand.getSrc()),
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
            .create<triton::TransOp>(op->getLoc(),
                                     ensureRowDimension(trans.getSrc()), order)
            .getResult());
    return true;
  }

  // 步骤 4h：带掩码的 store（O、lse 的写回）只有在 maskedStore 能力开启时
  // 才提升；否则通用克隆会构造出形状与提升后指针不匹配的 mask。
  // A masked store is lifted only when the capability is enabled; the generic
  // clone below would otherwise build a mask whose shape no longer matches the
  // lifted pointer.
  if (auto store = dyn_cast<triton::StoreOp>(op)) {
    if (candidate.runtimeGrid || (store.getMask() && features.maskedStore)) {
      return rewriteMaskedStore(store);
    }
    // 循环体内没有掩码的 store 必须失败关闭：提升后的循环跑到组内共享上界，
    // 扫描已提前结束的行在本轮仍会执行到这条 store，通用克隆会把整段
    // [R, ...] 写回，而它们的结果本应保持旧值。
    // An unmasked store inside the loop has to fail closed: the lifted loop runs
    // to the group's shared bound, so rows whose scan already ended still reach
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
  // A partial group evaluates padded rows as well. A no-wrap promise on the
  // original scalar need not hold in those inactive lanes; dropping it preserves
  // all defined results and prevents poison from entering their validity masks.
  if (auto flags = dyn_cast<arith::ArithIntegerOverflowFlagsInterface>(cloned))
    cloned->setAttr(flags.getIntegerOverflowAttrName(),
                    arith::IntegerOverflowFlagsAttr::get(
                        rewriter.getContext(), arith::IntegerOverflowFlags::none));
  valueMapping.map(op->getResults(), cloned->getResults());
  return true;
}

} // namespace cfg
} // namespace triton
} // namespace mlir
