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
#include <limits>
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

// 元素类型的存储字节数（指针按 8 字节）。
// Storage bytes of one element; a pointer occupies eight bytes.
static int64_t elementStorageBytes(Type element) {
  return isa<triton::PointerType>(element)
             ? 8
             : (element.getIntOrFloatBitWidth() + 7) / 8;
}

// 计算 dot tile 的存储字节数，用作 L0A/L0C 的每因子需求。
static int64_t tensorStorageBytes(RankedTensorType tensor) {
  Type element = tensor.getElementType();
  const int64_t bytes = elementStorageBytes(element);
  int64_t size = 0;
  if (__builtin_mul_overflow(tensor.getNumElements(), bytes, &size))
    return std::numeric_limits<int64_t>::max();
  return size;
}

// 识别可按规则 B 重建的查询行循环，返回候选（行数、能力开关、静态网格）。
// 本函数是「是否重写」的唯一判定入口：调度器入口 createQueryRowBatchingPlan
// 与规则 A 的 materializeQueryRowBatchingCandidate 都先走这里，返回 nullopt
// 就表示不改写。它只做形态 / 资源判定与 R 计算，不修改 IR。
//
// Matcher for rule B's query-row lift.  This is the single "should we rewrite?"
// entry shared by the scheduler and rule A's materialize entry; it only decides
// and derives R, and never edits the IR.  Every check is fail-closed.
//
// 判据按三层组织，任一层不满足都直接拒绝：
//
// 1) 启动契约门（函数开头的契约检查）。静态网格必须合法、函数体恰好一个 block、模块
//    内只有一个 public entry、模块上不能已有合并契约。前三条是重写器的实现
//    前提（它整体重建单入口函数）；后两条是幂等门，防止规则 A/B、IAT 或两次
//    调度重复提升。这一组条件与 DotRowCoalescing 的 matchDotRows 逐条相同，
//    区别只是那边把后两条写成字面量。
//
// 2) 结构门（walk 内逐 op 检查 + 走查结束后的后置条件）。只接受「单条查询行
//    循环 + 单 axis-0 pid」：pid 是行索引的唯一来源，多循环或多 pid 无法定义
//    一次 R 行合并。dot 必须是 rank-2 且 lhs 为 f16/bf16（cube 指令的类型
//    要求）；行相关的 dot 要求 A 已提升、B 未提升，因为 B 是跨查询行共享的
//    K/V tile，提升它没有收益，重写器也没有对应实现。循环必须在入口 block、
//    lb 为常量 0、step 为正常数，否则行号与有效位无法按仿射关系重建。
//    与 DotRowCoalescing 相比，B 额外允许 tt.reduce（bwd dq 形态）、条件与
//    pid 无关的 scf.if、tensor pointer、循环内行相关 load（由 maskedLoad 承担）
//    以及多个 store；DotRow 则要求 reduce 数为 0、store 恰好 1 个、循环内
//    不得有行相关 load，并要求输出地址是 pid 的仿射函数。这些放宽项的来源
//    不是 RFC，而是 NSA 与 bwd kernel 的真实形态；代价是重写路径必须提供
//    逐行掩码能力。
//
// 3) 资源与收益门（构造候选并选行的部分）。R 由 RFC #2686 第 4 节的四条阈值推出：
//    R·G×BK×elemBytes ≤ L0A、R·G×BN×4 ≤ L0C、单值预算（UB 代理）、
//    gridX/R ≥ coreNum；selectRowsPerProgram 取全部上界的最小值后向下取
//    2 的幂（实现层的重试按折半走，RFC 本身未要求 2 的幂）。R <= 1 表示没有
//    可合并的行，直接拒绝。最后用 canSplitWideAccumulators 做一次预检，避免
//    注定失败的宽累加器形态进入沙箱克隆与 verify。
//
// 参数语义：
// - l0aBudgetBytes / l0cBudgetBytes：资源快照中的安全预算（容量 × 安全
//   百分比）；0 表示目标架构的立方容量未知，对应的 L0 约束关闭（RFC 允许的
//   降级），不是「预算为零」的拒绝条件。
// - coreNum：物理核数，用于 gridX/R >= coreNum；运行期网格 grid 未知时为 0，
//   该约束自动不参与定界。
// - forcedRows：调用方钉住的因子。UB 超预算时上层按 2 的幂下调重试，重试
//   必须让所有依赖行数的判据（L0 需求、单值预算、拆分判据）使用同一个因子。
// - allowSplit：是否允许结构性拆半（宽累加器拆成两条半宽链）。它与 R 的
//   取值互相影响，所以由本函数统一算出并写进候选。
//
// 「循环是否已顺序化」不是这里的判据：按下标表读的稀疏循环由规则 A 负责
// 顺序化，A 改完之后才调用 B。B 自己调度时若遇到未顺序化的循环，会在重写
// 路径的能力开关（features.tableRead 关闭）处失败，而不是在匹配阶段拒绝，
// 这样 A/B 的判定口径只有一份，不会因为形态代理条件而漂移。
static std::optional<RowLiftCandidate>
matchQueryRows(triton::FuncOp fn, uint64_t l0aBudgetBytes,
               uint64_t l0cBudgetBytes, uint64_t coreNum,
               std::optional<int64_t> forcedRows = std::nullopt,
               bool allowSplit = true) {
  ModuleOp module = fn->getParentOfType<ModuleOp>();
  if (!module)
    return std::nullopt;

  auto grid = module->getAttrOfType<IntegerAttr>("hacc.grid_num_tiles");
  // 启动契约门与幂等门：静态网格必须合法（否则启动契约本身已坏），函数体
  // 单 block、唯一 public entry（重写器按单入口整体重建）。模块上已有
  // coalesce 因子或网格变换说明别的规则已经接管，再提升一次会叠加放大，
  // 所以直接拒绝。来源是重写器实现前提与幂等要求，RFC 不涉及。
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

  // ---- matcher 累积状态 -----------------------------------------------------
  // 下面的变量全部由紧随其后的前序遍历按定义序填充（计数与 max 单调更新，
  // 句柄在对应 op 首次出现时记录），遍历结束后供后置结构门、R 决策与候选组装
  // 使用。max 类初值为 1：既是 std::max 的单位元，也让 budget / demand 不会
  // 除零，等价于该约束初始不生效。
  // Matcher accumulator state.  The pre-order walk below fills all of it; the
  // post gates, the row-count decision and the candidate assembly consume it.
  // max accumulators start at 1: identity for std::max and a safe divisor.
  //
  // axis-0 pid 句柄（pids==1 保证已赋值）：作候选锚点，并用于上界单调性证明与
  // scf.if 的行无关判定。
  triton::GetProgramIdOp pid;
  // 唯一的 scf.for（loops==1 保证已赋值）：取 lb/step、证明上界单调性、检查
  // 循环体可推测性，并在拆分预检里使用。
  scf::ForOp loop;
  // 入口 block 的第一个 store：后置门据此保证存在可写回的输出（B 不分析它的
  // 地址，逐行掩码写回由公共引擎负责）。
  triton::StoreOp output;
  // 结构计数：后置门要求 pids==1、loops==1、dots>=1、stores>=1。
  unsigned pids = 0, loops = 0, dots = 0, stores = 0;
  // 仅用于 LLVM_DEBUG 打印，不参与判定：B 允许 tensor pointer 与 reduce。
  unsigned blockPointers = 0, reduces = 0;
  // dot 的 M 维最大值：targetDotRows / maxDotRows 上界（B 的 targetDotRows=0，
  // 该条不生效）；存入候选后参与单值预算，并在重写时展开 M = maxDotRows × R。
  int64_t maxDotRows = 1;
  // dot 的 N 维（值维）最大值：存入候选，exceedsUBBudget 用它判断值宽是否
  // 超过一个向量寄存器，从而决定是否按 4 倍放大 UB 需求。
  int64_t maxDotCols = 1;
  // 累加器元素的字节数：把「列数」换算成一个向量寄存器等于多少列；存入候选，
  // 与 maxDotCols 相乘后和 kVectorRegisterBytes 比较。
  // Bytes per accumulator element: turns a column count into vector registers.
  int64_t accumulatorElemBytes = 4;
  // 所有张量结果的最大字节数：selectRowsPerProgram 的单值预算上界
  // budget / maxTensorBytes（拆半档预算翻倍）。
  int64_t maxTensorBytes = 1;
  // 原始 dot 的 lhs tile 最坏字节需求：写入 policy.maxL0ABytes，对应 L0A 上界
  // l0ABudgetBytes / maxL0ABytes。
  int64_t maxL0ABytes = 1;
  // 原始 dot 的累加器 tile（C/D）最坏字节需求：写入 policy.maxL0CBytes，
  // 对应 L0C 上界 l0CBudgetBytes / maxL0CBytes。
  int64_t maxL0CBytes = 1;
  // 任一 reject 或张量尺寸溢出即置 false；后置结构门的第一项。
  bool safe = true;
  // 行相关性集合：以 axis-0 pid、循环 iter args/results 为种子，按操作数传播
  // 到结果；用于判断 dot 的提升方向（只能提 A）与 scf.if 条件是否行无关。
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

    // 顺带量出所有张量结果的字节数：单值预算用它做 UB 代理上界，溢出则说明
    // shape 已经不可信（fail-closed，避免用回绕后的偏小需求算出过大的 R）。
    // Record the byte size of every tensor result: the per-value UB proxy reads
    // it, and an overflow means the shape is untrustworthy.
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
      // 只把 axis 0（x 轴）当作查询行索引：行提升沿 x 轴合并 R 个 program
      // （RFC #2686 第 4 节的 gridX / R）。其它轴的 pid 保持标量、被所有行
      // 共享，与 DotRowCoalescing 完全一致，因此不算违规；但后置条件要求
      // axis-0 pid 恰好出现一次，否则无法定义行号与有效位。
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
      // dot 是行提升的收益来源，也是 L0 需求的来源：A 必须 rank-2 且为
      // f16/bf16（cube 指令只接受这些输入类型）；若 dot 本身与查询行相关，
      // 只允许提升 A、不允许提升 B——B 是跨行共享的 K/V tile，提升它既没有
      // 收益也没有实现。A/C/D 的最大字节数分别记作 L0A 与 L0C 的每因子需求。
      // The dot is both the payoff and the source of the L0 demand.  A must be
      // rank-2 f16/bf16; a row-dependent dot may lift A but never B, since B is
      // the tile shared by every row.
      auto a = dyn_cast<RankedTensorType>(d.getA().getType());
      auto b = dyn_cast<RankedTensorType>(d.getB().getType());
      if (!a || !b || a.getRank() != 2 || b.getRank() != 2 ||
          !(a.getElementType().isF16() || a.getElementType().isBF16()) ||
          (dependent &&
           (!lifted.contains(d.getA()) || lifted.contains(d.getB()))))
        reject(d, "dot operand or shape");
      else {
        maxDotRows = std::max(maxDotRows, a.getShape()[0]);
        maxDotCols = std::max(maxDotCols, b.getShape()[1]);
        // 累加器元素宽度决定「一个向量寄存器等于多少列」；D 是累加器结果。
        // The accumulator element width decides how many columns one vector
        // register holds; D carries the accumulator type.
        if (auto res = dyn_cast<RankedTensorType>(d.getD().getType()))
          accumulatorElemBytes =
              std::max(accumulatorElemBytes,
                       elementStorageBytes(res.getElementType()));
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

    // 白名单里都是重写器能按行重建的「无副作用结构 op」：索引构造
    // （make_range/splat/expand_dims/broadcast/addptr）、布局调整
    // （trans/reshape/advance）、循环与函数骨架。tt.assert 被放行是因为它是
    // 断言而非计算，重写会保留它；其余 op 必须属于 arith/math 且无 region，
    // 有 region 的 op 无法表达逐行控制流。
    // Structural, side-effect-free ops the rewrite can rebuild per row.
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
  // 后置结构条件：单循环、单 axis-0 pid、至少一个 dot 和一个输出 store。
  // 与 DotRowCoalescing 的差别：B 允许 reduce（dq 形态）、允许多个 store
  // （只把入口 block 的第一个 store 当输出）以及 tensor pointer；DotRow 要求
  // reduce 数为 0、store 恰好 1 个，并额外检查输出地址对 pid 的仿射性。
  // 循环必须带进位状态（否则没有可提升的行相关值）、下界为常量 0、步长为
  // 正常数，重写才能把「第 i 行」映射回原 pid 与循环迭代。
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
  // 与查询行相关的块指针（Q）保留成块描述符，行维作为最外轴、行有效位交给
  // 边界检查的零填充：展开成指针张量会被后端降级成逐元素间接访存，并常驻
  // 一条 i64 地址向量，既占 UB 又拿不到 DMA 带宽。
  // A query-row-dependent block pointer (Q) stays a block descriptor with the
  // row axis outermost and row validity delegated to the boundary check's zero
  // padding: expanding it into a pointer tensor degrades to per-element indirect
  // access, keeps an i64 address vector resident (UB) and loses DMA bandwidth.
  candidate.features.rowDescriptor = true;
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

  // 行数完全交给公共策略，规则 B 不自带收益模型：M 维不设上限
  // （targetDotRows 保持 0），拆半能力把单值字节预算翻倍（两条半宽链）。
  // L0A/L0C 需求是正确性约束：重写会把 lhs 与累加器按因子放大，预算为 0
  // 表示容量未知、该约束关闭（RFC #2686 第 4 节）。
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
  policy.coreNum = coreNum;
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

  // 重写期要用的量留在候选里：提升后的 M 由原始 M 与因子相乘得到，UB 预算
  // 判定还要用到原始 N 维与累加器元素宽度。
  // Keep what the rewrite needs on the candidate: the lifted M is the original M
  // times the factor, and the UB budget check also reads the original N dimension
  // and the accumulator element width.
  candidate.maxDotRows = maxDotRows;
  candidate.maxDotCols = maxDotCols;
  candidate.accumulatorElemBytes = accumulatorElemBytes;
  // 单值预算按「提升后的 M × 每个值一行一个向量寄存器」设定：末维不超过一个
  // 向量寄存器（f32 即 64 列）的原样保留，超过的（V 补到 128）落进拆半路径。
  // 阈值由 kVectorRegisterBytes 推出，不再写死列数。
  // The per-value budget is "lifted M × one vector register per row": a value
  // whose last dimension fits one vector register (64 f32 columns) stays whole,
  // and a wider one (V padded to 128) takes the split path.  The threshold is
  // derived from kVectorRegisterBytes instead of a hard-coded column count.
  candidate.splitValueBudgetBytes =
      allowSplit ? candidate.maxDotRows * candidate.rowsPerProgram *
                       kVectorRegisterBytes
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

// 值维超过一个向量寄存器时，静态活值估计会低估提升后的 UB 需求，原因有两项：
//   1. 拆半档有两条半宽链，但 canonicalValues 把它们归到同一个 owner 只计一份，
//      即少计「半宽链数」这一项；
//   2. 后端对循环携带的 UB 张量按两份缓冲，静态估计按一份计，即少计「循环
//      缓冲数」这一项。
// 放大系数由这两项相乘得到，不再写死一个经验常数。一个向量寄存器以内的值链
// 没有这两项放大，保持原判定。
// Once the value dimension exceeds one vector register the static live-byte
// estimate under-counts the lifted UB demand for two reasons: the split tier has
// two half-width chains but canonicalValues assigns them to one owner and counts
// them once, and the backend buffers loop-carried UB tensors twice while the
// estimate counts one copy.  The amplification is the product of these two
// model factors instead of a hard-coded constant.  Chains within one vector
// register keep the plain estimate.
constexpr uint64_t kSplitTierHalfChains = 2;
constexpr uint64_t kLoopCarriedBuffers = 2;

// 提升后的函数体峰值活值是否放不进目标核的 UB 预算。UB 容量未知（零）时
// 不施加约束，沿用既有行为；估计不可得时同样不拒绝。
// True when the lifted body's peak live bytes do not fit the target core's UB
// budget.  An unknown capacity (zero) leaves the check disabled, and an
// unavailable estimate is not a reason to reject.
static bool exceedsUBBudget(triton::FuncOp function,
                            const ResourceSnapshot &resources,
                            int64_t maxDotCols,
                            int64_t accumulatorElemBytes) {
  const std::optional<uint64_t> safe = resources.getSafeUBBudget();
  if (!safe)
    return false;
  LiveByteEstimate estimate = estimatePeakLiveBytes(function.getOperation());
  if (!estimate.known)
    return false;
  uint64_t demand = estimate.peakLiveBytes;
  // 判定条件用列数 × 元素字节，与 kVectorRegisterBytes 同一口径。
  // The predicate compares columns × element bytes against the same vector
  // register width the budget uses.
  if (maxDotCols * accumulatorElemBytes > kVectorRegisterBytes) {
    const uint64_t scale = kSplitTierHalfChains * kLoopCarriedBuffers;
    if (demand > std::numeric_limits<uint64_t>::max() / scale)
      return true;
    demand *= scale;
  }
  LLVM_DEBUG(llvm::dbgs() << "query-row: peak live " << estimate.peakLiveBytes
                          << " bytes, demand " << demand << " bytes, UB budget "
                          << *safe << " bytes\n");
  return demand > *safe;
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
      matchQueryRows(function, l0aBudgetBytes, l0cBudgetBytes,
                     resources.coreNum, forcedRows, allowSplit);
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
  // 创建计划时的物理核数，重新匹配时按同一约束复算因子。
  uint64_t coreNum = 0;

  // 保存候选、创建 epoch 以及已私有验证的沙箱模块。
  QueryRowBatchingPlan(RowLiftCandidate candidate, unsigned epoch,
                       OwningOpRef<ModuleOp> prepared, uint64_t l0aBudgetBytes,
                       uint64_t l0cBudgetBytes, uint64_t coreNum,
                       bool allowSplit)
      : candidate(candidate), epoch(epoch), prepared(std::move(prepared)),
        allowSplit(allowSplit), l0aBudgetBytes(l0aBudgetBytes),
        l0cBudgetBytes(l0cBudgetBytes), coreNum(coreNum) {}

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
                                  l0cBudgetBytes, coreNum, std::nullopt,
                                  allowSplit);
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
          !exceedsUBBudget(cloned, resources, candidate.maxDotCols,
                           candidate.accumulatorElemBytes))
        return std::unique_ptr<QueryRowBatchingPlan>(
            new QueryRowBatchingPlan(candidate, epoch, std::move(sandbox),
                                     l0aBudgetBytes, l0cBudgetBytes,
                                     resources.coreNum, allowSplit));

      if (factor == 2)
        break;
      // The row count already satisfies every resource constraint for `factor`,
      // so the next admissible value is simply the next power of two down.
      factor /= 2;
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
    auto tierCandidate = matchQueryRows(
        function, l0aBudgetBytes, l0cBudgetBytes, resources.coreNum,
        std::nullopt, allowSplit);
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
