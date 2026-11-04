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

#ifndef TRITON_TO_GRAPH_ROW_LIFTING_H
#define TRITON_TO_GRAPH_ROW_LIFTING_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>

namespace mlir {
namespace triton {
namespace cfg {

// 判断 `v` 是否等于 `pid` 本身或由它计算得到。函数不进入 region 内部，
// 因此由 `scf.if` 某一分支产出的值会被判为与 pid 无关；当这个结论决定
// 正确性时，应改用 `isPidInvariant`。
// True when `v` is `pid` itself or is computed from it.  Region bodies are not
// inspected, so a value produced by a branch of `scf.if` is reported as
// invariant; use `isPidInvariant` when that answer decides correctness.
bool dependsOnValue(Value v, Value pid);

// 判断 `value` 是否完全不依赖 `pid`，并会穿透嵌套 region：`scf.if` 的
// 结果来自分支 yield 的值，`tt.reduce` 的结果来自其源的组合，因此即使
// 依赖不在顶层操作数上也能被识别为相关。
// True when `value` is computed without any dependence on `pid`, looking
// through nested regions: an `scf.if` whose result comes from pid-dependent
// branch values, and a `tt.reduce` whose sources depend on `pid`, both count as
// dependent even though the dependence is not carried by a top-level operand.
bool isPidInvariant(Value value, Value pid);

// 标量表达式在 `pid` 取 `[0, grid - 1]` 时的有符号取值范围。`grid` 非正
// 表示运行期范围，此时按整个非负 i32 区间证明。
// Signed range of a scalar expression over `pid` taken from `[0, grid - 1]`.
// A non-positive `grid` denotes a runtime extent and is proved over the whole
// non-negative i32 range.
struct IntegerBounds {
  int64_t lo;
  int64_t hi;
};

// 同时证明 `value` 关于 `pid` 单调非降且中间整数运算不会回绕，并返回其
// 范围。语法上递增的表达式一旦中间 add/mul 回绕就不再单调，所以必须先
// 得到可靠范围，才能用最后一行的值作为整组的循环上界。
// Prove together that `value` is monotone non-decreasing in `pid` and that no
// intermediate integer operation wraps. A syntactically increasing expression
// is not monotonic if an intermediate add/multiply wraps, so a proven range is
// required before the last row's value can serve as the group's loop bound.
std::optional<IntegerBounds> inferMonotonicBounds(Value value, Value pid,
                                                  int64_t grid);

// 上界证明的布尔形式，供只需要判断「最后一行给出最大值」的调用方使用。
// 与 pid 无关的值在各提升行上是常量，因此由不变量源的 `tt.reduce` 或
// 条件不变的 `scf.if` 产生的上界可以直接接受，无需数值范围（统一按
// 「单调但无范围」处理，不单独区分 pid 无关）。
// 与 pid 有关的链只要语法上单调非降即可通过：末行上界由运行期重新求值得到，
// 证明器不必给出数值范围；范围放不进表达式位宽时也按「单调」接受，只有需要
// 按 hi 定表宽的下标表提升才要求 `inferMonotonicBounds` 返回具体范围。
// Boolean form of the bound proof for callers that only need "the last row
// supplies the maximum".  A pid-invariant value is constant across the lifted
// rows, so a bound produced by a `tt.reduce` of invariant sources, or by an
// `scf.if` with an invariant condition, is accepted without a numeric range.
// A pid-dependent chain only has to be syntactically monotone non-decreasing:
// the last row's bound is re-evaluated at run time, so no numeric range is
// needed; a range that does not fit the expression's bit width still counts as
// monotone, and only index-table lifting (which sizes its table from `hi`)
// requires the concrete range returned by `inferMonotonicBounds`.
bool isMonotoneInPid(Value value, Value pid, int64_t grid);

// 引擎尚未支持的各类模式的开关。所有开关默认关闭，未置位的能力会保持
// 原有的失败关闭路径。
// Capability switches for patterns the engine does not lift yet. Every switch
// defaults off, so an unset feature keeps the original fail-closed path.
struct RowLiftFeatures {
  // 循环体内出现 tt.reduce。
  // tt.reduce appears in the loop body.
  bool reduce = false;
  // 条件不依赖查询行的 scf.if。
  // scf.if whose condition does not depend on the query row.
  bool loopInvariantIf = false;
  // Q 和 O 由 tt.make_tensor_ptr 构造。
  // Q and O are built with tt.make_tensor_ptr.
  bool tensorPointer = false;
  // O 和 lse 带掩码写回。
  // O and lse are written back under a mask.
  bool maskedStore = false;
  // 循环体内与行相关的 load，在逐行有效掩码下重建，保证扫描已结束的行
  // 不再访问内存。
  // A row-dependent load in the loop body, rebuilt under the per-row active
  // mask so a row whose scan already ended never touches memory.
  bool maskedLoad = false;
  // 循环体内的标量下标表读被提升为每个 program 一次带掩码的 [R,S] 向量
  // 读，循环体只负责从中提取 [R] 列。
  // A scalar index-table read in the loop body is hoisted into one masked
  // [R,S] vector read per program; the body only extracts its [R] column.
  bool tableRead = false;
  // 输出累加器过宽，需要沿宽度拆成两半。
  // The output accumulator is too wide and must be split along its width.
  bool splitWideAccumulator = false;
};

// 单个提升值的字节预算。超过该宽度的值默认被拒绝，由
// splitWideAccumulator 能力负责将其对半拆分。
// Byte budget for one lifted value.  A value wider than this is declined by
// default and halved by the splitWideAccumulator capability.
constexpr int64_t kMaxLiftedValueBytes = 128 * 1024;

// 单个提升值在「一行」上的宽度上限：A2/A5 的向量寄存器是 128 个 f16 车道
// （256 B），换成 f32 即 64 列。超过一个向量寄存器的值链要折半，否则单条链
// 跨多个寄存器，后端无法把它整体常驻 UB。阈值由硬件位宽推出，不写死列数。
// Per-row width ceiling of one lifted value: the A2/A5 vector register holds
// 128 f16 lanes (256 bytes), which is 64 f32 columns.  A chain wider than one
// vector register has to be halved, otherwise a single chain spans several
// registers and the backend cannot keep it resident in UB.  The threshold is
// derived from the hardware width rather than a hard-coded column count.
constexpr int64_t kVectorRegisterBytes = 256;

// UB 多缓冲复制份数：`--enable-auto-multi-buffer` 在 A5（950 系列）默认打开，
// 后端把循环携带的 UB 张量复制成两份缓冲（hivm.multi_buffer >= 2），而
// estimatePeakLiveBytes 只按一份统计，于是静态峰值系统性低估 UB 需求。
// 实测 NSA bwd-dq（B7_T111_H2_HQ32_K110）：静态峰值 107,604 B，bishengir 的
// plan-memory 实际要 247,808 B（2.30×），按一份估算的预算判据因此放行了一个
// 装不进 216 KiB 的 4 行提升形态，真机报 ub overflow。
// 按两份建模：一轮流水的循环携带缓冲至少双缓冲。split-tier 的半宽链项不再单列
// ——宽累加器已由 splitWideAccumulator 与单值预算约束，重复计会和新 R 定界叠加
// 成 4 倍，把本可两行的形态压到不提升。
// UB multi-buffer copies: `--enable-auto-multi-buffer` is on by default on A5
// (950 family), and the backend double-buffers loop-carried UB tensors
// (hivm.multi_buffer >= 2) while estimatePeakLiveBytes counts each value once,
// so the static peak systematically under-estimates the UB demand.  Measured on
// NSA bwd-dq (B7_T111_H2_HQ32_K110): a static peak of 107,604 B became
// 247,808 B in bishengir's plan-memory (2.30x), so the one-copy check let a
// 4-row lifted form through that cannot fit 216 KiB and the device reported
// ub overflow.  Model two copies: a pipelined loop-carried buffer is at least
// double buffered.  The split-tier half-chain term is deliberately not
// re-applied: the wide accumulator is already bounded by splitWideAccumulator
// and the per-value budget, and counting it again would combine with the new R
// bound into 4x and suppress lifts that do fit.
constexpr uint64_t kUBBufferReplication = 2;

// 查询行数的收益策略。行数决策只在这里实现，资源模型只需替换本策略即可。
// Profitability policy for the query-row count.  The row-count decision lives
// here and nowhere else, so a resource model only has to replace this function.
struct RowCountPolicy {
  // 提升后 M（dot 行）维的上限；0 表示不限制。
  // Cap on the lifted M (dot-row) dimension; zero disables the cap.
  int64_t targetDotRows = 0;
  // 单个提升值的字节预算。
  // Byte budget for one lifted value.
  int64_t maxLiftedValueBytes = kMaxLiftedValueBytes;
  // 目标核的真实可用 UB（物理容量减去预留，再乘安全系数）。提升后常驻 UB 的
  // 活值随 R 线性增长，因此它和 L0A/L0C、网格一样是压低 R 的阈值；0 表示容量
  // 未知，该约束关闭。来源见 RFC #2686 第 4 节第三条约束。
  // Real usable UB of the target core (physical capacity minus reserved space, then
  // scaled by the safety percent).  Values resident in UB grow linearly with R,
  // so this bounds R exactly like L0A/L0C and the grid do; zero means the
  // capacity is unknown and the constraint is disabled.  It is the third constraint
  // of RFC #2686 section 4.
  uint64_t ubBudgetBytes = 0;
  // 拆分能力会同时生成两个半宽值而非一个全宽值，因此单值预算翻倍。
  // The split capability materialises two half-width values instead of one
  // full-width value, so the per-value budget doubles.
  bool splitWideAccumulator = false;
  // 原始 dot 的 L0A/L0C tile 最坏字节需求；重写按因子把它们放大，因此实际
  // 需求是 demand * factor。两值为 0 表示候选不含 dot，不受 L0 约束。
  // Worst-case L0A/L0C tile bytes of the original dots.  The rewrite widens
  // them by the factor, so the demand is `demand * factor`.  Zero means the
  // candidate has no dot and is not constrained.
  int64_t maxL0ABytes = 0;
  int64_t maxL0CBytes = 0;
  // L0A/L0C 预算（字节）；0 表示目标架构的立方容量未知，对应检查关闭。
  // L0A/L0C budgets in bytes; zero means the target's cube capacities are
  // unknown and the corresponding check is disabled.
  uint64_t l0ABudgetBytes = 0;
  uint64_t l0CBudgetBytes = 0;
  // 物理核数（纯 SIMT 用 AIV，其余用 AIC）；合并后轴 0 的程序数不得少于
  // 该值，0 表示未知并关闭检查。
  // Number of physical cores (AIV for pure-SIMT kernels, AIC otherwise); the
  // coalesced axis-0 program count must stay at least this large.  Zero means
  // unknown and disables the check.
  uint64_t coreNum = 0;
};

// 为一个 program 选择要处理的查询行数。`grid` 是静态启动网格，为 0 表示
// 运行期范围（最后一组用掩码处理）；`maxDotRows` 是最宽的提升后 M 维，
// `maxTensorBytes` 是候选会生成的最大值；`peakLiveBytes` 是原始函数的峰值
// UB 活值，作为 UB 约束的每因子需求（0 或未知时该约束不参与定界）。
// 返回 1 表示没有可获利的因子，调用方据此决定不做行提升。
// Pick the number of query rows one program handles.  `grid` is the static
// launch grid, or zero for a runtime extent whose partial group is masked;
// `maxDotRows` is the widest lifted M dimension, `maxTensorBytes` the largest
// value the candidate materialises, and `peakLiveBytes` the original peak live
// UB bytes, which is the per-factor demand of the UB constraint (zero or
// unknown disables it).  Returning 1 means no factor is profitable, which
// callers treat as "do not lift".
int64_t selectRowsPerProgram(int64_t grid, int64_t maxDotRows,
                             int64_t maxTensorBytes, int64_t peakLiveBytes,
                             const RowCountPolicy &policy = RowCountPolicy{});

// 一个可按「每个 program 处理多行」重建的函数体。
// One function body that can be rebuilt with several query rows per program.
struct RowLiftCandidate {
  triton::FuncOp function;
  Operation *anchor = nullptr;
  int64_t rowsPerProgram = 1;
  bool runtimeGrid = false;
  // 静态启动网格；运行期范围为 0。静态值让下标表读能在编译期确定列宽。
  // Static launch grid, or zero for a runtime extent.  A static value is what
  // lets the index-table read pick its column extent at compile time.
  int64_t grid = 0;
  RowLiftFeatures features;
  // 最宽的原始 dot M 维，供单值预算与拆分判据按「提升后的 M」换算。
  // Widest original dot M dimension, used to derive the lifted M dimension that
  // the per-value budget and the split predicate are expressed in.
  int64_t maxDotRows = 1;
  // 单个提升进位值的字节预算，超过即由 splitWideAccumulator 能力对半拆分。
  // 规则 B 用「提升后的 M × 每个值一行一个向量寄存器」重新设定它，于是末维
  // 超过一个向量寄存器的累加器会被对半拆开，阈值不再是写死的列数。
  // Byte budget for one lifted carried value; anything above it is halved by the
  // splitWideAccumulator capability.  Rule B re-derives it as "lifted M × one
  // vector register per row", so an accumulator whose last dimension exceeds one
  // vector register is halved and the threshold stops being a hard-coded count.
  int64_t splitValueBudgetBytes = kMaxLiftedValueBytes;
};

// 以下判定函数只检查 IR、不生成 IR，供规则 matcher 与重写路径共用：matcher
// 用它们提前拒绝注定失败的形态，避免付出一次沙箱克隆与 verify；重写路径再
// 判一次保证自己独立成立。两处共用同一份判据，就不会出现「匹配通过、重写
// 失败」的错位。
// These predicates only inspect the IR.  A rule's matcher uses them to decline
// a shape before a sandbox clone and verify, and the rewrite path evaluates the
// same predicate again so both stages cannot drift apart.

// 能否把循环内所有标量表读提升成每个 program 一次带掩码的 [R,S] 读：表读必须
// 是 `load(base + iv)` 且地址单用户，base 是整数元素标量指针并在提升后带行维，
// 上界可证且静态列宽不超过 kMaxTableColumns。没有标量表读时返回 true，因为
// 这种循环不需要列宽。
// True when every scalar table read in `loop` can be hoisted into one masked
// [R,S] read per program.  A loop without scalar table reads needs no column
// extent and passes.
bool canHoistTableReads(scf::ForOp loop, const RowLiftCandidate &candidate);

// 能否把块指针展开成显式指针张量：pointee 是静态形状的 RankedTensor，
// offset/shape/stride 的元素都是整数。`enabled` 对应 tensorPointer 能力开关。
// True when the block pointer can be expanded into an explicit pointer tensor:
// the pointee is a statically shaped ranked tensor and every index operand is
// integral.  `enabled` mirrors the tensorPointer capability switch.
bool canExpandTensorPointer(triton::MakeTensorPtrOp makePtr, bool enabled);

// 能否重建条件与查询行无关的 scf.if：条件为标量 i1 且不依赖 `pid`，每个分支
// 单 block，结果不携带块指针。
// True when the scf.if can be rebuilt with every lifted row on the same branch:
// a scalar i1 condition independent of `pid`, single-block branches, and no
// block pointer among the results.
bool liftsRowInvariantIf(scf::IfOp ifOp, Value pid);

// 能否把超过单值预算的循环进位值沿宽度拆半：预算内的值不拆；超过两倍预算、
// 末维不是偶数、或累加器岛不合法时拒绝。`rowsPerProgram` 是提升后的行数，
// 用来把单行的字节数换算成提升后的宽度；`budgetBytes` 是提升后的单值预算。
// 能力未开启时恒为 true。
// True when every carried value above the per-value budget can be halved along
// its width; values within the budget are left alone.  `rowsPerProgram`
// converts a single row's byte count into the lifted width, and `budgetBytes`
// is the lifted per-value budget.  Without the capability the check is vacuous.
bool canSplitWideAccumulators(scf::ForOp loop, int64_t rowsPerProgram,
                              bool enabled,
                              int64_t budgetBytes = kMaxLiftedValueBytes);

// 只提升与查询行相关的值。与行无关的 K/V load 保持二维；被提升的 lhs
// [BT,G,K] 在每个 dot 处展平成 [BT*G,K]，而不是生成 batched dot。
// Lift only query-dependent values. Invariant K/V loads stay two-dimensional;
// a lifted lhs [BT,G,K] is reshaped to [BT*G,K] at each dot, not batched dots.
class RowLiftBuilder {
public:
  // 用给定的重写器与候选描述构造行提升引擎；能力开关取自候选。
  // Build the row-lifting engine from the rewriter and the candidate; the
  // capability switches are taken from the candidate.
  RowLiftBuilder(IRRewriter &rewriter, RowLiftCandidate candidate);

  // 在 `fn` 内执行行提升：克隆函数体并逐条重建，成功返回 true。
  // Rewrite `fn` in place by rebuilding its body operation by operation.
  bool run(triton::FuncOp fn);

private:
  // 被展开成显式地址的块指针。形状与偏移都会保留，以便在使用点把 load 的
  // boundary check 转成掩码。
  // A block pointer expanded into explicit addresses. Both the shapes and the
  // offsets are kept so a load's boundary check can be turned into a mask at
  // its own use site.
  struct LiftedDescriptor {
    Value pointer;              // [R]+tile 指针张量
    SmallVector<Value> shapes;  // 每个 tile 维的 i64 形状
    SmallVector<Value> offsets; // 每个 tile 维的 i64 偏移
    SmallVector<int64_t> tileShape;
  };

  // 按操作种类分派单条指令的重建；无法支持时返回 false，整次提升失败。
  // Rebuild one operation according to its kind; false declines the rewrite.
  bool rewrite(Operation *op);
  // 在 axis+1 上重建 tt.reduce（前插了行维），并原样复用 combine region。
  // Rebuild tt.reduce on axis+1 after the row dimension was inserted.
  bool rewriteReduce(triton::ReduceOp reduce);
  // 把 tt.make_tensor_ptr 的块指针展开成显式 [R]+tile 指针张量；若该描述符
  // 已被规划成拆半，则走拆半路径。
  // Expand a block-pointer descriptor into an explicit [R]+tile pointer tensor,
  // or take the split path when the descriptor was planned to be halved.
  bool rewriteTensorPointer(triton::MakeTensorPtrOp makePtr);
  // 展开一个描述符。`lastExtent` 是最后一维的 tile 宽度（拆半时为半宽），
  // `lastOffsetOverride` 非空时替换最后一维的偏移（第二半整体后移半宽）。
  // Expand one descriptor.  `lastExtent` is the trailing tile extent (half the
  // width when splitting) and a non-null `lastOffsetOverride` replaces the
  // trailing offset, which is how the second half is shifted by one half width.
  bool expandTensorPointer(triton::MakeTensorPtrOp makePtr, int64_t lastExtent,
                           Value lastOffsetOverride,
                           LiftedDescriptor &descriptor);
  // 用展开后的指针张量重建描述符 load，并把 boundaryCheck 转成掩码。
  // Rebuild a descriptor load on the expanded pointer tensor.
  bool rewriteDescriptorLoad(triton::LoadOp load,
                             const LiftedDescriptor &descriptor);
  // 用给定描述符发出一次 load 并交回结果，不做映射登记；拆半路径按两个描述符
  // 各调用一次，因此结果类型取自描述符的指针形状。
  // Emit one load for the given descriptor and hand back its result without
  // registering a mapping.  The split path calls it once per half, so the result
  // shape comes from the descriptor's pointer shape.
  bool emitDescriptorLoad(triton::LoadOp load, const LiftedDescriptor &descriptor,
                          Value &result);
  // 用展开后的指针张量重建描述符 store，并把 boundaryCheck 转成掩码。
  // Rebuild a descriptor store on the expanded pointer tensor.
  bool rewriteDescriptorStore(triton::StoreOp store,
                              const LiftedDescriptor &descriptor);
  // 把值广播到目标形状：标量做 splat，行提升值补齐缺失的尾部单位维。
  // Broadcast `v` to `target`: a scalar becomes a splat and a row-lifted value
  // gains the missing trailing unit dimensions first.
  Value broadcastFullValue(Value v, RankedTensorType target);
  // 生成沿某个 tile 轴递增的 i64 索引网格，并广播到目标形状。
  // Build an i64 index grid that increases along one tile axis.
  Value expandTileRange(Location loc, int64_t axis, int64_t extent,
                        RankedTensorType target);
  // 把 boundaryCheck 指定的各轴越界判断合并成一个显式 i1 掩码。
  // Turn the boundary checks of the given axes into one explicit i1 mask.
  Value buildDescriptorMask(const LiftedDescriptor &descriptor,
                            ArrayRef<int32_t> boundaryCheck,
                            RankedTensorType target);
  // 重建条件与查询行无关的 scf.if，两个分支都按行提升后的类型重建。
  // Rebuild an scf.if whose condition does not depend on the query row.
  bool rewriteIfOp(scf::IfOp old);
  // 重建一个分支 region，返回新的 yield 值。`split` 逐结果说明该值是否以两个
  // 半宽值表示，两个分支必须给出同一判断。
  // Rebuild one branch region of an scf.if and its yield values.  `split` says
  // per result whether it is represented by two half-width values; both branches
  // must agree on it.
  bool rewriteIfBody(Region &from, Region &to, ArrayRef<bool> split);
  // 合并一次提升访存涉及的所有掩码：描述符边界掩码、原始 mask 操作数以及
  // 运行期有效行掩码。没有任何掩码时返回空 Value；某一部分无法对齐到指针
  // 形状时 `ok` 置 false。
  // Merge every mask that applies to one lifted memory access: the descriptor
  // boundary mask, the original mask operand, and the runtime valid-row mask.
  // Returns a null Value when no mask applies; `ok` is false when a part
  // cannot be aligned to the pointer shape.
  Value combineMemoryMask(Value pointer, Value originalMask, Value extraMask,
                          bool &ok);
  // 把若干已成形（或可尾部广播）的掩码分量合并到指针形状上；`base` 是已经
  // 合并好的分量（可为空）。不做行维补齐，拆半路径传入的就是半宽掩码。
  // Merge mask parts that are already shaped (or only need trailing broadcast)
  // onto the pointer shape; `base` is an already merged part (possibly null).
  // No row dimension is inserted, because the split path hands in half-width
  // masks.
  Value mergeMasks(Value pointer, Value base, ArrayRef<Value> parts, bool &ok);
  // 循环外提升的标量下标表读：每个 program 一次带掩码的 [R,S] 向量读，
  // 每次迭代只提取对应的 [R] 列。
  // A scalar index table read hoisted out of the loop: one masked [R,S]
  // vector read per program, with its [R] column extracted per iteration.
  struct TableRead {
    Value table;    // [R,S] 值，被掩码排除的位置为 0
    Value zero;     // [R,S] 全零，用于清掉未选中的列
    Value columns;  // [S] 表列下标
  };

  // 为 `old` 规划循环外下标表读；返回 false 表示拒绝整个循环的提升。
  // Plan the hoisted reads for `old`; false declines the whole loop.
  bool planTableRead(scf::ForOp old);
  // 用 one-hot 选择与行求和，从 [R,S] 表中取出第 `iv` 列。
  // Extract column `iv` of the [R,S] table with a one-hot select and row sum.
  Value extractTableColumn(const TableRead &read, Value iv);
  // 用整数 add 组合子在 `axis` 上求和。
  // Sum `value` along `axis` with an integer add combiner.
  Value sumAlongAxis(Location loc, Value value, int64_t axis);
  // `iv < ub` 广播到所有提升行；循环体内每个与行相关的访存都在它下面重建。
  // `iv < ub` splatted over the lifted rows; every row-dependent memory
  // access in the loop body is rebuilt under it.
  Value createLoopActiveMask(scf::ForOp old, scf::ForOp loop);
  // 用两个连续半宽值表示一个提升值。splitWideAccumulator 能力会对 dot、
  // 累加器上的逐元素链和 store 各重建两次，保证没有单个操作生成超过预算
  // 的全宽值。
  // A lifted value represented by two contiguous half-width pieces. The
  // splitWideAccumulator capability rebuilds the dot, the elementwise chain on
  // the accumulator and the store once per half, so no single operation
  // materialises a value wider than the per-value byte budget.
  struct SplitValue {
    Value first;
    Value second;
  };

  // 找出超出字节预算且可安全拆半的累加器，返回其循环进位索引，并登记要按
  // 最后一维拆半的描述符。必须在重建函数体之前调用：描述符可能位于循环之前，
  // 一旦先按整宽展开就无法再拆。
  // Find the accumulators that exceed the byte budget and can be split safely,
  // and record the descriptors that must be halved.  This has to run before the
  // body is rebuilt: a descriptor above the loop is already expanded by the time
  // the loop is rebuilt, so a whole-width expansion could no longer be halved.
  bool planWideAccumulatorSplits(scf::ForOp old,
                                 SmallVectorImpl<int64_t> &indices);

  // 一个按最后一维拆成两半的描述符。展开成指针张量时两半各自持有一份指针，
  // 与行无关而走克隆路径时则持有一对块指针。
  // One descriptor halved along its last dimension.  The expanded form keeps one
  // pointer tensor per half; a row-invariant descriptor keeps a pair of block
  // pointers instead.
  struct SplitDescriptor {
    LiftedDescriptor first;
    LiftedDescriptor second;
    Value firstPointer;
    Value secondPointer;
    bool expanded = false;
  };

  // 沿最后一维把值拆成两个连续半宽值（仅用于进位初值等无法结构拆分的形态）。
  // Splits `value` along its last dimension into two contiguous halves; only used
  // for the loop-carried initial value, which has no structural source.
  bool splitLastDimension(Value value, SplitValue &halves);
  // 取一个已规划拆半值的两个半宽提升值：先复用重写期已登记的两半，否则按规划
  // 的半宽从原始值直接重建（splat 常量与 splat 折叠成半宽常量，其余形态先拆
  // 再各自补行维），避免先提升成全宽再拆。两者都不可用时返回 false，调用方据
  // 此放弃整次提升。
  // Return the two half-width lifted values of a value planned for halving: reuse
  // the halves registered during rewriting, otherwise rebuild them from the
  // original value using the planned half width (a splat constant or a scalar
  // splat collapses into a half-width constant; anything else is halved first and
  // lifted per half), so a full-width lifted temporary is never materialised.
  // Returning false makes the caller decline the whole lift.
  bool ensureRowHalves(Value old, SplitValue &halves);
  // 把一个整宽操作数按半宽重建：只有「各列取值相同」的形态（标量 splat、
  // 沿最后一维从 1 广播、splat 常量）可以这样折叠，重建后同一个半宽值同时
  // 供两半使用。返回空 Value 表示无法结构折叠。
  // Rebuild a full-width operand at half width.  Only shapes whose value is
  // identical across the columns (a scalar splat, a broadcast from a trailing
  // unit dimension, a splat constant) collapse this way, and the single result is
  // shared by both halves.  A null Value means the operand cannot be collapsed.
  Value rebuildHalfOperand(Value full, RankedTensorType halfType);
  // 与行无关的块指针按最后一维拆成两个半宽块指针；与行相关的则两半各自展开
  // 成半宽指针张量。
  // Split a descriptor into two half-width pieces: a row-invariant block pointer
  // becomes a pair of half-width block pointers, and a row-dependent one is
  // expanded once per half.
  bool rewriteSplitTensorPointer(triton::MakeTensorPtrOp makePtr,
                                 SplitDescriptor &descriptor);
  // 用拆半描述符发出两次半宽 load。
  // Emit one half-width load per half of a split descriptor.
  bool rewriteSplitDescriptorLoad(triton::LoadOp load, SplitDescriptor &descriptor);
  // 按操作种类分派拆分累加器的重建（dot / store / 逐元素算术）。
  // Dispatch the split-aware rebuild by operation kind.
  bool rewriteSplitOp(Operation *op);
  // 用两个半宽 B 与对应的累加器半宽各重建一次 dot。
  // Rebuild the dot once per half-width accumulator and matching B half.
  bool rewriteSplitDot(triton::DotOp dot);
  // 把纯逐元素算术按半宽重建两次。
  // Rebuild pure elementwise arithmetic once per half width.
  bool rewriteSplitElementwise(Operation *op);
  // 把指针、值和掩码都拆半后发出两次半宽 store。
  // Split pointer, value and mask, then emit two half-width stores.
  bool rewriteSplitStore(triton::StoreOp store);
  // 把 scf.for 重建为每个 program 处理多行的循环。
  // Rebuild an scf.for into a loop that handles several query rows.
  bool rewriteLoop(scf::ForOp old);
  // 在类型最前面插入行维 [R]。
  // Prepend the leading row dimension [R] to a type.
  RankedTensorType getLiftedType(Type old);
  // 取原始值对应的已重建值。
  // Look up the rebuilt value for an original SSA value.
  Value getMappedValue(Value old);
  // 判断值是否已被提升；提升后其类型与原始类型不同，因此无需额外布尔状态。
  // Every lifted value gains a leading row dimension, so its type already
  // records this state; no separate boolean needs to be maintained.
  bool hasRowDimension(Value old);
  // 返回带行维的映射值；标量做 splat，张量先扩维再广播。
  // Return the mapped value with a leading row dimension, splatting a scalar
  // or expanding and broadcasting a tensor as needed.
  Value ensureRowDimension(Value old);
  // 生成一次保持元素顺序的 reshape（不重排元素，避免混入其他 program 的值）。
  // Emit a reshape that preserves element order.
  Value reshape(Value v, ArrayRef<int64_t> shape);
  // 把 [R,M,K] 展平成 [R*M,K]，以便复用普通 tt.dot。
  // Flatten [R,M,K] to [R*M,K] so a plain tt.dot can be reused.
  Value flattenDotRows(Value v);
  // 补齐尾部单位维后广播到目标形状。
  // Broadcast a value to `target`, adding trailing unit dimensions first.
  Value broadcastRowValue(Value v, RankedTensorType target);

  IRRewriter &rewriter;
  RowLiftCandidate candidate;
  RowLiftFeatures features;

  // 原始 SSA 值到重建后计算的映射。
  // Original SSA values mapped to the rebuilt computation.
  IRMapping valueMapping;
  // 已展开为指针张量的块指针，按原始描述符值索引。
  // Block pointers that were expanded into pointer tensors, keyed by their
  // original descriptor value.
  DenseMap<Value, LiftedDescriptor> liftedDescriptors;
  // 由两个半宽值代替单个提升值的那些值。消费方必须先走拆分感知路径，
  // 否则依赖探测会查不到全宽映射。
  // Values carried by two half-width values instead of one lifted value.
  // Consumers are rebuilt through the split-aware path before the generic
  // dependency probing can look up a missing full-width mapping.
  DenseMap<Value, SplitValue> splitValues;
  // 在最后一行求值、用于形成共享循环上界的标量值。
  // Scalar values evaluated at the last row to form the shared loop bound.
  IRMapping lastRowMapping;
  Value validRows;
  // 循环外读取一次的标量下标表，按原始循环体读取的列地址索引。
  // Scalar index tables read once before the loop, keyed by the column address
  // the original body loaded from.
  DenseMap<Value, TableRead> columnTables;
  // 重建后循环的归纳变量（供下标表列提取使用）与由它导出的逐行有效掩码；
  // 掩码为空表示当前循环尚未生成。
  // Induction variable of the rebuilt loop (used by index-table column
  // extraction) and the per-row validity mask derived from it; an empty mask
  // means the current loop has not materialised one yet.
  Value loopInductionVar;
  Value rowActiveMask;
  // 循环进位值中的哪些要被拆半，在重建函数体之前就已确定：描述符可能位于
  // 循环之外，必须先拆再展开。
  // Which carried values are halved, decided before the body is rebuilt because
  // a descriptor above the loop has to be split before it is expanded.
  SmallVector<int64_t> plannedSplitIndices;
  // 规划阶段认定将由两个半宽值表示的那些原始值（含累加器、dot 结果与穿过
  // scf.if 的结果）。重写 scf.if 时必须在建出结果类型之前就知道哪些结果要
  // 占两项半宽类型，因此这份判断不能只存在于重写期的 splitValues。
  // Original values that planning decided to represent as two half-width values,
  // including the accumulator, dot results and the values that flow through an
  // scf.if.  Rebuilding an scf.if has to know which results take two half-width
  // types before it is created, so this decision cannot live only in the
  // rewrite-time `splitValues`.
  DenseSet<Value> plannedSplitValues;
  // 规划阶段为每个要拆半的原始值登记的半宽（最后一维的宽度）。在生成任何 IR
  // 之前填充，因此半宽值的类型无需等到重写期 splitValues 才知道。
  // Half width (trailing extent) recorded during planning for every value that
  // has to be halved.  It is filled before any IR is generated, so a half-width
  // type no longer depends on the rewrite-time `splitValues`.
  DenseMap<Value, int64_t> plannedHalfWidths;
  // 需要按最后一维拆半的描述符，键为 tt.make_tensor_ptr 的结果。
  // Descriptors to halve along their last dimension, keyed by the
  // tt.make_tensor_ptr result.
  DenseMap<Value, SplitDescriptor> splitDescriptors;
};

} // namespace cfg
} // namespace triton
} // namespace mlir

#endif // TRITON_TO_GRAPH_ROW_LIFTING_H
