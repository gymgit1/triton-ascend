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

#ifndef TRITON_TO_GRAPH_QUERY_ROW_BATCHING_H
#define TRITON_TO_GRAPH_QUERY_ROW_BATCHING_H

#include "TritonToGraph/GraphOptimizationRule.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include <memory>

namespace mlir {
namespace triton {
namespace cfg {

struct ResourceSnapshot;

// A→B 的唯一接口：规则 A（SelectedBlockScanRule）在自己的沙箱副本上把循环改成
// 顺序扫描后，调用本函数让规则 B 立刻完成行提升，而不是「先只读查询、再由调度
// 器单独执行规则 B」。这样 A 拿到的是 B 真实的重写结果：返回 success 就代表
// 该副本可以连同 B 的行提升与启动契约一起提交；返回 failure 就代表 B 在这个
// 形态上跑不通，A 不应提交顺序扫描。
//
// 语义（仿 materializePersistentTaskStripMiningCandidate）：
//   - 就地改写传入的 module/function，不克隆、不提交、不写回原模块；
//   - 调用方必须传私有沙箱模块：匹配失败或重写失败时，沙箱可能已被部分修改；
//   - grid 为静态时在 module 上发布 hacc.coalesce_factor / coalesce_axis /
//     coalesce_grid_ceil_div；grid 为运行时（沙箱没有 hacc.grid_num_tiles）时
//     给函数补隐藏 extent 参数并发布 hacc.program_grid_transforms；
//   - 成功返回时 module 已通过 verify，调用方负责一次性提交（运行时网格须用
//     commitProgramGridFunctionFromSandbox）。
// 入参 resources 提供 L0A/L0C 预算；预算未知（0）时不施加 L0 约束。
//
// 规则 A 的顺序化职责不在本函数内：B 只要求循环已经是单循环、entry block、
// 唯一 axis0 program id 等行提升前提，不再检查循环体内是否还有下标表读。
//
// The single A→B entry point.  Rule A (SelectedBlockScanRule) calls this on its
// own sandboxed copy right after turning the sparse loop into a sequential scan,
// so rule B performs its row lifting immediately instead of being queried and
// later scheduled on its own.  A therefore obtains B's real result: success means
// the copy can commit B's row lifting and launch contract together with A's
// rewrite, failure means B cannot run on this shape and A must not commit.
// The rewrite is in place: the caller owns the sandbox, the module may be
// partially modified on failure and must not be the live IR.  Static grids
// publish the coalesce attributes on the module; runtime grids add the hidden
// extent argument and publish hacc.program_grid_transforms.  The L0A/L0C budgets
// in the resource snapshot bound the lifting factor; an unknown capacity (zero)
// leaves the factor unconstrained.
LogicalResult materializeQueryRowBatchingCandidate(
    ModuleOp module, triton::FuncOp function, const ResourceSnapshot &resources);

// 为查询行循环返回一份在私有沙箱中重建并验证过的计划。仅供 GraphOptimizePass
// 的规则 B 调度段使用；规则 A 走上面的 materialize 接口。两条入口共用同一份
// 重写实现，因此调度路径与 A 调用路径不会产生不同结果。
//
// Return a privately rewritten and verified plan for query-row batching.  Used
// only by the graph-optimize scheduler; rule A uses the materialize entry above.
// Both entries share one rewrite implementation, so the scheduled path and the
// A path cannot diverge.  Unsupported inputs return nullptr without touching the
// original IR.
std::unique_ptr<RewritePlan>
createQueryRowBatchingPlan(triton::FuncOp function, unsigned epoch,
                           const ResourceSnapshot &resources);

} // namespace cfg
} // namespace triton
} // namespace mlir

#endif // TRITON_TO_GRAPH_QUERY_ROW_BATCHING_H
