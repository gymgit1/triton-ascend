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

// 规则 B 只做行提升：保持循环结构，把与查询行相关的值前插行维 [R]（每个
// program 处理 R 个查询行），逐行有效掩码与掩码写回由 RowLiftBuilder 完成。
// 它不要求循环已经是顺序扫描：循环体内与行相关的访存由重写路径的逐行掩码
// (maskedLoad/maskedStore) 兜住，因此判定口径只有一份。
//
// Rule B owns row lifting only: it keeps the loop structure and prepends the
// query-row dimension [R] to row-dependent values (R query rows per program),
// relying on RowLiftBuilder for per-row masks and masked write-back.  It does
// not require the loop to be a sequential scan: row-dependent accesses inside
// the loop are covered by the per-row masks the rewrite emits.

// 为查询行循环返回一份在私有沙箱中重建并验证过的计划。不支持输入时返回
// nullptr，且不修改原始 IR。
//
// Return a privately rewritten and verified plan for query-row batching.
// Unsupported inputs return nullptr without touching the original IR.
std::unique_ptr<RewritePlan>
createQueryRowBatchingPlan(triton::FuncOp function, unsigned epoch,
                           const ResourceSnapshot &resources);

} // namespace cfg
} // namespace triton
} // namespace mlir

#endif // TRITON_TO_GRAPH_QUERY_ROW_BATCHING_H
