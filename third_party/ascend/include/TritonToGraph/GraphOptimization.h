/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

#ifndef TRITON_TO_GRAPH_GRAPH_OPTIMIZATION_H
#define TRITON_TO_GRAPH_GRAPH_OPTIMIZATION_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace mlir {
namespace triton {
namespace cfg {

// The identity of every rule is one bit of the pass rule mask, so the mask
using GraphOptimizationRuleMask = uint32_t;

enum class GraphOptimizationRuleId : GraphOptimizationRuleMask {
  LoadStoreTranspose = 1,
  TransposePointwiseReorder = 2,
  StoreCoalescing = 4,
  // RowCoalescing includes a shared-rhs dot pattern and a pure-SIMT-only
  // elementwise pattern. It is intentionally
  // scheduled once after the normal per-function phases because its launch
  // contract must not be suppressed by their rewrite budget.
  RowCoalescing = 8,
  // DiagonalMaskRemoval rewrites compute logic rather than memory access, so
  // it is architecture and template independent.
  DiagonalMaskRemoval = 128,
  // ConvertModuloToMask linearizes wrapped tile addresses.  It runs before the
  // memory-access rules so they see the linear form.
  ConvertModuloToMask = 256,
  // The following identities are owned by the layout/memory compatibility
  // passes.  They deliberately are not GraphOptimizationRule candidates and
  // are not added to GraphOptimizePass's per-function phase loop.
  StridedAxisCoalescing = 16,
  ChunkCoalescing = 32,
  StridedLoadStoreRewrite = 64,
  IndependentAxisTensorize = 512,
  PersistentTaskStripMining = 2048,
  ResidentLoadForwarding = 4096,
  IntermediatePrecisionBoundaryElision = 8192,
  StoreCoveragePlanning = 16384,
  ContiguousBlockAccessFormation = 32768,
  // 规则 A（selected-block scan）：把按下标表选择的稀疏循环改写成顺序扫描，
  // 且必须在规则 B 行提升之前完成。默认关闭，只有显式置位该规则位时才注册。
  // Rule A (selected-block scan) turns the table-indexed sparse loop into a
  // sequential scan and must run before rule B lifts it.  It is default-off and
  // is registered only when this bit is explicitly opted in.
  SelectedBlockScan = 65536,
  // 规则 B（query row batching）：把按下标表选择的稀疏块循环按查询行批量提升，
  // 每组 R 行共享 K/V tile。它与 RowCoalescing 共用 RowLiftBuilder，但 matcher
  // 独立，因此两条规则可分别开关；按 RFC #2686 与 IAT 同处 ProgramMapping 阶段。
  // Rule B (query row batching) batches R query rows of a table-indexed sparse
  // block loop so one K/V tile feeds them.  It shares RowLiftBuilder with
  // RowCoalescing but owns its own matcher, so the two rules stay independently
  // gated; RFC #2686 places it in the ProgramMapping phase, alongside IAT.
  QueryRowBatching = 131072,
};

enum class GraphOptimizationRulePhase : uint8_t {
  DiagonalMaskRemoval,
  ConvertModuloToMask,
  // 规则 A 紧跟地址线性化之后执行，让后面带预算的访存类规则看到顺序扫描形态。
  // Rule A runs right after address linearization so the memory-access rules
  // below already see the sequential-scan form.
  SelectedBlockScan,
  ProgramMapping,
  PersistentTaskMapping,
  LoadStoreTranspose,
  TransposePointwiseReorder,
  ResidentLoadForwarding,
  IntermediatePrecisionBoundaryElision,
  StoreCoveragePlanning,
  StoreCoalescing,
  ContiguousBlockAccessFormation,
  RowCoalescing,
  Compatibility,
};

constexpr const char *
getGraphOptimizationRuleName(GraphOptimizationRuleId rule) {
  switch (rule) {
  case GraphOptimizationRuleId::LoadStoreTranspose:
    return "LoadStoreTranspose";
  case GraphOptimizationRuleId::TransposePointwiseReorder:
    return "TransposePointwiseReorder";
  case GraphOptimizationRuleId::StoreCoalescing:
    return "StoreCoalescing";
  case GraphOptimizationRuleId::RowCoalescing:
    return "RowCoalescing";
  case GraphOptimizationRuleId::DiagonalMaskRemoval:
    return "DiagonalMaskRemoval";
  case GraphOptimizationRuleId::ConvertModuloToMask:
    return "ConvertModuloToMask";
  case GraphOptimizationRuleId::StridedAxisCoalescing:
    return "StridedAxisCoalescing";
  case GraphOptimizationRuleId::ChunkCoalescing:
    return "ChunkCoalescing";
  case GraphOptimizationRuleId::StridedLoadStoreRewrite:
    return "StridedLoadStoreRewrite";
  case GraphOptimizationRuleId::IndependentAxisTensorize:
    return "IndependentAxisTensorizeRule";
  case GraphOptimizationRuleId::PersistentTaskStripMining:
    return "PersistentTaskStripMiningRule";
  case GraphOptimizationRuleId::ResidentLoadForwarding:
    return "ResidentLoadForwardingRule";
  case GraphOptimizationRuleId::IntermediatePrecisionBoundaryElision:
    return "IntermediatePrecisionBoundaryElisionRule";
  case GraphOptimizationRuleId::StoreCoveragePlanning:
    return "StoreCoveragePlanningRule";
  case GraphOptimizationRuleId::ContiguousBlockAccessFormation:
    return "ContiguousBlockAccessFormationRule";
  case GraphOptimizationRuleId::SelectedBlockScan:
    return "SelectedBlockScanRule";
  case GraphOptimizationRuleId::QueryRowBatching:
    return "QueryRowBatchingRule";
  }
  return "Unknown";
}

constexpr GraphOptimizationRuleMask
getGraphOptimizationRuleMask(GraphOptimizationRuleId rule) {
  return static_cast<GraphOptimizationRuleMask>(rule);
}

constexpr GraphOptimizationRulePhase
getGraphOptimizationRulePhase(GraphOptimizationRuleId rule) {
  switch (rule) {
  case GraphOptimizationRuleId::DiagonalMaskRemoval:
    return GraphOptimizationRulePhase::DiagonalMaskRemoval;
  case GraphOptimizationRuleId::ConvertModuloToMask:
    return GraphOptimizationRulePhase::ConvertModuloToMask;
  case GraphOptimizationRuleId::IndependentAxisTensorize:
    return GraphOptimizationRulePhase::ProgramMapping;
  case GraphOptimizationRuleId::PersistentTaskStripMining:
    return GraphOptimizationRulePhase::PersistentTaskMapping;
  case GraphOptimizationRuleId::LoadStoreTranspose:
    return GraphOptimizationRulePhase::LoadStoreTranspose;
  case GraphOptimizationRuleId::TransposePointwiseReorder:
    return GraphOptimizationRulePhase::TransposePointwiseReorder;
  case GraphOptimizationRuleId::ResidentLoadForwarding:
    return GraphOptimizationRulePhase::ResidentLoadForwarding;
  case GraphOptimizationRuleId::IntermediatePrecisionBoundaryElision:
    return GraphOptimizationRulePhase::IntermediatePrecisionBoundaryElision;
  case GraphOptimizationRuleId::StoreCoveragePlanning:
    return GraphOptimizationRulePhase::StoreCoveragePlanning;
  case GraphOptimizationRuleId::StoreCoalescing:
    return GraphOptimizationRulePhase::StoreCoalescing;
  case GraphOptimizationRuleId::ContiguousBlockAccessFormation:
    return GraphOptimizationRulePhase::ContiguousBlockAccessFormation;
  // 规则 A 必须在规则 B 检查循环之前完成改写，因此该阶段排在所有带预算的
  // 访存阶段之前。
  // Rule A has to finish rewriting the loop before rule B inspects it; the
  // phase sits before every budgeted memory-access phase for that reason.
  case GraphOptimizationRuleId::SelectedBlockScan:
    return GraphOptimizationRulePhase::SelectedBlockScan;
  // 按 RFC #2686，规则 B（query row batching）与 IAT 同处 ProgramMapping 阶段。
  // Rule B (query row batching) runs in the ProgramMapping phase, alongside IAT,
  // as specified by RFC #2686.
  case GraphOptimizationRuleId::QueryRowBatching:
    return GraphOptimizationRulePhase::ProgramMapping;
  case GraphOptimizationRuleId::RowCoalescing:
    return GraphOptimizationRulePhase::RowCoalescing;
  case GraphOptimizationRuleId::StridedAxisCoalescing:
  case GraphOptimizationRuleId::ChunkCoalescing:
  case GraphOptimizationRuleId::StridedLoadStoreRewrite:
    return GraphOptimizationRulePhase::Compatibility;
  }
  return GraphOptimizationRulePhase::Compatibility;
}

constexpr std::array<GraphOptimizationRuleId, 17>
    kGraphOptimizationRuleRegistry = {
        GraphOptimizationRuleId::LoadStoreTranspose,
        GraphOptimizationRuleId::TransposePointwiseReorder,
        GraphOptimizationRuleId::StoreCoalescing,
        GraphOptimizationRuleId::RowCoalescing,
        GraphOptimizationRuleId::DiagonalMaskRemoval,
        GraphOptimizationRuleId::ConvertModuloToMask,
        GraphOptimizationRuleId::StridedAxisCoalescing,
        GraphOptimizationRuleId::ChunkCoalescing,
        GraphOptimizationRuleId::StridedLoadStoreRewrite,
        GraphOptimizationRuleId::IndependentAxisTensorize,
        GraphOptimizationRuleId::PersistentTaskStripMining,
        GraphOptimizationRuleId::ResidentLoadForwarding,
        GraphOptimizationRuleId::IntermediatePrecisionBoundaryElision,
        GraphOptimizationRuleId::StoreCoveragePlanning,
        GraphOptimizationRuleId::ContiguousBlockAccessFormation,
        GraphOptimizationRuleId::SelectedBlockScan,
        GraphOptimizationRuleId::QueryRowBatching,
};

constexpr bool hasUniqueSingleBitGraphOptimizationRuleIds() {
  for (std::size_t index = 0; index < kGraphOptimizationRuleRegistry.size();
       ++index) {
    const GraphOptimizationRuleMask mask =
        getGraphOptimizationRuleMask(kGraphOptimizationRuleRegistry[index]);
    if (mask == 0 || (mask & (mask - 1)) != 0)
      return false;
    for (std::size_t other = index + 1;
         other < kGraphOptimizationRuleRegistry.size(); ++other) {
      if (mask ==
          getGraphOptimizationRuleMask(kGraphOptimizationRuleRegistry[other]))
        return false;
    }
  }
  return true;
}

static_assert(hasUniqueSingleBitGraphOptimizationRuleIds(),
              "GraphOptimizationRuleId values must be unique single bits");

constexpr GraphOptimizationRuleMask kLegacyGraphOptimizationRuleMask =
    getGraphOptimizationRuleMask(GraphOptimizationRuleId::LoadStoreTranspose) |
    getGraphOptimizationRuleMask(
        GraphOptimizationRuleId::TransposePointwiseReorder) |
    getGraphOptimizationRuleMask(GraphOptimizationRuleId::StoreCoalescing) |
    getGraphOptimizationRuleMask(GraphOptimizationRuleId::RowCoalescing) |
    getGraphOptimizationRuleMask(GraphOptimizationRuleId::DiagonalMaskRemoval) |
    getGraphOptimizationRuleMask(GraphOptimizationRuleId::ConvertModuloToMask) |
    getGraphOptimizationRuleMask(
        GraphOptimizationRuleId::StridedAxisCoalescing) |
    getGraphOptimizationRuleMask(GraphOptimizationRuleId::ChunkCoalescing) |
    getGraphOptimizationRuleMask(
        GraphOptimizationRuleId::StridedLoadStoreRewrite);

constexpr GraphOptimizationRuleMask kKnownGraphOptimizationRuleMask =
    kLegacyGraphOptimizationRuleMask |
    getGraphOptimizationRuleMask(
        GraphOptimizationRuleId::IndependentAxisTensorize) |
    getGraphOptimizationRuleMask(
        GraphOptimizationRuleId::PersistentTaskStripMining) |
    getGraphOptimizationRuleMask(
        GraphOptimizationRuleId::ResidentLoadForwarding) |
    getGraphOptimizationRuleMask(
        GraphOptimizationRuleId::IntermediatePrecisionBoundaryElision) |
    getGraphOptimizationRuleMask(
        GraphOptimizationRuleId::StoreCoveragePlanning) |
    getGraphOptimizationRuleMask(
        GraphOptimizationRuleId::ContiguousBlockAccessFormation) |
    getGraphOptimizationRuleMask(
        GraphOptimizationRuleId::SelectedBlockScan) |
    getGraphOptimizationRuleMask(
        GraphOptimizationRuleId::QueryRowBatching);

constexpr GraphOptimizationRuleMask kDefaultEligibleGraphOptimizationRuleMask =
    kLegacyGraphOptimizationRuleMask |
    getGraphOptimizationRuleMask(
        GraphOptimizationRuleId::IndependentAxisTensorize) |
    getGraphOptimizationRuleMask(
        GraphOptimizationRuleId::PersistentTaskStripMining);

constexpr GraphOptimizationRuleMask kFixedDefaultOffGraphOptimizationRuleMask =
    getGraphOptimizationRuleMask(
        GraphOptimizationRuleId::ResidentLoadForwarding) |
    getGraphOptimizationRuleMask(
        GraphOptimizationRuleId::IntermediatePrecisionBoundaryElision) |
    getGraphOptimizationRuleMask(
        GraphOptimizationRuleId::StoreCoveragePlanning) |
    getGraphOptimizationRuleMask(
        GraphOptimizationRuleId::ContiguousBlockAccessFormation) |
    getGraphOptimizationRuleMask(
        GraphOptimizationRuleId::SelectedBlockScan) |
    getGraphOptimizationRuleMask(
        GraphOptimizationRuleId::QueryRowBatching);

constexpr GraphOptimizationRuleMask kDefaultGraphOptimizationRuleMask =
    kDefaultEligibleGraphOptimizationRuleMask;

constexpr GraphOptimizationRuleMask kAllGraphOptimizationRuleMask =
    kKnownGraphOptimizationRuleMask;

static_assert(kLegacyGraphOptimizationRuleMask == 511,
              "legacy graph optimization rule mask is an ABI contract");
static_assert(kDefaultGraphOptimizationRuleMask ==
                  kDefaultEligibleGraphOptimizationRuleMask,
              "IAT and PTSM must remain enabled by default");
static_assert(kKnownGraphOptimizationRuleMask ==
                  (kDefaultEligibleGraphOptimizationRuleMask |
                   kFixedDefaultOffGraphOptimizationRuleMask),
              "known graph rule identities must retain their scope partition");
static_assert((kDefaultEligibleGraphOptimizationRuleMask &
               kFixedDefaultOffGraphOptimizationRuleMask) == 0,
              "fixed-default-off rules must not become default candidates");
static_assert((kDefaultGraphOptimizationRuleMask &
               ~kDefaultEligibleGraphOptimizationRuleMask) == 0,
              "default graph rules must stay within the approved candidates");
static_assert(static_cast<unsigned>(getGraphOptimizationRulePhase(
                  GraphOptimizationRuleId::ResidentLoadForwarding)) <
                  static_cast<unsigned>(getGraphOptimizationRulePhase(
                      GraphOptimizationRuleId::StoreCoveragePlanning)),
              "resident forwarding must run before store planning");
static_assert(static_cast<unsigned>(getGraphOptimizationRulePhase(
                  GraphOptimizationRuleId::StoreCoveragePlanning)) <
                  static_cast<unsigned>(getGraphOptimizationRulePhase(
                      GraphOptimizationRuleId::ContiguousBlockAccessFormation)),
              "store planning must run before access formation");
static_assert(static_cast<unsigned>(getGraphOptimizationRulePhase(
                  GraphOptimizationRuleId::ConvertModuloToMask)) <
                  static_cast<unsigned>(getGraphOptimizationRulePhase(
                      GraphOptimizationRuleId::SelectedBlockScan)),
              "modulo-to-mask must run before the selected-block scan");
static_assert(static_cast<unsigned>(getGraphOptimizationRulePhase(
                  GraphOptimizationRuleId::SelectedBlockScan)) <
                  static_cast<unsigned>(getGraphOptimizationRulePhase(
                      GraphOptimizationRuleId::IndependentAxisTensorize)),
              "selected-block scan must run before program mapping");

constexpr bool
isValidGraphOptimizationRuleMask(GraphOptimizationRuleMask ruleMask) {
  constexpr GraphOptimizationRuleMask unknownRuleBits =
      ~kKnownGraphOptimizationRuleMask;
  return (ruleMask & unknownRuleBits) == 0;
}

struct IndependentAxisTensorizeRuleOptions {
  bool enabledForCompileMode = true;
  bool iatAndPtsmEnabled = false;
};
struct PersistentTaskStripMiningRuleOptions {
  bool enabledForCompileMode = true;
};
struct StoreCoalescingRuleOptions {
  bool enabledForCompileMode = true;
};
struct ResidentLoadForwardingRuleOptions {};
struct IntermediatePrecisionBoundaryElisionRuleOptions {};
struct StoreCoveragePlanningRuleOptions {};
struct ContiguousBlockAccessFormationRuleOptions {};
struct SelectedBlockScanRuleOptions {
  bool forceApply = false;
  // Rule A 只有与规则 B 同批开启时才允许连带 materialize B；规则位的判断在
  // 注册阶段完成，避免规则内部再去解码 mask。
  // Rule A may materialize Rule B only when Rule B's bit is opted in as well.
  // The mask decision is made at registration time, so the rule itself never
  // decodes the mask.
  bool queryRowBatchingEnabled = false;
};

struct GraphOptimizationOptions {
  // A zero mask intentionally disables every native GraphOptimizationRule.
  // Layout/memory compatibility stages retain their original fixed scheduling
  // and do not use this option as a new opt-out.
  GraphOptimizationRuleMask enabledRuleMask = kDefaultGraphOptimizationRuleMask;
  unsigned maxRewritesPerFunction = 64;
  unsigned ubCapacityBytes = 0;
  unsigned mappingUBCapacityBytes = 0;
  unsigned storeCoalescingUBBudgetBytes = 0;
  unsigned ubSafetyPercent = 80;
  unsigned reservedUBBytes = 0;
  // Cube on-chip buffer capacities for program-mapping resource checks.
  // Zero capacities keep the L0A/L0B/L0C candidate checks disabled.
  unsigned mappingL0ACapacityBytes = 0;
  unsigned mappingL0BCapacityBytes = 0;
  unsigned mappingL0CCapacityBytes = 0;
  unsigned l0SafetyPercent = 80;
  // RowCoalescing changes the launch grid and is valid only for
  // compile_mode="simt_only".  Keep the source selector rather than a
  // second derived force flag so every consumer follows one mode contract.
  std::string compileMode = "simd_simt_template";
  IndependentAxisTensorizeRuleOptions independentAxisTensorize;
  PersistentTaskStripMiningRuleOptions persistentTaskStripMining;
  StoreCoalescingRuleOptions storeCoalescing;
  ResidentLoadForwardingRuleOptions residentLoadForwarding;
  IntermediatePrecisionBoundaryElisionRuleOptions
      intermediatePrecisionBoundaryElision;
  StoreCoveragePlanningRuleOptions storeCoveragePlanning;
  ContiguousBlockAccessFormationRuleOptions contiguousBlockAccessFormation;
  SelectedBlockScanRuleOptions selectedBlockScan;
};

std::unique_ptr<OperationPass<ModuleOp>>
createGraphOptimizePass(GraphOptimizationOptions options = {});

} // namespace cfg
} // namespace triton
} // namespace mlir

#endif // TRITON_TO_GRAPH_GRAPH_OPTIMIZATION_H
