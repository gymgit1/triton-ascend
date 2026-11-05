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
// SelectedBlockScanRule, Rule A (#2686, rule bit 65536): the driver shell.
//
// Only the framework-facing layer: the RewritePlan that owns a pre-verified
// sandbox, the GraphOptimizationRule, the joint Rule A + Rule B commit contract
// and the factory. No IR pattern knowledge; the matcher and the rewriter are
// their own translation units.

#include "TritonToGraph/GraphOptimizationRule.h"
#include "TritonToGraph/ProgramGridTransform.h"
#include "TritonToGraph/QueryRowBatching.h"
#include "TritonToGraph/RowLifting.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/LogicalResult.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "llvm/Support/Debug.h"

#include <memory>

#define DEBUG_TYPE "graph-optimize"

using namespace mlir;
using namespace triton;
using namespace cfg;

// Joint Rule A + Rule B decision point: Rule A never consults a read-only Rule
// B matcher. It hands its private sandbox to the one Rule B entry declared in
// QueryRowBatching.h, which rewrites Rule B's row lifting in place and verifies
// the result, so landing the sandbox means both rewrites are already in it and
// a failure means neither lands. There is no separate enable flag: Rule B's own
// implementation is the gate.

#include "SelectedBlockScanFacts.h"

namespace {

// Identity of a rewrite across re-matches. Driver-internal, so it keeps
// internal linkage and stays out of the shared header.
// Shape identity: two candidates emit the same rewrite even when their IR
// handles come from different modules. This is what compares the live match
// against the sandbox clone's match.
static bool sameScanShape(const ScanCandidate &a, const ScanCandidate &b) {
  return a.recurrence.index() == b.recurrence.index() &&
         a.tiles.index() == b.tiles.index() &&
         a.table.length == b.table.length &&
         tileBlockSize(a.tiles) == tileBlockSize(b.tiles);
}

static bool sameScanCandidate(const ScanCandidate &a, const ScanCandidate &b) {
  // The integers alone are not an identity: two different loops in the same
  // function can have the same shape, and the plan must anchor on the loop it
  // actually matched. The anchor handles are compared first, then the shape.
  if (a.function != b.function || a.loop.forOp != b.loop.forOp)
    return false;
  if (a.recurrence.index() != b.recurrence.index() ||
      a.tiles.index() != b.tiles.index())
    return false;
  if (a.rowPosition != b.rowPosition || a.table.root != b.table.root ||
      a.scanBound != b.scanBound)
    return false;
  if (kTileLoad(a.tiles) != kTileLoad(b.tiles) ||
      vTileLoad(a.tiles) != vTileLoad(b.tiles))
    return false;
  if (scoreDotOf(a.recurrence) != scoreDotOf(b.recurrence) ||
      valueDotOf(a.recurrence) != valueDotOf(b.recurrence))
    return false;
  return a.table.length == b.table.length &&
         tileBlockSize(a.tiles) == tileBlockSize(b.tiles);
}

// ---- Plan / Rule / factory ----

class SelectedBlockScanPlan final : public RewritePlan {
  ScanCandidate candidate;
  unsigned epoch;
  // 已接受并自行验证过的沙箱：里面既有规则 A 的扫描重建，也可能已有规则 B
  // 的行提升与启动契约。
  // The accepted sandbox: it already carries Rule A's scan rebuild and, when
  // the joint gate ran, Rule B's row lifting and launch contract.
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
  Operation *getAnchor() const override { return candidate.loop.forOp; }

  unsigned getCreationEpoch() const override { return epoch; }

  LogicalResult revalidate(GraphOptimizationContext &context) const override {
    if (context.getFunction() != candidate.function)
      return failure();
    // Re-run the full structural match and compare the anchor and parameters.
    std::optional<ScanCandidate> current =
        matchSelectedBlockScan(candidate.function);
    if (!current || current->loop.forOp != candidate.loop.forOp ||
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
    // Commit only: the sandbox is already verified.  A runtime grid makes Rule
    // B append hidden extent arguments, so the function signature has to travel
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
    for (StringLiteral name : {StringLiteral("hacc.coalesce_factor"),
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
         const ResourceSnapshot &resources) {
    // The tablegen accessors used below are not const-qualified in this tree,
    // so work on a local mutable copy of the function.
    triton::FuncOp function = candidate.function;
    ModuleOp module = function->getParentOfType<ModuleOp>();
    if (!module)
      return nullptr;

    OwningOpRef<ModuleOp> sandbox(ModuleOp::create(function.getLoc()));
    // The sandbox must carry the module attributes: Rule B's matcher reads
    // module-level gates, and a sandbox without them would decline forever.
    sandbox->getOperation()->setAttrs(module->getAttrs());

    IRMapping mapping;
    sandbox->getBody()->push_back(function->clone(mapping));
    auto cloned = cast<triton::FuncOp>(&sandbox->getBody()->front());

    // Re-match on the clone so recorded operations point into the sandbox.
    std::optional<ScanCandidate> sandboxCandidate =
        matchSelectedBlockScan(cloned);
    if (!sandboxCandidate || !sameScanShape(*sandboxCandidate, candidate))
      return nullptr;

    // 沙箱诊断只在这一段被吞掉；预算门要在线上函数上发警告，所以它在作用域外。
    // Sandbox diagnostics are swallowed only in this scope: the budget gate
    // warning must reach the user, so it runs after the scope.  Rule B installs
    // its own handler, so its rewrite stays quiet as before.
    {
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
    }

    // 规则 A 自己的形态先过预算门，联合形态在最终沙箱上再过一次（规则 B 会放大
    // tile 与活值集合）。
    // Rule A's own form passes the gate first; the joint form (rule B widens
    // the tiles and the live set) is measured again before the commit.
    if (budgetOverflow(cloned, resources, "rule A's rewrite", function))
      return nullptr;

    // 联合决策：A 直接在沙箱里跑 B 的真实实现（QueryRowBatching.h 的
    // materialize 入口），B 收不收就是唯一的开关，没有额外要同步的规则位。
    // materialize 就地重写并 verify，失败即整个计划作废；成功则沙箱里已经有
    // B 的改写与启动契约，apply() 只需把沙箱提交回去。
    if (!forceApply) {
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
      // Rule B rewrote the sandbox after Rule A's cleanup+verify, so re-run the
      // same hygiene on the joint result before it can be committed.  A failure
      // declines the plan (never fails apply()), matching the rule's contract.
      if (failed(runSandboxStructuralCleanup(*sandbox)) ||
          failed(verify(*sandbox))) {
        LLVM_DEBUG(llvm::dbgs() << "selected-block-scan: joint rewrite failed "
                                   "post-batching verification @"
                                << function.getName() << "\n");
        return nullptr;
      }
      if (budgetOverflow(cloned, resources, "the joint rule A + rule B rewrite",
                         function))
        return nullptr;
    } else {
      LLVM_DEBUG(llvm::dbgs() << "selected-block-scan: force-apply lands rule "
                                 "A without rule B for @"
                              << function.getName() << "\n");
    }

    // Launch-contract mutual exclusion.  A joint rewrite may either hand the
    // grid to the launcher (hacc.program_grid_transforms / hacc.coalesce_*) or
    // decode a flat program id inside the kernel (tt.get_num_programs) -- never
    // both: the launcher would divide the grid while the kernel still assumes
    // the original one, silently computing a fraction of the queries.
    {
      bool hasGridContract = (*sandbox)->hasAttr(kProgramGridTransformsAttr) ||
                             (*sandbox)->hasAttr("hacc.coalesce_factor");
      bool readsProgramCount = false;
      cloned.walk([&](triton::GetNumProgramsOp) { readsProgramCount = true; });
      if (hasGridContract && readsProgramCount) {
        LLVM_DEBUG(llvm::dbgs() << "selected-block-scan: grid contract and "
                                   "flat program-id decoding must not coexist @"
                                << function.getName() << "\n");
        return nullptr;
      }
    }

    return std::unique_ptr<SelectedBlockScanPlan>(
        new SelectedBlockScanPlan(candidate, epoch, std::move(sandbox)));
  }
};

class SelectedBlockScanRule final : public GraphOptimizationRule {
  bool forceApply;

public:
  explicit SelectedBlockScanRule(const SelectedBlockScanRuleOptions &options)
      : forceApply(options.forceApply) {}

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
               << " (BS=" << tileBlockSize(candidate->tiles)
               << " BN="
               << scanTileWidth(
                      tileBlockSize(candidate->tiles),
                      std::holds_alternative<AdditiveRecurrence>(
                          candidate->recurrence))
               << " S=" << candidate->table.length << ")\n");

    if (auto plan = SelectedBlockScanPlan::create(
            *candidate, context.getEpoch(), forceApply,
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
