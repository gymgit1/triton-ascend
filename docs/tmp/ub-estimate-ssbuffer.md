# 资源估算模型：ssbuffer 改动与 UB 估算说明

> 适用代码：`third_party/ascend/lib/TritonToGraph/ResourceCostModel.cpp`、`third_party/ascend/include/TritonToGraph/ResourceCostModel.h`
> 验证用例：NSA bwd dQ（`parallel_nsa_bwd_kernel_dq`），`B2-T660-H2-HQ32-D60-S16-bs32-f16`，core_num=28
> 片上预算：UB 262,144 B，`ub_safety_percent=80` → 安全线 **209,715 B**

## 0. 结论摘要

| 项 | 数值 |
|---|---|
| R=16 模型估算（本轮改动后，无校准） | ≈278,528 B（推算） |
| R=16 编译器实测 UB | 243,936 B（`Allocated UB size = 1951488 bits`） |
| 模型误差 | +14%（保守） |
| R=16 是否过 80% 安全线 | 否（278,528 > 209,715） |
| R=8 模型估算 / 最终选择 | 139,264 B → 通过，`factor = 8` |
| 改动前模型估算（R=16） | 412,864 B（multibuffer=False）/ 356,368 B（multibuffer=True） |

---

## 1. 本次 ssbuffer 相关改动

### 1.1 背景：ssbuffer 会复制哪些缓冲

`AddDynamicCVPipeline`（ssbuffer）按三类依赖各自维护多份缓冲，默认份数见
`third_party/ascend/lib/DynamicCVPipeline/Common/BufferCountManager.cpp:41-43`：

| 类别 | 默认份数 | 覆盖对象 | 落点 |
|---|---|---|---|
| `InterCore` | 2 | 跨 CUBE/VECTOR 的 transfer：dot 的 A/B 输入、fixpipe 输出 | 输入侧 L1(cbuf)，输出侧 UB |
| `IntraCore` | 3 | 同核内、跨迭代有依赖的循环临时量 | UB |
| `LoadStore` | 1 | GM→UB 的 load | UB |

几点实测事实（来自手写版 `nsa_bwd_dq_kernel_tiled_stream` 的 ttadapter）：

- 多缓冲分配已经落在 IR 上：同一逻辑标量出现多份拷贝（手写版 R=16 产物里 `6 × 1x32xi32`、`6 × 256x32xi32`、`4 × 4x16x16x16xf16`，对应 `IntraCore=3` 与 ping-pong 的叠加），dot 的操作数常驻 cbuf；
- 但 `hivm.dynamic_plan` / `hivm.load_buffer_pool` 均未出现（`canEnableDynamicPlan` 判否），说明“多缓冲已落地”和“启用动态调度”是两件事，后者只影响运行期调度属性；
- 只有预检失败（`SSBUFFER return code=...`）才会 `FallbackHelper.restore()` 回滚这些分配。

### 1.2 改动清单

1. **新增常量**（`ResourceCostModel.h`）

   ```cpp
   constexpr uint64_t kUBInterCoreReplication = 2;          // 跨核 ping-pong
   constexpr uint64_t kUBIntraCoreReplication = 3;          // 同核旋转
   constexpr uint64_t kUBPipelinedReplicatedMaxBytes = 4096; // 仅小缓冲按槽复制
   ```

2. **新增判定函数**（`ResourceCostModel.cpp`）

   - `isDotOperation(Operation *)`：`tt.dot` / `tt.dot_scaled`；
   - `crossesCoreBoundary(Value)`：值的定义或被 dot 使用 → 判定为跨核缓冲（InterCore）；
   - `isDotOperandOnly(Value)`：仅被 dot 消费、且自身不是 dot 结果 → 该值在 lowering 后落 L1(cbuf)，**不计入 UB**；
   - `isIntraCoreBuffer(Value)`：循环内定义、非 `tt.load`/`tt.descriptor_load`，且被 `scf.yield` 携带或逃逸出循环 → 标记 IntraCore；
   - `isPipelinedBuffer(Value)`：循环内 load 或循环内 store 的写入值（原有逻辑）。

3. **活跃区间结构增加两个标志**

   ```cpp
   struct IntervalState {
     ...
     bool pipelined = false;
     bool interCore = false;   // 原 ssbuffer，与 intraCore 命名对齐
     bool intraCore = false;
   };
   ```

4. **峰值扫描中的复制因子合并**

   ```cpp
   uint64_t replication = 1;
   if (state->pipelined && state->bytes <= kUBPipelinedReplicatedMaxBytes)
     replication = kUBMultiBufferReplication;                 // 2
   if (state->interCore)
     replication = std::max(replication, kUBInterCoreReplication);  // 2
   if (state->intraCore)
     replication = std::max(replication, kUBIntraCoreReplication);  // 3
   ```

   用 `max` 而不是相乘：同一缓冲可能同时被 pipelining 和 InterCore 覆盖（循环内 load 再喂给 dot），但物理上只多一份拷贝。

5. **落点过滤**：`addValue` 中，`isDotOperandOnly` 的值直接跳过，不再计入 UB 活跃字节。

6. **pipelined 因子只作用于小缓冲**（≤4 KiB）：依据是实测打开/关闭 multibuffer 的真实 UB 只差 2,048 B，说明大 tile 并没有被复制，只有索引/标量类小缓冲多了一份。

7. **已删除**（试验产物，不在最终代码中）：

   - 估算结果的 68% 校准系数（会把模型从 +14% 高估推到 −22% 低估）；
   - `TRITON_ROW_LIFT_FACTOR`（强制抬升因子）与 `TRITON_GRAPH_UB_IGNORE`（跳过预算闸门）两个环境变量。

---

## 2. 当前 UB 估算模型逻辑

### 2.1 预算来源

`ResourceSnapshot::getSafeUBBudget()`（`ResourceCostModel.cpp:391`）：

```
safeUB = (ub_capacity_bytes - reserved_ub_bytes) × ub_safety_percent / 100
       = 262,144 × 80% = 209,715 B
```

参数由图优化 pass 的 `graph_optimize` 选项注入，与架构相关（910_95/950 为 256 KiB）。

### 2.2 估算步骤（`estimatePeakLiveBytes`）

1. **收集候选值**：遍历函数内所有 op 结果与 block 参数；过滤
   - 非静态形状 / 非常规元素类型 → `ResourceCostRejectReason::DynamicShape` / `UnknownElementType`；
   - 只被 dot 消费的值（`isDotOperandOnly`）→ 跳过，视为 L1 常驻；
   - 记录每个值的字节数、定义序号、最后使用序号；循环携带值（`scf.yield`）延伸到循环末尾。

2. **逐位置扫描**：对每个定义/使用点，累加当前所有活跃值的 `bytes × replication`；
   - 互斥分支（`getConditionalRegion` + `areMutuallyExclusive`）只取最大值，不叠加；
   - 同时维护两个峰值：`peakLiveBytes`（不区分流水）与 `peakPipelinedLiveBytes`（含复制因子）。

3. **输出**：`LiveByteEstimate{ known, overflow, peakLiveBytes, peakPipelinedLiveBytes }`。

### 2.3 闸门如何使用

| 使用点 | 判定 | 失败后果 |
|---|---|---|
| `SelectedBlockScanRule` → `budgetOverflow`（`Rules/SelectedBlockScanBuild.cpp:853`） | `known && !overflow && peakPipelinedLiveBytes <= safeUB` | 整个 joint plan 放弃（不会退化成只做 scan 重写） |
| `QueryRowBatchingRule::fitsQueryRowBudgets` | 同上 | `factor /= 2` 后重试，直到 2 |
| `DotRowCoalescing` | 同上 | 放弃该规则 |

注意：**R 的选择不使用 UB**。`selectRowsPerProgram`（`RowLifting.cpp:206`）里的 `peakLiveBytes` 由两个调用点硬编码为 0，R 实际由 L0A（`52428/maxL0ABytes`）、L0C、单值字节、`grid/core_num` 取 min 决定；UB 只决定“这个 R 能不能用”。

---

## 3. 与实际 UB 的差别及原因

### 3.1 数据对比（R=16，同一 TIR）

| 配置 | 改动前模型 | 改动后模型 | 编译器实测 | 实测占 262,144 |
|---|---|---|---|---|
| multibuffer=False | 412,864 | ≈278,528* | **241,888 B** | 92.3% |
| multibuffer=True | 356,368 | ≈278,528* | **243,936 B** | 93.1% |

\* 改动后 R=16 未过闸门，无打印；数值按“实测 68% 校准版 189,399 / 0.68”推算。

### 3.2 差异来源

1. **词法活跃 vs 物理槽复用（主因）**
   模型按 `[定义, 最后使用]` 区间求和，只识别“互斥分支”这一种复用。真实分配器在生命周期结束后立刻复用同一 slot：R=16 产物里 48 个 UB 分配合计 **1,573,632 B**，而实际峰值只有 **243,936 B**，复用率 **6.4×**。

2. **复制因子作用在“值”上，而不是“槽”上**
   旧版对每个 pipelined 大 tile 整块 ×2（例如 `256x32xf32` 记 32 KB→64 KB）；实测打开 multibuffer 只多 2,048 B。这是改动 6（≤4 KiB 才乘 2）要修的部分。

3. **落点未区分（已修）**
   dot 的操作数在 lowering 后是 `#hivm.address_space<cbuf>`（L1），不占 UB；旧模型把它们当 UB 计入，是 `isDotOperandOnly` 过滤掉的部分。

4. **CUBE/VECTOR 两个 scope 的时间交错**
   ssbuffer 的目的就是让 CUBE 块与 VECTOR 块交替执行、共用同一片 UB；模型把二者在同一位置叠加，等于假设它们完全同时活跃。

5. **大 tile 的下游切分**
   dot 在 lowering 后按 16×16×16 的 fractal 片段处理，部分中间量被折叠/复用；模型按整块 `[256,64]` 计字节。

6. **IntraCore 近似的宽度**
   `isIntraCoreBuffer` 用“循环内定义 + yield/逃逸”近似真实 `intraCoreDependentMap`。真实的 ×3 主要落在小标量（`1x32xi32`、`256x32xi32`），近似规则会把它扩到大张量上，属于偏保守的一侧。

### 3.3 使用建议

- 当前模型在 R=16 上是 **+14% 的保守估计**，方向安全，适合做闸门。
- 若要启用 R=16：实测占用 93.1%，需要显式把 `ub_safety_percent` 提到 ≥94；**不建议**用系数校准，那会让模型在别的形状上低估。
- 换新 kernel/形状做校准的方法：`TRITON_PRINT_UB_BITS=1` / `TRITON_MEMORY_DISPLAY=1` 取 bishengir 的 `Allocated UB size`，与 `TRITON_GRAPH_UB_DEBUG=1` 打印的 `graph-ub[rule] ... demand=` 对照。

---

## 附：复现命令

```bash
# 编译（clean LD_LIBRARY_PATH 避免 trae-cn 的 libstdc++ 污染）
source ~/set_env.sh
export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v trae-cn | paste -sd:)
unset LD_PRELOAD
cd ~/graph/triton-ascend && cmake --build build/cmake.linux-x86_64-cpython-3.11 -j 16

# 编译级验证（不占 NPU）：打印 form / demand / factor
TRITON_GRAPH_UB_DEBUG=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=/tmp/dump \
  TRITON_CACHE_DIR=/tmp/tc python -u /tmp/q3_compile_only.py
```

调试输出含义：

```
graph-ub[rule]: form=the scan rewrite  demand=16592  ubBudget=209715
graph-ub[rule]: form=the joint scan + row-lift rewrite demand=139264 ubBudget=209715
graph-ub[rowcount]: ... limit=25 R=16        # selectRowsPerProgram 的候选 R
```
