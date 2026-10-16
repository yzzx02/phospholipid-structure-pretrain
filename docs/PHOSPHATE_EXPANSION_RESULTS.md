# Phosphate 扩充与 200 万预算模拟结果

2026-10-16。按本轮补充要求，新版完整库预算上限为 **2,000,000**，同时报告 1,500,000 对照。
本轮完成极性片段扩充、来源核验与链组合 dry-run；**未运行最终完整库组装，未训练模型**。
方法、配置和复现入口见 [PHOSPHATE_EXPANSION.md](PHOSPHATE_EXPANSION.md)。

## 1. 冻结 baseline 与全审核表统计

原正式表未覆盖，103,397 个头基全部保留。
SHA-256：`8b675f2b3d6a5d19dbc6801c3cf68a5782d153a5c9107fc2def2572177bbea8b`。

| Chemistry family | Baseline | APPROVABLE | MULTIPLE_ATTACHMENT_AMBIGUITY | CHARGE_AMBIGUOUS | FILTER_REVIEW |
|---|---:|---:|---:|---:|---:|
| phosphate | 39,591 | 42,169 | 3,858 | 70,392 | 201,357 |
| phosphonate | 62,867 | 64,644 | 7,215 | 2,944 | 22,232 |
| polyphosphate | 939 | 990 | 31,615 | 6,911 | 36,387 |

四类状态合计 **490,714** 个审核片段。Baseline 是第二层筛选后的正式表，
因此不应与第一层 APPROVABLE 列直接等同。

## 2. MULTIPLE_ATTACHMENT 救回

对 29,208 个真实 source CID 重新枚举，得到 103,139 个酸性位点候选，
合并等价位点后对 42,688 个目标片段重新执行化学、plausibility 和 builder 边界检查。

| Family | 原 MULTIPLE_ATTACHMENT 片段 | 通过筛选 | 去重后新增 |
|---|---:|---:|---:|
| phosphate | 3,858 | 3,586 | 3,586 |
| phosphonate | 7,215 | 7,060 | 7,060 |
| polyphosphate | 31,615 | 31,279 | 31,279 |
| **合计** | **42,688** | **41,925** | **41,925** |

CORE 5,395、EXTENDED 36,530、HOLD 198、REJECT 565。
通过者与 baseline 无重复；未将 CHARGE_AMBIGUOUS 或 FILTER_REVIEW 片段救回。
输出保存真实 CID、原始 RDKit 原子索引和去盐后组分内索引的对应关系。

## 3. 受控 phosphate analog

救回后 phosphate 为 43,177，仍少于 phosphonate 69,927，因此继续执行受控转换。
对全部 69,927 个可用 phosphonate 检查，满足单 P、单个非环 P–C 单键等限制的有 **53,908** 个。
排除多 P 15,522 个、环内 P–C 497 个。

全部 53,908 次转换均通过“全部重元素不变、仅 O+1”的组成检查。
转换后再次筛选：CORE 26,208、EXTENDED 27,691、HOLD 9；
9 个 HOLD 因新形成的局部连接触发既有 builder 边界而未保留。

| 转换及去重结果 | 数量 |
|---|---:|
| 满足转换条件 | 53,908 |
| 转换后通过 CORE/EXTENDED | 53,899 |
| 与 baseline / rescued phosphate 重复 | 3,737 |
| 与其他生成 analog 重复 | 179 |
| 重复合计 | 3,916 |
| **新增 unique phosphate analog** | **49,983** |

新增 analog 标记为 `generated_phosphate_analog`；其 `source_cids=[]`、`source_count=0`。
真实父分子 CID 仅保存在 `parent_source_cids`，不冒充生成图自身的数据库记录。
这些结构是规则生成的候选，不代表对应完整脂质已在数据库或实验中发现。

## 4. 扩充后的极性片段表

| Family | Baseline | 救回新增 | Analog 新增 | Expanded v0.2 |
|---|---:|---:|---:|---:|
| phosphate | 39,591 | 3,586 | 49,983 | **93,160** |
| phosphonate | 62,867 | 7,060 | 0 | **69,927** |
| polyphosphate | 939 | 31,279 | 0 | **32,218** |
| **合计** | **103,397** | **41,925** | **49,983** | **195,305** |

其中 **145,322** 个有真实 PubChem 片段来源，**49,983** 个是生成 analog。
未为平衡数量删除任何 phosphonate，来源字段不参与训练目标或采样权重。

正式扩充输出：`data/private/polar_expansion_v0.2/polar_moieties_expanded_v0.2.tsv`。
SHA-256：`1c64767c36323d6e4ec5e9dc8bd73d741d1ce5d0cea94e0df008df0e243b6e6d`。
另有既有特殊结构包中的一个 generated keto-inositol 头基，
所以预算模拟输入为 **195,306** 个头基，其中 phosphate 为 93,161。

## 5. Family-aware 预算对照

固定 seed=42；每头基三个共同 A/A anchors，加上特殊头基必选组合，
共预留 **585,926** 个 head/chain 配对。
额外背景权重 phosphate:phosphonate:polyphosphate = **2:1:1**；
该比例作用于每头基的 extras，不包括共同 anchors。

冻结链表仍为 **A177 / O84 / P61 = 322**，链组合池为 41,679，双键位置未修改。

| 预算项目 | 1,500,000 对照 | **2,000,000 主方案** |
|---|---:|---:|
| Mandatory anchors + special pairs | 585,926 | 585,926 |
| Mandatory 后剩余预算 | 914,074 | 1,414,074 |
| AA | 1,030,092 | 1,232,779 |
| OA | 319,907 | 567,218 |
| PE-P/A | 1 | 3 |
| Lyso | 150,000 | 200,000 |
| **计划配对总数** | **1,500,000** | **2,000,000** |
| 每头基背景数范围 | 6–12 | 7–15 |
| 每头基平均背景数 | 7.6803 | 10.2403 |

| Chemistry family | 1.5M 数量 | 1.5M 占比 | **2M 数量** | **2M 占比** |
|---|---:|---:|---:|---:|
| phosphate | 869,895 | 57.9930% | **1,192,847** | **59.64235%** |
| phosphonate | 431,361 | 28.7574% | **552,565** | **27.62825%** |
| polyphosphate | 198,744 | 13.2496% | **254,588** | **12.72940%** |

两个预算都使 phosphate 多于 phosphonate，无须继续调高 family weight。
2M 时每头基平均 extras：phosphate 9.80406，phosphonate 4.90203，polyphosphate 4.90204。
所有头基保留，且每个头基下的 chain combination ID 不重复。

这两列是**精确枚举的计划 head/chain 配对数**，尚未生成完整分子图。
正式组装时的 sanitize 拒绝和完整图 canonical 去重可能减少实际条数，
因此不能将 dry-run 数字称为已经生成的 unique 完整结构数。

## 6. 特殊结构与 P 链覆盖缺口

后续执行时用户已明确撤回 P 全覆盖要求：P/A 仅保留为 exact PE 的少量特殊分支，
不新增强制覆盖配额，不改 2M 主体采样。下文的覆盖预留建议仅为本轮历史分析；
最新执行规则见 [FULL_LIBRARY_V02.md](FULL_LIBRARY_V02.md)。

既有五个 cyanomethyl / cyanoethyl / keto-inositol 特殊头基全部继续 must-include。
在 2M 计划中分别获得 14、14、15、15、13 个不同背景；必选代表组合也保留。

**当前计划还不适合直接作为覆盖完整的训练库。** 现有规则仅允许 exact PE 使用 P/A，
而符合条件的 exact PE 头基只有一个。当前每头基近似均匀的预算令 P/A 采样过少：

| 预算 | P/A 配对数 | 覆盖 P chain token | 未覆盖 |
|---|---:|---:|---:|
| 1.5M | 1 | 1 / 61 | 60 |
| 2M | 3 | 3 / 61 | 58 |

2M 仅覆盖 `P-22:2`、`P-22:4`、`P-24:4`。
建议正式组装前先为每个冻结 P token 预留至少一个 exact-PE P/A 配对，
并从其他 extras 等量扣减，继续保持总数 ≤2M。
**此额外覆盖预留尚未实施**，本轮保留原 2:1 dry-run 结果供后续调整。
该缺口已单列在 `family_dry_run/pe_pa_coverage_gap.json`。

## 7. 核验与测试

独立审计 `verification_audit.json` 通过：

- 核对 baseline 103,397 个图保留及 expanded 195,305 个头基。
- 检查 28,756 个来源 CID 的 attachment 端点，共 101,153 个原子索引记录；
  其中独立重建并核对 canonical 片段 **150** 个样本。
- 检查全部 53,908 个合格转换的 O+1 重元素变化，以及生成图 CID 无冒认。
- 冻结链表未变，SHA-256 为 `7364cac1d631109e21453e71e641092899a4e819b4f7696f95b8af23abf1d8c1`。
- 两档预算均未超限，五个特殊头基均保留。

六组独立化学测试共 **211 项**：library_builder 48、pubchem_polar 49、
chain_extraction 39、polar_plausibility 23、library_v02 20、polar_expansion 32。
新增测试覆盖位点拆分、等价合并、反应性与 builder 边界、单 O 插入、
环 P–C / 多 P 排除、dedupe、来源字段、确定性采样、旧版预算兼容，
以及微型实际组装与 dry-run 配对一致性。运行不依赖 Torch，不训练模型。

## 8. 本轮范围与产物

本轮没有继续联网补采 PubChem。原快照仍为 **583,937 / 983,258 CIDs**，
未获取的 399,321 个不在本轮扩充范围中。

本地目录 `data/private/polar_expansion_v0.2/` 保存 baseline manifest、family/status 交叉表、
rescued TSV、逐转换审计、expanded TSV、来源核验和两档预算报告。
GitHub 只同步代码、方法和聚合结果，不上传来源库和私有批量表。
初次 builder 边界调试产物保存在 `polar_expansion_initial_boundary_draft/`，已由正式结果取代。

明确停止于本轮计划指定的 dry-run：**未训练，未运行最终 1.5M 或 2M 正式组装**。
