# 2M 正式结构库与 Morgan 检索审计

本轮执行完整结构库组装、质量审计和检索可用性审计，不训练、不创建 train/validation/test 划分。
所有批量输入、完整结构、指纹索引和逐目标指标保存在本地，不上传 GitHub。
正式实测数量、分组检索指标与 API 核验见 [FULL_LIBRARY_V02_RESULTS.md](FULL_LIBRARY_V02_RESULTS.md)。

## 本轮最新 P 链规则

用户在执行过程中明确更新：**不要求 61 个 P-chain token 全覆盖，不将 P 覆盖设为正式组装前的硬性条件，
不为 P 覆盖改变 2M 主体采样**。

因此本轮没有新增 183 个 P/A mandatory pairs；保留原有少量 exact PE-P/A 特殊分支。
非 exact PE 不使用独立 P 标签，使用既有 O-chain 表示与 O 链池。
冻结 A177/O84/P61 链表和代表双键位置保持不变，不新增臆测的 P→O token 映射。
所有 P 使用仍须满足 exact PE 权限；P 覆盖仅统计，A/O 仍要求全部至少出现一次。

`prepare_2m_library.py` 逐个枚举原始 2M head/chain 配对并统计链 token；
配对 schedule SHA-256 必须与上一轮原始 2M dry-run 完全一致。
维持三组共同 A/A anchors、五个特殊头基、generated keto-inositol，以及 2:1:1 family extras 权重。

```bash
python scripts/prepare_2m_library.py
python scripts/build_2m_library.py --execute
python scripts/audit_full_library.py
python scripts/audit_morgan_readiness.py
```

## 正式组装

正式输入为 expanded polar v0.2 的 195,305 个头基，加一个既有 generated keto-inositol。
候选配对上限 2,000,000；拒绝或 canonical 去重后不自动补库。

采用最多四个 CPU worker，每批 800 个头基。每个分片按原配对计划执行：
完整图组装 → sanitize → 十种支持元素检查 → 总电荷 0 → 中性分子式及元素向量检查
→ canonical non-isomeric SMILES → Morgan 2048-bit 指纹。
现有 InChIKey 字段沿用原 builder 的计算与验证。
片段图缓存只用于复用解析结果，不改原图。每批候选图独立写成 Parquet，并保存输入签名、
完成计数、拒绝理由和文件 SHA-256。已成功且签名匹配的分片可在中断后复用。

全局 canonical 去重使用磁盘 SQLite 存储完整 SMILES 键，不采用有碰撞风险的单独哈希去重。
完整结构分批流式输出，不在内存中保存两百万条记录。
重复候选的来源映射另存 `canonical_duplicate_provenance.parquet`；主表每个 canonical 图只保留一条。
正式输出在 `data/structure_library_v0.2/`，工作分片在 `data/private/library_2m_assembly_work/`。

主表保留 structure_id、canonical_smiles、neutral_formula、architecture、chain_tokens、
polar_id、polar_fragment_smiles、chemistry_family、plausibility tier、structure_origin、is_special_case、
直接/父 CID、attachment provenance 和 256-byte Morgan 指纹。
来源字段只作追溯；family 权重仅按 chemistry_family 分配，不按来源分配。
sn 是确定性的计算代表放置，不是对生物 sn 位置的预测。

## 质量审计

组装阶段每条成功候选都已真实经过 RDKit 检查。正式报告分开记录 planned pairs、
successfully assembled、sanitize/formula/其他拒绝、canonical duplicates、final unique。
同时检查 family、AA/OA/PE-PA/lyso、来源与 tier 分布、全部头基的 min/median/max、
共同 anchors 与五个特殊头基覆盖。

独立审计再次逐条核对最终主表的 formula 元素向量、structure_id 哈希、头基关联、
P 只用于 exact PE、生成 analog 不冒认 direct source CID，并重新统计链/头基/family 覆盖。
均匀随机抽取 300 个完整图，额外包含全部 special 和 PE-P/A 图，从冻结片段重新组装，
独立比对 canonical SMILES、formula 和重新计算的 Morgan 指纹。
不把样本重建描述为对全库第二次独立重建。

## 索引与精确 benchmark

检索固定 Morgan radius=2、2048 bits、Tanimoto、similarity ≥0.4、K≤10，排除自身和相同 canonical 图。
图不同而指纹相同的模板仍合法，不因 fingerprint collision 而合并结构。

将二进制指纹以 memory-mapped NumPy 数组保存。先用 128-bit digest 分组，再逐条比较完整
2048 bits 确认组内一致；digest 不一致或碰撞不会静默合并。
每个相同指纹组保存按 structure_id 排序的 canonical 图 postings。
HNSW 仅索引不同的指纹向量，使用 [USearch](https://github.com/unum-cloud/usearch) 2.26.4 的
bit-packed Tanimoto 距离，connectivity=32、expansion_add=256。
候选组再经 **RDKit BulkTanimotoSimilarity 精确重排**，最后展开为不同完整结构模板。
HNSW 返回的候选已按原生 Tanimoto 距离排序；保留前 16 个不同指纹组交给 RDKit 重排，
覆盖十个模板及排除目标所需的组数，并在相同流程的 exact benchmark 中检验召回。
RDKit 指纹对象缓存最多 65,536 组，避免反复构造对象导致全库审计过慢。
若某指纹组至少有 11 个不同图，则排除目标后可直接得到十个 similarity=1 的模板；这是精确捷径。

先固定 seed=42，从正式库随机抽 20,000 个 reference graphs，其中随机取 10,000 个 targets。
对该 reference subset 做 200,000,000 次 exact Tanimoto 比较，并与索引候选重排结果比较。
这些行仅用于近邻算法 benchmark，不作为训练/验证划分。

尝试 `(expansion_search, candidate_groups)` 为 (128,64)、(256,128)、(512,256)、(1024,512)，
选首个满足以下阈值的设置，再用于正式全库索引：

- tie-aware recall@10 ≥0.99；
- Top-1 score recall ≥0.995。

同时报告严格 ID recall@10 与平均 score regret，避免隐瞒并列造成的 ID 差异。
Tie-aware 定义：必须找回所有严格高于 exact 第 K 名分数的模板；剩余名额可从该分数的并列图中任取。
小规模 reference subset 的召回表现不是全库检索严格精确的证明，全库统计明确标记为 approximate search based。
未执行全库 O(N²) 两两比较。

## 全库检索可用性审计

对全库所有目标统计 ≥1、≥5、10 个 usable templates 的比例、zero-template 数量、
max similarity 和 top-10 mean similarity 分布。
不足十个模板时，mean 为可用模板的均值，不补零；无模板目标另计。
同指纹组内所有目标的数值指标相同，可按组计算后精确映射回每个 canonical 结构。
按 chemistry_family、CORE/EXTENDED、结构来源，以及 PubChem-derived / generated analog 分组。
来源和标签仅在统计分组时使用，不进入候选检索。
对罕见零模板结果追加分块 exact Tanimoto 扫描以确认；最多复核 100 个零模板指纹组，
若超过该上限则明确报告尚未精确确认，避免退化为全库两两计算。

`IndexedMorganRetriever.retrieve_smiles()` 提供可复用查询 API，返回结构 ID、Parquet row index 和精确重排分数。
`data/private/morgan_readiness_v0.1/` 保存原表哈希、指纹缓存、HNSW、benchmark、逐目标指标与汇总报告。

数据划分、检索泄漏规则以及 structure-only 训练留待本轮库和审计结果完成后单独讨论。
