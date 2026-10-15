# MS-DIAL 真实链证据提取结果 v0.1

执行日期：2026-10-15。2025、2022 均全量完成；仅输出人工审核材料。
未训练，未生成最终 150 万结构库，未生成批准版 chains.tsv 或 p_to_o_mapping.tsv。

## 扫描与性能

|项目|2025|2022|
|---|---:|---:|
|总记录|5,263,863|3,879,270|
|有 SMILES|5,257,054|3,872,461|
|无 SMILES|6,809|6,809|
|unique SMILES（原字符串）|2,514,906|1,889,960|
|unique InChIKey|2,514,900|1,889,954|
|宽筛候选记录|860,124|757,014|
|RDKit 验证有效候选记录（含缓存）|860,124|757,014|
|识别到甘油磷酸骨架|740,246|660,623|
|有普通链证据的源记录|739,887|660,272|
|链 observation|1,874,828|1,724,806|
|候选 invalid SMILES|0|0|
|容器/字段解析错误|0|0|
|本版本新计算的不同 SMILES|484,136|3,300|
|复用另一版图解析的不同 SMILES|0|443,488|
|已提交 checkpoint 累计扫描秒数|2,408.53|907.13|
|峰值 RSS MiB|1,553.83|1,339.59|

峰值 RSS 以 100 ms 采样，包含 native 解压和 mmap 工作集；累计扫描时间含解压、读取、化学处理、写盘，不含迁移、汇总和中断时未提交片段的重算。
两个全量运行实际墙钟时间（含迁移、汇总及最终核验）：81.58 分钟。
读取说明：早期字段核查曾误展开一条含 5 项的极小谱峰列表，未打印峰值、未用于证据提取。全量生产扫描对 Spectrum 调用 skip()，不反序列化峰列表。
2022 逐条读取独立元数据；只对完全相同 SMILES、相同化学算法/RDKit 指纹复用纯图解析结果，支持数和名称不复用。
1 万 → 10 万试跑通过后才开始全量；10 万阶段为 22,752 条候选、41,712 条链 observation。

## Token 与候选状态

|项目|2025|2022|
|---|---:|---:|
|unique A tokens|183|178|
|unique O tokens|84|84|
|unique P tokens|61|56|
|UNIQUE|322|312|
|DOMINANT|2|0|
|CONFLICT|4|6|
|LOW_EVIDENCE|0|0|

状态是库内审核分级，不是天然结构正确性的置信度。LOW_EVIDENCE 优先；DOMINANT 要求首位≥0.80、次位≤0.20。
父结构以 non-isomeric canonical full SMILES 去重；同一母结构重复链会增加 observation，但只增加一次 parent 支持。

## 五条 anchor

|Token|2025 主要 DB 位置|2025 母结构支持|2025 位置版本数|2022 主要 DB 位置|版本结果|
|---|---|---:|---:|---|---|
|A-16:0|[]|22858|1|[]|MATCH|
|A-18:0|[]|22859|1|[]|MATCH|
|A-18:1|[9]|22858|1|[9]|MATCH|
|A-20:4|[8, 11, 14, 17]|19224|1|[8, 11, 14, 17]|MATCH|
|A-22:6|[4, 7, 10, 13, 16, 19]|19224|1|[4, 7, 10, 13, 16, 19]|MATCH|

**A-20:4 必须重点审核：库内位置不能自动当作常见花生四烯酸的 Δ5,8,11,14。**
两个版本一致只能说明库约定一致，不能作为天然双键位置的独立验证。

## 2025 内部多位置候选

|Token|状态|首位 DB 位置|首位 fraction|次位 DB 位置|次位 fraction|
|---|---|---|---:|---|---:|
|A-24:1|CONFLICT|[15]|0.5943|[13]|0.4057|
|A-26:1|CONFLICT|[17]|0.5902|[15]|0.4098|
|A-28:1|CONFLICT|[19]|0.5430|[17]|0.4570|
|A-30:1|DOMINANT|[21]|0.8008|[19]|0.1992|
|A-32:1|DOMINANT|[23]|0.8008|[21]|0.1992|
|A-34:1|CONFLICT|[25]|0.7909|[23]|0.2091|

## 版本核对

- MATCH: 312
- POSITION_CONFLICT: 6
- LINKAGE_CONFLICT: 0
- ONLY_2025: 10
- ONLY_2022: 0

上述 POSITION_CONFLICT 的位置版本集合在两库相同，分歧来自首位候选排序变化；不是某库独有了新的位置版本。

|Token|版本结果|2025 状态|2022 状态|
|---|---|---|---|
|A-10:1|ONLY_2025|UNIQUE|ABSENT|
|A-11:1|ONLY_2025|UNIQUE|ABSENT|
|A-12:1|ONLY_2025|UNIQUE|ABSENT|
|A-14:2|ONLY_2025|UNIQUE|ABSENT|
|A-24:1|POSITION_CONFLICT|CONFLICT|CONFLICT|
|A-26:1|POSITION_CONFLICT|CONFLICT|CONFLICT|
|A-28:1|POSITION_CONFLICT|CONFLICT|CONFLICT|
|A-30:1|POSITION_CONFLICT|DOMINANT|CONFLICT|
|A-32:1|POSITION_CONFLICT|DOMINANT|CONFLICT|
|A-34:1|POSITION_CONFLICT|CONFLICT|CONFLICT|
|A-8:1|ONLY_2025|UNIQUE|ABSENT|
|P-10:1|ONLY_2025|UNIQUE|ABSENT|
|P-11:1|ONLY_2025|UNIQUE|ABSENT|
|P-12:1|ONLY_2025|UNIQUE|ABSENT|
|P-14:2|ONLY_2025|UNIQUE|ABSENT|
|P-8:1|ONLY_2025|UNIQUE|ABSENT|

## P / O 与 PE-P

输出 token 的 P 双键数包含图中 vinyl C=C；原名称计数口径另存 source_p_db_count_includes_vinyl。
O/P 等质量只在实际观察到的 token 并集中比较附着链残基组成，不创建新 token，不批准映射。

- 2025 PE-P unique tokens: 61
  - naming:P:DB_COUNT_DIFFERS: 5,551
  - source_p_includes_vinyl:False: 5,551
- 2022 PE-P unique tokens: 56
  - naming:P:DB_COUNT_DIFFERS: 4,760
  - source_p_includes_vinyl:False: 4,760
  - naming:P:SOURCE_LABEL_LINKAGE_CONFLICT: 4,704
  - source_p_includes_vinyl:True: 4,704
  - naming:O:SOURCE_LABEL_LINKAGE_CONFLICT: 588
- 实际观察到的 O/P 等组成候选对：56

源标签与图分类不符的 observation：{"2025": 0, "2022": 5292}
2022 共 5,292 条源标签与图分类不符：4,704 条用 O- 名称对应图 P，588 条用 P- 名称对应图 O。相关 63 个 token 均强制复核。
2022 另有 P- 名称不计 vinyl 双键的记录；不能把这些记录与 O- 名称的图 P 混成一个统一的源 P 计数口径。
source_naming_conventions.tsv 保留实际原名称和计数；版本表新增 source_label_conflicts 字段并强制相关 token 人工复核。
LINKAGE_CONFLICT 的 0 指图结构比较；不表示源名称的 O/P 定义完全一致。
PE-P 专表严格排除 LPE / EtherLPE；它们仍保留在完整链证据中。

PE-P 每 token 的代表位置、去重母结构数和名称示例见 pe_plasmalogen_chain_report.tsv。

## 拒绝与范围限制

### 2025

- rejection:unsupported_modified_chain: 32,062
- detail:branched_chain: 3,792
- rejection:not_target_backbone: 119,878
- detail:no_linear_glycerol_phosphate: 119,878
- detail:heteroatom_or_oxidized_chain: 28,270
- rejection:cannot_identify_chain_bond: 188
- detail:no_attached_chain: 188

修饰链拒绝涉及类别：{"Ac2PIM1:unsupported_modified_chain": 24, "Ac2PIM2:unsupported_modified_chain": 24, "Ac3PIM2:unsupported_modified_chain": 288, "Ac4PIM2:unsupported_modified_chain": 3456, "LPC:unsupported_modified_chain": 7, "LPE:unsupported_modified_chain": 6, "PC:unsupported_modified_chain": 7, "PE:unsupported_modified_chain": 6, "PG:unsupported_modified_chain": 6, "PI:unsupported_modified_chain": 6, "PS:unsupported_modified_chain": 6, "PA:unsupported_modified_chain": 1, "EtherOxPC:unsupported_modified_chain": 7056, "EtherOxPE:unsupported_modified_chain": 2352, "Others:unsupported_modified_chain": 1, "OxPC:unsupported_modified_chain": 9408, "OxPE:unsupported_modified_chain": 2352, "OxPG:unsupported_modified_chain": 2352, "OxPI:unsupported_modified_chain": 2352, "OxPS:unsupported_modified_chain": 2352}

### 2022

- rejection:unsupported_modified_chain: 24,977
- detail:branched_chain: 3,792
- rejection:not_target_backbone: 96,391
- detail:no_linear_glycerol_phosphate: 96,391
- detail:heteroatom_or_oxidized_chain: 21,185
- rejection:cannot_identify_chain_bond: 188
- detail:no_attached_chain: 188

修饰链拒绝涉及类别：{"Ac2PIM1:unsupported_modified_chain": 24, "Ac2PIM2:unsupported_modified_chain": 24, "Ac3PIM2:unsupported_modified_chain": 288, "Ac4PIM2:unsupported_modified_chain": 3456, "EtherOxPC:unsupported_modified_chain": 4704, "EtherOxPE:unsupported_modified_chain": 2352, "Others:unsupported_modified_chain": 1, "OxPC:unsupported_modified_chain": 4704, "OxPE:unsupported_modified_chain": 2352, "OxPG:unsupported_modified_chain": 2352, "OxPI:unsupported_modified_chain": 2352, "OxPS:unsupported_modified_chain": 2352, "PC:unsupported_modified_chain": 3, "PE:unsupported_modified_chain": 2, "PS:unsupported_modified_chain": 2, "PG:unsupported_modified_chain": 2, "PI:unsupported_modified_chain": 1, "PA:unsupported_modified_chain": 1, "LPC:unsupported_modified_chain": 3, "LPE:unsupported_modified_chain": 2}

头基糖上的额外酰链、N-acyl 等不属于本次甘油 O-C 拆链范围；不会从名称补链或补双键位置。
环/支链/额外杂原子/三键等明确拒绝；同一母结构中的普通链仍可贡献证据。
库内支持数不是生物丰度，也不能据此证明某个位置版本更符合天然样品。

## 文件与核验

新增代码位于 src/structure_pretrain/chain_extraction/；入口 scripts/extract_msdial_chain_evidence.py。
配置 configs/chain_extraction.yaml；独立依赖 requirements-extraction.txt；独立 CI .github/workflows/chain_extraction.yml。
本地审核包：data/private/msdial_chain_review/。大表的 .parquet 路径为分片数据集目录。
根目录另有两版 metadata.parquet 和 metadata_report.json；_state/ 保存 SQLite/checkpoint/化学指纹。

- anchor_chain_validation.json
- chain_version_comparison.tsv
- extraction_report.json
- msdial2022_chain_crosscheck.parquet
- msdial2022_chain_evidence.parquet（分片目录）
- msdial2022_chain_representative_candidates.tsv
- msdial2022_chain_variants.parquet
- msdial2022_metadata_report.json
- msdial2025_chain_evidence.parquet（分片目录）
- msdial2025_chain_representative_candidates.tsv
- msdial2025_chain_variants.parquet
- msdial2025_metadata_report.json
- op_equivalence_candidates.tsv
- pe_plasmalogen_chain_report.tsv
- rejections.parquet（分片目录）
- source_naming_conventions.tsv
- verification_audit.json

测试：39 项提取测试、48 项现有建库测试通过；无 Torch/训练导入。
核验检查：全库记录索引连续、metadata/证据/拒绝行数与 checkpoint 一致；全部 positional variants 重新从 fragment graph 解析核对；候选 parent 支持与 SQLite 去重结果一致。

verification_audit.json 记录实际核验计数。原库和大表不上传 GitHub；现有 builder 和模型代码未修改。

本机项目位于 E: USB HDD。2022 中途停止在已提交 checkpoint，完整复制 SQLite 状态（含 WAL/SHM）到 C: NVMe，quick_check 通过后通过原路径目录联接续跑；E: 保留旧状态备份。
活动状态目录：C:\Users\Administrator\.cache\msdial-chain-extraction\20261015-v01
HDD 旧状态备份：E:\yzx\磷脂结构预测模型\phospholipid-structure-pretrain\data\private\_state_hdd_backup_20261015
