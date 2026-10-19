# v0.3 增量建库与同头基排除审计

v0.2 的 2,000,000 条正式结构、所有原始输入哈希和原 Morgan 检索产物保持原样。
v0.3 只向原记录后追加成功组装、全局去重后的新结构；不重新采样或重建原来的 2M 图。
本轮不训练模型，不创建正式训练/验证/测试集。

## 增量选择与化学检查

独立入口为 `scripts/extend_library_v03.py`，默认提出 15,000 个未用过的 head/chain 配对。
它拒绝覆盖现有输出目录，不修改旧版本的 1.5M / 2M 上限或旧采样器。

每个头基计算三个指标：原完整结构数量、已有链 strata 数量、原检索每目标最大相似度的头基内均值。
链 strata 沿用原链长和不饱和度分箱：A/A 使用 linkage × 链长 × 不饱和度，ether/lyso 使用 linkage × 链长。
分别取三个指标的逆向经验百分位，重复值用 midrank，优先分数为：

`0.4 × 低覆盖百分位 + 0.3 × 低多样性百分位 + 0.3 × 低相似度百分位`。

每个入选头基最多提议一个新配对，优先补未有的 strata，再考虑 strata 数量与新 chain token。
候选由固定 seed 的分层旋转抽取，不计算全部 head × chain 笛卡尔积。
化学家族不设新增配额。P/A 维持原少量 exact PE 特殊分支，本轮不提议任何 P 配对。
每个配对的指标、选择原因、是否新增 strata/token 均写入 `increment_selection.parquet`。

组装复用原 `assemble`、RDKit sanitize、元素/同位素/自由基/电荷/价态检查及分子式元素向量校验。
先以原 canonical SQLite 的只读连接排除旧结构，再以完整 canonical 字符串去除增量内部重复。
独立审计还从实际 v0.2 Parquet 再扫描一次，确认 canonical 与配对均无交集，避免依赖旧 SQLite 正确性。
所有新增图均独立重组、重算分子式与指纹；原 2M 的每行、每列均与 v0.3 前缀比较。
失败和重复记录单独输出，不自动补齐数量。

`input_hashes.json` 按原始字节复制。部分历史路径含 Windows 编码造成的乱码时，
以仓库相对后缀解析实际路径并重新核对原 SHA-256；解析结果另存 `resolved_frozen_inputs.json`，不改原记录。

## 新版指纹索引

`scripts/build_v03_index.py` 以整个 v0.3 建立独立 cache 和 USearch 索引。
Morgan 参数仍为 radius=2、2048 bits、includeChirality=false；阈值 0.4，返回最多 10 个不同 canonical 图。
指纹相同的不同图仍保留；128-bit 分组哈希均与完整 2048 bits 核对，防止哈希碰撞合并。
新分组序号按新 cache 重建，未将旧 USearch 文件直接拼接到改变后的 group 编号上。
新索引经过 10,000 目标 / 20,000 参考的 exact benchmark 和新增/全库抽样查询核验。
这个索引验证不等同于对 v0.3 做完整检索普查。

## 同头基排除实验

对照实验使用冻结的 **v0.2 全部 2M 作为目标及模板池**，与原完整检索结果对照。
v0.3 的索引单独交付，不混用两个版本的分母。

两个实验条件都排除目标自身，使用相同 Morgan 参数、Tanimoto ≥0.4 和 K≤10：

1. identity-only：复现原 16 个近邻 FP group 的 rerank 规则，重新记录每个目标的实际邻居。
2. same-polar excluded：排除所有 `template.polar_id == target.polar_id` 的模板，再选前十条。

同头基比例同时提供三个口径：全部检索目标–模板对中的比例、至少有一个同头基模板的目标比例、
top1 同头基比例。缺少邻居时只以实际邻居为分母；相似度均值不补零。

不能先截断 posting 前 11 条再做头基过滤，也不能复用“相同 FP group 有 ≥11 条就必有 10 个模板”的捷径。
新逻辑扫描至 `全库每头基最大记录数 + K` 的 posting 前缀，候选 FP group 深度至少为该值再加一，
并纳入当前 ANN 候选列表中的相似度边界 ties。相同指纹、不同头基的图仍是合法模板。

基准特意富集相同头基的完整链背景；单纯随机抽独立结构，会低估这种过滤对检索的影响。
默认 2,000 目标 / 20,000 参考，4,000 万次 exact Tanimoto 比较。
核对 native B1 Tanimoto 与 RDKit 的数值一致性，对所有最终选中组用 RDKit 重算精确分数。
近邻召回仍属于 ANN 的范围；它不证明全库 exact top10。

全库检索最初取 512 个 group，必要时扩展至 1024 / 2048 / 4096。
**过滤后仍不足 10 个邻居的结果，逐个与全库所有 FP group 精确比较**，确认覆盖率与真缺失，
不把候选截断造成的空结果直接判为无模板。这个步骤不会计算完整的 N×N 矩阵。

每条目标的两组邻居 row index 和相似度都存为 `.npy`；row index 对应冻结原 cache 的 `row_ids.npy`。
全部结果逐行检查无自匹配、无重复、分数过阈值及排除条件。
checkpoint 绑定结构、索引、polar code 哈希和检索设置，可断点续跑；CPU worker 共享只读 mmap。

## 本地执行入口

```powershell
../.venv/Scripts/python.exe scripts/extend_library_v03.py
../.venv/Scripts/python.exe scripts/audit_increment_v03.py
../.venv/Scripts/python.exe scripts/build_v03_index.py --threads 2
../.venv/Scripts/python.exe scripts/audit_head_exclusion.py --threads 1 --workers 2
```

USB 硬盘上运行时，可以给最后一项指定更快存储中的 `--output` 和 `--working-cache`。
working cache 必须是同一原 cache 的副本；source/index/polar code 哈希必须与断点一致。
归档完成后将新产物复制回 `data/private/head_exclusion_audit_v0.1/`，再次核对全部文件哈希。
计算副本可清理，原始 cache 始终保留。

主要本地产物：

- `data/structure_library_v0.3/structures.parquet`：原 2M 全部记录 + 成功增量。
- `increment_structures.parquet`、`increment_selection.parquet`：实际新增图与提议配对的来源/选择依据。
- `baseline_preservation_manifest.json`、`preservation_verification.json`、`verification_audit.json`：保留与化学审计。
- `data/private/morgan_index_v0.3/`：新 Morgan cache、指纹分组、索引及 benchmark。
- `data/private/head_exclusion_audit_v0.1/`：成对全库检索结果、polar 编码、统计和断点。

正式结构、私有输入、大索引和逐目标邻居不上传 GitHub。GitHub 仅保存代码、fixture tests、方法和汇总结果。
