# MS-DIAL 真实链证据提取 v0.1

本模块只提取人工审核材料，不训练、不建最终结构库、不生成批准版 chains.tsv / p_to_o_mapping.tsv。
现有 library_builder、GINE、Morgan 和 Transformer 保持不动。

## 安装与运行

```powershell
python -m pip install -r requirements-extraction.txt
python -m pip install --no-deps -e .
python scripts/extract_msdial_chain_evidence.py --source 'D:\MSDIAL\Msp20250409095351_NCDK_conventional_converted_dev.lbm2' --version 2025 --max-records 10000 --output data/private/validation_final
# 检查表结构/统计后，再增加 9 万条；max-records 指本次新增记录数。
python scripts/extract_msdial_chain_evidence.py --source 'D:\MSDIAL\Msp20250409095351_NCDK_conventional_converted_dev.lbm2' --version 2025 --max-records 90000 --resume --output data/private/validation_final
# 两级验证完成后全扫，默认输出到 data/private。
python scripts/extract_msdial_chain_evidence.py --source 'D:\MSDIAL\Msp20250409095351_NCDK_conventional_converted_dev.lbm2' --version 2025
python scripts/extract_msdial_chain_evidence.py --source 'D:\浏览器下载文件\MSDIAL4.9.221218\MSDIAL ver.4.9.221218 Windowsx64\Msp20221205132019_conventional_converted.lbm2' --version 2022
```

中断后同一命令加 `--resume`；`--start-block` 从零计数，从后续块开始会明确标记为不完整。
恢复会核对源路径、大小、mtime、配置和起始块；不要在同一个状态目录混用修改过的提取算法。
`--report-only` 从 SQLite 重生成审核汇总，无需再次扫描。所有本地原始数据和审核材料都在 gitignore 中。

## 读取与内存

每个 LBM2 raw LZ4 块用原生解压器写入磁盘 mmap；随后关闭 mmap，以 1 MiB 缓冲供 MessagePack 逐字段读取。
只反序列化六个元数据字段，记录索引由扫描器添加。Spectrum 及其余字段调用 `Unpacker.skip()`。
单个 raw 块约 1 GiB，mmap 页可由操作系统回收；这是分块处理，不是逐字节压缩流解码。
每 1 万条写一个 Zstd Parquet 分片；独立 SMILES、InChIKey、化学缓存和父结构支持数用 SQLite 统计。
没有 526 万记录或全库唯一 SMILES 的 Python set。每次 checkpoint 先原子重命名分片，再提交 SQLite。
2022 可只读复用已完成 2025 扫描的**完全相同 SMILES** 化学缓存，条件是化学代码和 RDKit 版本指纹一致。
仍逐条扫描 2022 元数据；其证据、父结构去重和频次独立统计，不复用 2025 的名称或支持数。
`shared_structure_cache_hits` 报告复用的不同 SMILES 数；不命中的结构正常执行 RDKit 图解析。
恢复会重新解压当前块，跳过已提交记录；不重复计数。临时 raw 文件在正常退出时清理。
峰值 RSS 以 100 ms 周期采样，包含原生解压及映射工作集；运行时间包含解压、解析、化学处理和分片写入，汇总耗时另计。
SQLite 有大量随机读写，建议 `data/private/_state/` 位于 SSD；最终 Parquet 可留在容量较大的机械盘。
若使用目录联接，必须在提取进程退出后复制完整状态（包含尚存的 WAL/SHM），校验数据库，再从原路径续跑。
保留源状态备份，不能在正在写库时直接移动 SQLite 主文件。

字段依据作者仓库固定提交 `8556036fb73d77e4d703eb4263defe025d604a5d`：

|版本|作者类|ID/Name/Ontology/SMILES/InChIKey/Class|跳过 Spectrum|
|---|---|---|---|
|2025|[MoleculeMsReference](https://github.com/systemsomicslab/MsdialWorkbench/blob/8556036fb73d77e4d703eb4263defe025d604a5d/src/Common/CommonStandard/Components/MoleculeMsReference.cs)|0/5/7/8/9/14|4|
|2022|[MspFormatCompoundInformationBean](https://github.com/systemsomicslab/MsdialWorkbench/blob/8556036fb73d77e4d703eb4263defe025d604a5d/src/Common/CommonStandard/Database/MspFormatCompoundInformationBean.cs)|0/3/17/7/8/2|11|

## 化学和计数约定

- 文本宽筛包含 lyso、ether、氧化及少见磷脂类别；SMILES 含 P 的记录也进入结构核验，防止缩写遗漏。
- SMARTS 识别线性三碳甘油及末端 O-phosphate；去除磷酸 O 的对称匹配。允许 CL 的多个互不重叠甘油骨架。
- 只切甘油 O 与链端 C 的键。A 包含 carbonyl C；O/P 不包含甘油侧的 O。只保留线性碳链及 A 的 carbonyl O。
- 环、支链、额外 O/N 等、三键、芳香、带电、同位素、自由基链明确拒绝。一个母结构中的正常链仍可保留，特殊链另报拒绝。
- 头基糖上的额外酰链、N-acyl 等不在甘油 O-C 拆链范围内，不从名称补齐。
- 从 attachment 端 C1 遍历，记录所有 C=C 位置，不保留 E/Z。P 必须有图中 C1=C2；不是依据名字分类。
- **所有 token 的 DB 数均为图中 C=C 总数，P 包含 vinyl bond**，故常见名称 `P-18:0` 的图 token 可能是 `P-18:1`。
  `p_db_count_includes_vinyl=true` 描述输出 token；`source_p_db_count_includes_vinyl` 描述名称与图比较所得的源命名口径。
  原名称 O- 但图为 vinyl ether 的记录标记 `SOURCE_LABEL_LINKAGE_CONFLICT`，不静默转成源库认可的 P 命名。
- `parent_structure_id` 对应 SQLite parents.id；父结构为非立体 canonical full SMILES，不以 adduct / ScanID / InChIKey 计频。
  同一母结构两条相同链产生两次 observation，但该 variant 和 token 的 parent 支持均只加一。
- token 的总支持数是所有 variant 的父结构**并集**。不同 variants 可在同一父结构共存，支持和可超过并集。

## 候选与版本核对

配置：`dominant_fraction=0.80`，`low_evidence_min_unique_parents=5`。
先判 LOW_EVIDENCE；证据足且只有一个位置版本为 UNIQUE；多版本时首位 ≥0.80 且次位 ≤0.20、首位严格领先才为 DOMINANT；其他为 CONFLICT。
同支持数用 canonical fragment 排序保证可复现，不代表生物学偏好。所有结果只是人工审核候选。
库内母结构支持数不是实测谱图次数或生物丰度，亦受原库结构枚举范围影响。
即使两个库版本都采用同一位置，也只能证明库约定一致，不能作为天然双键位置的独立验证。

MATCH 要求两个版本首位候选及完整 positional variant 集合均相同；否则 POSITION_CONFLICT。
缺失精确 token 时保留 ONLY_2025 / ONLY_2022；如果另一版本实际观察到同分子组成的相反 O/P linkage，则记 LINKAGE_CONFLICT，列出实际 counterpart。
O/P 等质量表只在两版本实际观察 token 的并集中找同组成对，formula 是带 attachment 的链残基组成（dummy 质量为 0），不自动创造或批准映射。
尚未完成两版本全扫时，交叉结果为临时结果。

## 审核包

`data/private/msdial_chain_review/` 内包含计划指定文件；另保留 2022 evidence/variants/candidates/metadata report 便于追溯。
`msdial2022_chain_crosscheck.parquet` 为 token 级交叉核对表，与 `chain_version_comparison.tsv` 同字段；
逐条 2022 链证据在 `msdial2022_chain_evidence.parquet` 分片数据集中。
PE-P 专表严格使用 PE / EtherPE / OxPE，排除 LPE / EtherLPE；后者仍保留在完整 evidence 和 variants 中。
PE 专表直接从已提交链证据重新聚合，父结构支持数仍在 SQLite 临时表去重，无需重新扫描原库。
`source_naming_conventions.tsv` 单独汇总原名称的 O/P 标记及 DB 数口径，保留名称示例。
版本表的 MATCH / LINKAGE_CONFLICT 描述分子图；源名称与图类型不符另列 source_label_conflicts 字段，强制 review_required=true，避免图一致时漏掉命名问题。
大表的 `.parquet` 路径是**分片数据集目录**，不是单一文件：

```python
import pyarrow.dataset as ds
evidence = ds.dataset('data/private/msdial_chain_review/msdial2025_chain_evidence.parquet', format='parquet')
print(evidence.schema)
print(evidence.count_rows())
# 用 scanner().to_batches() 分批读取，避免 to_table() 一次展开全表。
```

rejections 数据集以 2025/2022 子目录组织，保留 source_version 字段。
metadata report 的 `parse_failures` 指容器/字段错误（致命失败，不静默跳过）；候选结构解析失败另列 `candidate_invalid_smiles`。
每条 observation 均有源名称、记录索引和 InChIKey，可以回到完整 metadata；SQLite 不属于最终批准链表。

验证：`python -m pytest -q --confcutdir=tests/chain_extraction tests/chain_extraction`。
提取 CI 独立安装依赖，不安装 Torch、不触发训练工作流。
