# PubChem 极性片段审核流水线 v0.1

本轮只冻结 MS-DIAL 2025 链表、检索真实 PubChem 含磷酸/膦酸结构并输出审核候选。
入口不导入 Torch，不训练，不枚举完整脂质，也不生成批准版 `polar_moieties_pubchem_v0.1.tsv`。

## 运行

```bash
python -m pip install -r requirements-extraction.txt
python -m pip install --no-deps -e .
python scripts/harvest_pubchem_polar.py --cache-dir /path/on/ssd/pubchem-polar-v01
```

Windows 默认缓存位于 `%LOCALAPPDATA%/pubchem-polar-v01`（SSD），输出仍在项目的
`data/private/pubchem_polar_review/`。在 WSL 中可显式指定 Linux SSD 路径。
本流程只需 CPU/RDKit，无 ROCm、显卡或 MS-DIAL 2022 依赖。

默认顺序运行 `stage1`（每查询 1,000）、`stage2`（每查询 10,000）、`full`。
可用 `--stages stage1` 单独检查第一阶段，再用 `--stages stage2 full` 继续。
更换电脑时复制缓存可减少重复下载；要重新获取 PubChem 快照，请使用新的缓存目录。
缓存混合的是逐请求获取时间，并非 PubChem 数据库的同一时刻冻结快照。

重复命令可恢复：CID 查询和 HTTP 响应按完整 URL 哈希缓存，属性按 CID 存入 SQLite/WAL，
已下载的 CID 不因查询重叠而重复下载。化学审核输出从缓存重新计算，避免从不完整 Parquet
续写。`checkpoint.json` 记录进度；网络错误 CID 单独列在报告中，不冒充化学拒绝。

## 链表冻结

只读取现有 `msdial2025_chain_variants.parquet` 与 2025 representative candidates。
两个表均确认仅一个位置版本的 token 才可导出；不重新扫描 LBM2。
严格核验 A=177、O=84、P=61，共 322，且排除集恰好为
`A-24:1, A-26:1, A-28:1, A-30:1, A-32:1, A-34:1`。

输出 `data/private/msdial_chain_review/msdial_derived_chains_v0.1.tsv`，通过现有 builder
的全部 322 条链校验，并保存输入/输出 SHA-256。沿用所有现有双键位置，包括 A-20:4 的
`[8,11,14,17]`。P token 继续按分子图统计包括 vinyl 在内的全部 C=C，显式标记约定。
普通后续 phosphatidyl-X 仅用 A/O，P 仍只用于 PE-P 分支；本入口不做这些组合。

## 查询与完整性

所有检索 SMARTS、正负对照 ID、阈值都在 `configs/pubchem_polar_queries.yaml`。
使用 PUG REST `compound/fastsubstructure/smarts/cids/JSON`，三个查询分别覆盖四氧配位
phosphate acid、一 P-C 键 phosphonate acid、带 P-O-P 的 polyphosphate acid。
查询之间可重叠，报告保留交集；片段分类优先 polyphosphate，其次 phosphonate，再 phosphate。

先验证本地结构与 SMARTS，再通过 PubChem 名称属性接口验证真实 CID；正对照包括 PA/PE/PC/
PS/PG/PI、aminoethylphosphonate、phosphonoacetate 和 pyrophosphate。
负对照包括 TPPO、triphenyl phosphate、trimethyl phosphate、P=S；金属负对照是本地构造的
含铜多组分记录，不虚构其 PubChem CID。任一对照失败则不启动大查询。

默认全局串行 3 requests/s（可配置，但拒绝 >4）；属性每批 500 CID；少于 151 个未缓存 CID
用短 GET，更大批次使用标准 PUG REST POST 表单，避免过长 URL 被服务端拒绝。429/503 等暂时错误
指数退避，重试同样限速；遵守数字 `Retry-After`，记录请求/重试数。
只有一个网络工作线程，预取下一批属性，与主线程的 RDKit 计算重叠；不并发发送 HTTP。

首两阶段明确标记 `SAMPLED_PARTIAL`，不是随机样本，不能用其保留率推断全库保留率。
full 的 `MaxRecords` 为 1,000,000：只有响应少于上限才按该 API 查询认定完整。
若饱和或失败，按配置把 matched P 的其余 O 配体分成 degree-one / non-degree-one 子查询，
取 CID 并集。划分对父查询在该 P 上穷尽；多 P 分子仍可能跨子查询重复。
若任何子查询仍饱和/失败，保留 `QUERY_TRUNCATED` / `QUERY_FAILED` 和未完成状态。
此版本只实现配置中的一层穷尽划分，仍饱和的叶节点需要继续扩展配置；绝不宣称已全量。
完整性指配置 SMARTS 的结果，不能等同于 PubChem 中所有可能的含磷化学表示或真实化学空间。

## 组分、化学边界和连接

只允许一个含 P 的 connected component。多个含 P 组分直接拒绝。
仅删除明确的简单离子：Li/Na/K、Mg/Ca、NH4 和卤素离子；Cu 等其他金属拒绝。
乙醇、胺类等有机共组分不能随意删除。保存原始结构、选中组分和移除离子。
拒绝不支持元素、同位素、自由基、解析/标准化/价态失败。

组分内**每个 P** 都必须具有 P=O，且单键配体为三个 O，或两个 O 加一个 C。
因此排除 phosphinate、P=S、P-N、P-H、phosphine oxide 等；其他部位的 N/S、糖、环、
杂环和芳环可保留。只将 terminal P-OH / P-O− 的 O 原位替换成 degree-one `[*:1]`。
这个 dummy 单键直接接 P，表示未来 glycerol-O 的连接位置；不切 P-O-C 或 P-C。
遍历所有酸性氧，记录 atom index、所有候选和去重前数量，不靠已知头基白名单筛选。

MW≤600、C≤30、heavy atoms≤60、P≤3、最长非环非芳香连续碳路径≤10。
这些是**选中酸前体组分**（去简单离子并做可逆电荷标准化）的软阈值，超限保留为
`FILTER_REVIEW`，不能默认进入批准库。最长碳路径包含非芳香 C=C，不把环骨架当脂肪尾。

## 电荷、去重与状态

用 RDKit MolStandardize Cleanup/Uncharger 中和可逆质子化，保留季铵等永久电荷。
将残余磷酸/膦酸、羧酸和磺酸的 terminal 酸性氧转成统一酸形式，再枚举能平衡永久
正电的去质子化位置。等价结果合并；不等价的电荷位置全部记录在候选
`charge_alternatives`，以字典序最小键作稳定审核代表，并标 `CHARGE_AMBIGUOUS`。
超过九个可枚举酸性氧时不尝试组合爆炸，保留审核状态。PC 的两个原始酸性氧替换途径
应归一为相同的 N+/O− 片段；PE 的可逆 NH3+/O− 与中性表示应归一。

主键为 non-isomeric standardized fragment SMILES，忽略手性/E/Z，保留连接差异。
`polar_id` 是主键 SHA-256 前 16 位。合并全部 source CIDs 和命中查询。
审核 TSV 中的分子式/MW是带开放连接位点的 **fragment 本身**（dummy 不计元素/质量），
不是酸前体，更不是组装后的完整脂质；原始 PubChem 公式和 MW 在 raw 表。

状态优先级为 `CHARGE_AMBIGUOUS > FILTER_REVIEW > MULTIPLE_ATTACHMENT_AMBIGUITY > APPROVABLE`。
候选表保留电荷、阈值和多位点证据；fragment 表保留 `review_flags` 和 `filter_flags`，
多重问题不会因为一个主状态被丢失。`attachment_site_count_before_dedup` 在 fragment 表
是所有 CID 原始连接位点对该 fragment 的累计贡献数；在 candidate 表是该 CID 的总位点数。
`APPROVABLE` 仅表示自动检查可进入审核，**不表示已获人工批准**。

报告分别提供位点候选→片段和 `(CID, distinct fragment)`→片段的去重率。
第二个分母避免多位点分子一个 CID 产生多个片段时出现误导的负去重率。
默认 `allow_aa/allow_oa=true, allow_pa=false`；只有与 PE control 完全相同的片段可设 P/A。

现有 builder 的 `polar_family` 枚举和净电荷要求不适用于全部审核候选。
本轮不修改 builder、不自动输出 approved 表。以后导出时需把 PE 精确映射到 `PE`，
其他 phosphate/polyphosphate 映射到 `pubchem_expanded` 等现有合法值，同时保留化学 family
列或来源记录，并重新执行 builder 校验。净电荷非零/歧义和未批准的多连接结果不得直接导出。

## 输出与测试

审核目录包含 query summary、raw、P components、attachment candidates、fragments、rejections、
review.tsv 和 report.json；前两阶段在 `stage1/`、`stage2/`，full 在目录根部。
报告列出 20 个有真实 PubChem 来源、在已知对照之外的代表片段；“扩展”不声称文献新颖性。
原始库、缓存及本地大表不上传 GitHub；代码、测试、方法说明与汇总结果可复现。

```bash
python -m pytest -q --confcutdir=tests/pubchem_polar tests/pubchem_polar
python -m pytest -q --confcutdir=tests/library_builder tests/library_builder
python -m ruff check .
python -m ruff format --check .
```

独立 CI 只运行离线化学/缓存/限速/退避/截断/Parquet 测试和 builder 回归，不执行网络抓取和训练。

实际输出完成后可用 `python scripts/audit_pubchem_polar.py` 离线核验。它逐条检查 fragment
的 P/dummy 边界、source CID 外键、候选/拒绝对 raw 的完整划分、跨表聚合数和查询来源，
输出 `verification_audit.json`；阶段 1/2 可将对应目录作为参数。

API 参考：<https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest>。
