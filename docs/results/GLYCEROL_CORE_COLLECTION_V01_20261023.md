# 甘油＋头基核心参考库 v0.1（2026-10-23）

## 实际交付

本轮是独立 CPU 数据整理，不启动新训练。完整结构、原始下载、人工决定、图册及 Excel 均保留在 G 盘；GitHub 只含代码、配置和聚合统计。

| 项目 | 数量 |
|---|---:|
| 唯一无立体连接图 | 352 |
| 结构检查通过（尚非人工定稿） | 130 |
| 待判断 | 6 |
| 排除候选（保留原图与来源） | 216 |
| 可追溯来源记录 | 25340 |
| 有原始天然核心文献依据的连接图 | 22 |
| 有实际酰基链完整数据库结构的核心 | 62 |
| 非磷酸核心图／结构通过 | 25 / 21 |
| 天然连接规则扩展涉及的图（可与已知核心重叠） | 9 |
| 原人工保留图归并后／本轮结构通过 | 134 / 101 |

已检查 LMSD 的 GP/GL 18423 条和 PubChem 15820 个 CID；同时解析冻结 LSG、MS-DIAL 规则并保留旧参考库。未切出来源不能直接当作“未发现头基”：大部分是无头基 TAG/DAG、额外脂质锚点或不支持边界；所有原因另存。

这是首批可复现收集，并未穷尽 PubChem 或全部天然脂质。数据库结构记录、规则定义、原始天然脂质核心证据分别标记；不因 PubChem 收录或名称含“lipid”而认定天然。原有人工排除 3 个、待判断 2 个仍独立保留。

## 结构与质量规则

- 每个核心仅一个稳定 ID。忽略立体化学作为模板去重条件，不枚举立体异构体；来源原始 SMILES 保留，便于追溯。
- 不同连接图分别收录。肌醇的编号依赖问题例外：PIP₂(3,4) 与 PIP₂(4,5) 在去立体信息后图同构，保留原始位置名称，将该图列为位点待核实，未把它作为确定单一位点模板定稿。
- 显示优先使用经来源键确定的甘油连接 P 的 `O=P...`；多 P 时用明确边界定位。源图只有 P=S 等特殊键时保留真实图并标记，不改成 P=O。
- 质量使用去除脂肪链、甘油链连接 O 以 H 封端后的核心平均分子量。糖基或肌醇 ≤600 Da，其余 ≤400 Da；另保存精确质量。不可中性化的带净电荷核心待判断。
- 来源全脂质根据真实甘油–O–链键切分，验证连接图和完整分子式（包括 H）往返。短/长脂肪酰链锚点与氨基酸酯分别处理，不能把 Ala/Lys-PG 当作第二个脂质锚点。
- 额外长酰基/烷基、CL/BMP 等多锚点不进入默认候选；卤素待天然证据核实，Boc/Cbz/苄基保护等明显合成特征排除候选。完整原始记录仍保存。
- 新糖基候选仅在源糖的明确环状异头 OH 上使用已有糖苷–甘油连接规则；不猜测糖环化或新位点。它们显式为候选，不宣称已报道天然脂质。

天然非磷酸核心依据包括 [DGTS](https://pubmed.ncbi.nlm.nih.gov/10897481/)、[DGDG](https://pubmed.ncbi.nlm.nih.gov/9363436/)、[GlcADG](https://pubmed.ncbi.nlm.nih.gov/24558041/)、[SQDG](https://pubmed.ncbi.nlm.nih.gov/28264430/)、[DGTA](https://pubmed.ncbi.nlm.nih.gov/9309879/)、[DGCC](https://doi.org/10.1016/0031-9422(94)85041-0) 和 [3-SGDG](https://pubmed.ncbi.nlm.nih.gov/36266352/)。这些证据针对相应核心／类别；来源库的每一种酰基链组合没有因此被升级成已核实天然物。3-SGDG 文献中的合成 SGlcDG 对照也不会被标成天然。

## 校验结果

完整数据库脂质图连接／分子式往返 24899 条。其中 24038 条还满足原始立体图往返；本轮不以立体异构体分拆模板。数据库声明分子式与其 SMILES 不一致 18 条，独立列出，不修补源图、不用作逐个天然链组合的证据。

13 项新化学边界测试通过。离线 HTML 验证默认136项、全部352项、搜索、审核状态、撤销、备份、UTF-8 BOM 与真正换行的 CSV 导出；352组 SVG/PNG 均存在，12页 PNG 图册及2张工作表已目视检查。Excel 摘要由 COUNTIFS 驱动，临时改动及恢复检查通过，没有公式错误。该 HTML 测试使用真实脚本和最小 DOM 环境，未代替完整浏览器兼容性测试。

## 原 195,306 库的独立注释

原 Parquet SHA256 校验前后相同。新注释 195306 条，结构规则通过 66252 条；没有将这个筛选写回冻结库、标签、划分或正在使用的模板池。

```json
{
  "core_mass_above_limit": 111737,
  "halogen_hold_for_natural_evidence": 47525,
  "extra_head_long_alkyl_or_acyl_chain": 14927,
  "synthetic_protecting_group_suspected": 9387
}
```

原库来源类型、原有“完整脂质/程序组装”标记和来源 CID 保留在独立表；“匹配到已知天然核心”与“该完整脂质链组合已被证实天然”是不同字段。大量规则或数据库碎片没有天然实证，保持未核实。

## 文件与复现

私有根目录：`G:/yzx/linzhiyuce/phospholipid-structure-pretrain/outputs/glycerol_core_collection_v01_20261023`。

核心：`glycerol_core_catalog.parquet/.csv`、`core_name_alias_index.parquet`、`core_sources.jsonl.gz`；人工查看：`index.html`、`svg/`、`png/`、`atlas/`、`glycerol_core_review_v01.xlsx`；不确定来源：`sources_pending_review.csv`、`database_formula_mismatches.csv`、`numbered_position_review_private.json`；全库注释：`frozen_195306_future_core_annotations.parquet`。

```powershell
. scripts/enter_cuda_v04.ps1
python scripts/fetch_glycerol_core_sources_v01.py
python scripts/collect_glycerol_cores_v01.py
python scripts/render_glycerol_core_review_v01.py
# 工作簿通过 bundled Node + @oai/artifact-tool 导出；不使用 openpyxl/xlsxwriter 写入。
python scripts/query_glycerol_core_v01.py DGTS
```

后续模板数量仍以既有5,000主规模和500/2,000/5,000/10,000/20,000对照作为实验设计起点。这里的通过图不是全部训练模板池，也未证明5,000对扩展库最优。先补齐来源与位点审核，再冻结新的甘油＋头基目标范围，重算覆盖率和官能团／分子式覆盖后决定；不自动选库或重训。真实谱图模板与结构预训练模板仍分开。

## 训练边界与状态观察

V2 的 73 个受保护源代码／配置哈希均未改变。集合工作仅 CPU，不更新模型参数或新标签。

检查时 E2 的原训练和监督进程已不存在，最后进度为2026-10-23 14:21:57、step36,300、epoch7；旧状态文件仍显示 running，不能视为仍在训练或正常完成。本次有界系统事件查询未见新增4101或系统重启，退出原因尚不明确，WATCHDOG目录仍不可读。本轮没有自动重启，也未改动旧 checkpoint／停止锁。
