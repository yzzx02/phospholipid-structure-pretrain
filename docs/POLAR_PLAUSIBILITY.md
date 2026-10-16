# PubChem 第二层筛选与 v0.2 组装入口

本阶段离线处理已有 APPROVABLE 快照，生成正式 CORE + EXTENDED 输入，并定向核验少量特殊结构。
不等待缺失的 PubChem CID、不扩大原检索范围、不训练、不运行最终完整结构库。
现有 PubChem 下载代码和 v0.1 builder 文件保持原样；v0.2 采用独立入口。

## 运行第二层筛选

```bash
python scripts/filter_pubchem_polar.py
```

输入 `data/private/pubchem_polar_review/`，输出
`data/private/pubchem_polar_plausibility_v0.1/`。
反应性 SMARTS 全部位于 `configs/polar_reactive_rules.yaml`，大小与极性阈值在
`configs/polar_plausibility.yaml`。只处理 `status == APPROVABLE`，其他状态保留原审核包。
四条反应性规则分别检查 O–halogen、acyl halide、sulfonyl halide、O–O；P–O–P 没有 O–O 键。
腈、C#C、芳环、杂环、C–halogen 和 S 均不单独触发拒绝；不使用头基相似度、标题或数据库脂质分类。

描述符针对开放 P 边界片段；dummy 不计质量与重原子。最长脂肪碳路径沿用原 PubChem 定义：
非环、非芳香碳组成的最长路径，包含羰基碳和不饱和碳。它不是疏水性或稳定性的实验测量。
规则通过只说明适合当前枚举空间，不证明天然存在、合成可行性或生理稳定性。

发现了一项原计划未显式区分的兼容性情况：现有 builder 拒绝带完整酰基甘油磷脂子结构的
polar fragment。化学阈值通过、无反应性标记而触发该限制的条目记录为
`HOLD / BUILDER_BOUNDARY_REVIEW:intact_lipid_as_polar`，不归为 reactive REJECT，原结构保留。
报告单列 `builder_boundary_hold_count`。未放宽或修改原 builder 的边界规则。

正式 TSV 保留 builder 要求的十列，并增加 `chemistry_family`、`polar_plausibility_tier`、
`source_cids`、`source_count`。原始分子 SMILES 从真实 source CID 的 raw 表连接获得。
旧 builder 的 `polar_family` 使用其既有枚举；原 phosphate/phosphonate/polyphosphate 分类
保留在 `chemistry_family`，不会丢失。只有 exact PE 片段开放 P/A。
输出逐条经过未修改的 `import_polars` 读取，并记录 SHA-256。

同时输出三组阈值统计、全部四层 Parquet、scaffold 统计、分层 spot-check 和代表例子。
conservative/default/broad 只改变 core 的 MW/C 上限，extended 边界相同，因此正式可用总数
可以完全不变。Scaffold 仅作统计，不参与删减或训练权重。

## 特殊结构核验

```bash
python scripts/audit_special_cases_v02.py \
  --approved-polar data/private/pubchem_polar_plausibility_v0.1/polar_moieties_pubchem_v0.1.tsv
```

独立搜索两个精确 motif，扫描 fragments Parquet、review TSV 和 raw Parquet；
先检查是否含三键，再对候选做 RDKit 子结构匹配，不以字符串命中作为化学判定。
三张表的搜索范围及命中数明确记录，不能外推成 PubChem 全库不存在某结构。
默认从每类最小的 approved 真实片段选一个，组装三种指定链背景，共至多六个腈基代表结构。

PI 从标准头基逐一将五个非磷酸连接羟基改为羰基。去掉立体信息后有三种不同连接图，
分别位于距磷酸连接环碳 1/2/3 条环键处；这个图距离不是生化或 IUPAC 位号。
逐项检查 ring/phosphate bond 不变，头基及整分子均只减少 H2。
输出标准 PI 与三个 mono-keto 头基，每个使用两组链背景，共八个代表结构。

`configs/special_case_evidence_v02.tsv` 记录有限范围的网页/文献定向核验，
仅提供来源、检索分类和说明，不设 LEVEL、不参与 loss/采样/标签。
搜索索引可能不完整，部分数据库/原文存在访问限制；“未核实 exact graph”不等于不存在。

cyanoethyl 合成保护基、中间体与最终脂质分开；早期 phosphatidylinosose 的类别实验线索
也不自动证明任一指定链对或氧化位置。生成的完整 SMILES 全部明确标记程序组装。
真实头基 CID 不能直接充当完整脂质证据。
`is_full_lipid_reported=false` 的精确定义是：本轮未将该 exact full graph 与明确来源独立匹配。

## v0.2 默认只检查输入和预算

```bash
python scripts/build_structure_library_v02.py
```

检查正式 TSV/完成报告的哈希及行数，要求 CORE + EXTENDED；缺失输入即失败。
要求冻结链表 A177/O84/P61，链原位置不变。特殊头基先经过相同化学规则；真实 PubChem
头基必须已在 approved 表中，只有计划允许的程序生成 mono-keto 变体可作为可追溯补充。

常规使用七个共享 A/A anchors；当预算占用达到 90% 时，每个头基先保留三个共同 anchors，
再以固定 seed 的头基顺序和轮转给其余四个 anchors 分配剩余预算的 35%，其余预算留给多样链。
必选特殊代表组合在随机采样之前预留；连三个共享 anchors 都放不下时明确失败，不删头基。
比例均可配置，不使用证据来源调整采样或损失。

未来明确运行正式组装时，入口需加 `--execute`；**本轮不运行该选项**。
组装复用既有分子图连接、规范化和去重逻辑；SQLite 去重临时库默认在系统临时盘，
产物写入 `data/structure_library_v0.2/`，包含完整库、链、头基、链组合、特殊结构、
拒绝表、报告与输入哈希配置。总数硬限 1,500,000。

## 离线测试

```bash
for suite in library_builder pubchem_polar chain_extraction polar_plausibility library_v02; do
  python -m pytest -q --confcutdir="tests/$suite" "tests/$suite"
done
```

v0.2 整体组装测试只使用明确标记的微型 fixture，验证预算、规则、去重与来源字段，
不使用私有大表，也不训练。完整库规模下的速度和磁盘用量尚未实测。
每个化学测试目录分别设置 confcutdir，避免加载依赖 PyTorch 的模型总测试配置。
