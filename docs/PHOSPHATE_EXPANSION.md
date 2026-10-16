# Phosphate 扩充与 family-aware dry-run

本轮按用户补充要求，将新版完整库可配置上限设为 **2,000,000**，同时保留 1,500,000 预算对照。
只运行极性片段扩充、来源核验和链组合预算模拟，不执行最终全库组装、不训练。
v0.1 builder 的 1,500,000 上限不变；只有明确的 `structure-library-v0.2` 可用 2,000,000。

本轮实测计数、来源核验和 P 链覆盖缺口见
[PHOSPHATE_EXPANSION_RESULTS.md](PHOSPHATE_EXPANSION_RESULTS.md)。

## 冻结 baseline

`data/private/pubchem_polar_plausibility_v0.1/polar_moieties_pubchem_v0.1.tsv` 保持原样。
输入必须匹配 SHA-256 `8b675f2b3d6a5d19dbc6801c3cf68a5782d153a5c9107fc2def2572177bbea8b`，
以及 phosphate 39,591、phosphonate 62,867、polyphosphate 939。
扩充输出另存 `data/private/polar_expansion_v0.2/polar_moieties_expanded_v0.2.tsv`。

```bash
python scripts/expand_phosphate_polars.py
```

输入仅使用既有 PubChem 快照，没有继续下载。全审核表先输出 family × status 交叉表。

## Multi-attachment 拆分

仅选择全审查表 `MULTIPLE_ATTACHMENT_AMBIGUITY` 条目及其真实 source CID。
逐 CID 用原来的 PubChem 化学流程重新枚举所有酸性 P–OH/P–O− 位点，再匹配目标片段。
不救回 CHARGE_AMBIGUOUS 或 FILTER_REVIEW 片段。
源候选若仍有 charge/filter 标记，不通过该来源将其升为可用。

同一来源的等价位点先按去立体 canonical SMILES 合并，多个 source CID 再按同一图聚合。
每个不同图重新通过 sanitize、现有 reactive SMARTS、CORE/EXTENDED 阈值和 builder 边界。
不以传统头基名称、文献是否报道或分子相似度决定保留。

`structure_origin=pubchem_multi_attachment_split`。
`source_attachment_provenance` 每项同时保存 CID、原始 raw SMILES 经 RDKit 默认解析后的
零起始原子索引和所选 P 组分的局部索引；避免去掉盐组分后索引偏移却无法追溯。
原始 raw SMILES 保存在已有快照，TSV 还保留代表 CID 的 `original_smiles`。

## 受控 phosphate analog

只有合并救回片段后 phosphate 仍少于 phosphonate 才启动。
对所有当前 CORE/EXTENDED phosphonate 检查：只有一个 P、重建酸前体的局部 P 化学合规、
恰好一个 P–C 单键且该键不在环中。多 P 和 P–O–P 不转换。
将该边替换为 P–O–C，只增加一个 O，不改其余连接和原子。

每个图比较全部非 H 元素计数，必须仅 O+1；H 由 RDKit 重新计算并记录差值。
新图再次经过相同 reactive、plausibility 和 builder 规则。
转换前合规不保证转换后可用，例如新出现的 P–O–glycerol 路径可能触发 builder 边界；
这类候选进入 HOLD，记录理由，不视为程序成功保留。

新图与 baseline、rescued phosphate 及已经生成的 analog 共同去重。
已有图保留原 ID 和原始来源，可以增加 `also_generated_as_phosphate_analog` 及 parent 列。
全新图使用内容哈希 ID，`structure_origin=generated_phosphate_analog`，
`source_cids=[]`、`source_count=0`；真实父分子的 CID 只进入 `parent_source_cids`。
这不会把父分子证据冒充为新图的数据库记录。来源字段不用于训练标签、loss 或采样权重。
不设生成数量目标，不在数量平衡后提前停止，不删除 phosphonate。

## Family-aware 链预算

```bash
python scripts/dry_run_expanded_library.py
```

默认对 1,500,000 和 2,000,000 各模拟一次。脚本没有 `--execute` 参数。
先对每个头基安排三个共同 A/A anchors，再预留既有 nitrile 与 keto-inositol 的必选代表组合。
其余四个原 anchors 仍在合法链组合池中，不再先为所有头基强制预留全部七个。
这样给链多样性和 family-aware extras 留出预算。

默认额外背景权重为 phosphate 2、phosphonate 1、polyphosphate 1。
先以 `头基数 × family weight` 做有上限的比例分配，再用最大余数法分配整数名额；
family 内使用 seed 派生的固定顺序分配剩余名额，与输入 CID/行顺序无关。
每个头基仍受 max-extra 上限约束。所有权重必须为有限正数。

2:1 指每个头基的**额外**背景，不是含 mandatory anchors 的总背景之比。
例如 baseline 在 150 万预算下，如果先统一安排七个 anchors，2:1 extras 仍可能使
phosphonate 总结构多于 phosphate；三个共同 anchors 的方案更适合当前目标。
不硬编码 phosphate 百分比，报告实际整数预算结果。

lyso 名额按额外预算比例分配，默认目标为总结构的 10%。其余使用既有 AA/OA/PE-PA
及碳链长度/不饱和度分层生成器。仅 exact PE 开放 P/A，P-lyso 默认关闭。
每个头基跳过 mandatory 和已经选中的 chain ID，避免复制相同完整结构进行 oversampling。
采样只依赖 chemistry_family、合法组合、seed；不依赖 structure_origin 或 evidence_source。

dry-run 逐个枚举将使用的 head/chain 配对，输出 family、AA/OA/PA/lyso 计数、
每头基整数名额、特殊头基覆盖和 schedule SHA-256，不生成完整分子图。
这些配对数是预算模拟的精确计数；正式组装的 sanitize 拒绝与整图 canonical 去重可能减少实际条数。

## 未来正式入口

原 v0.2 uniform 配置保留；新配置是 `configs/library_build_expanded_v02.yaml`。
未来使用新表时，以下命令仍只做预检查：

```bash
python scripts/build_structure_library_v02.py \
  --config configs/library_build_expanded_v02.yaml \
  --polar data/private/polar_expansion_v0.2/polar_moieties_expanded_v0.2.tsv \
  --plausibility-report data/private/polar_expansion_v0.2/polar_expansion_report.json
```

本轮不加 `--execute`。正式 builder 已复用相同 weighted quota 和 chain schedule，
继续对生成的完整图做 canonical 去重；来源、parent CID 和 attachment provenance 进入输出字段。
已有特殊头基仍走 must-include 逻辑，新增 analog 不改变链表和双键位置。

## 测试

```bash
for suite in library_builder pubchem_polar chain_extraction polar_plausibility library_v02 polar_expansion; do
  python -m pytest -q --confcutdir="tests/$suite" "tests/$suite"
done
```

新增测试包括不等价/等价位点、盐组分索引、反应性和 builder 边界、单 O 插入、
元素 delta、环 P–C/多 P 禁止、已有图去重、来源不冒认、baseline 保留、
family quota 的 150 万/200 万边界、固定 seed、不复制完整图和微型正式组装与 dry-run 一致。
