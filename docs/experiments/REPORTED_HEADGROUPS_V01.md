# 已报道 Polar 参考库 v0.1：收集与复现

本轮只收集真实来源结构、规范化、名称映射和去重。完整数据位于
`outputs/reported_headgroups_v01_20261020/`，受Git忽略，原始下载、人工文献转录和全部CSV均保留。
不修改现有模型、Tokenizer、Builder、195,306种冻结Polar或数据划分；不执行训练或谱图任务。

## 来源及处理边界

- LIPID MAPS：保存2026-10-20全LMSD、GP/SP REST表及extended SDF；全表50,586条，
  含P SMILES条目11,453条。没有具体结构的分类占位行不当作分子收录。
- LSG：固定commit `519d004ac2c182090b845d22863faf592baa6b52`。
  用AST读取所有类别的局部SMILES、tooltip、动态表达式和引用候选，不执行其谱图生成代码。
  `GenerateLipids.Glycerolipid`实际把`sn3.smiles`直接放在甘油O之前，按此开放端接`[*:1]`。
  动态依赖额外链而没有真实结构的表达式进入待审，不人为指定链来补齐。
- MS-DIAL：固定commit `349d7ef6edbd09e34c0eb38b849bd9fd8af79e7e`全部含P结构Header，
  加官方LipidBlast VS12固定头基工作表的前三个已存储结构示例。
  原工作簿只读XML，不执行宏；理论库结构不当作实验谱图。
  本轮未遍历所有MSP发布版，不能据此宣称MS-DIAL所有结构均已覆盖。
- PubChem：仅核对LMSD已链接CID及少量已报道类别的名称，保存139个链接CID属性结果。
  未启动弱证据候选的泛检索或理论结构扩充。名称命中不等于结构身份正确。
- 文献：私有`raw/literature_curation.csv`逐项记录DOI/PMID、原结构出处和图/段落位置。
  人工转录图已保留原始下载。未核查的源码引用不会自动变成“文献已报道”。

## 身份、连接和校验

按当前模型的非手性canonical graph定义结构身份，`HG-`后接SHA256前20位。
全图中只切明确的链承载甘油O-P键，保留该P索引；鞘脂使用独立保守骨架匹配。
LSG按源码开放端识别连接P，不能取canonical SMILES中的第一个P。
禁止把短磷酸丝氨酸头基错认成鞘氨醇骨架；有相应回归测试。

只作可逆质子化/非手性规范化，保留原始立体SMILES，不改重原子连接、不删除特殊脂质臂。
显式同位素标记的原始来源进入待审；当前表示不能保留时不静默去标，也不冒报Builder兼容。
按完整残余图去重，不按分子式或基础核心筛选。CL/BMP等残余图可能保留另一脂质臂，
必须连同原始结构保留，不能用裁剪来绕过现有Builder。

调用现有`validate_polar_fragment`、`assemble`，并以甘油、OH封端片段及脱水数独立检查含H元素守恒。
来源级分别报告完整分子式与完整图往返。代表性sn排序不能复现所有来源的位置异构体，
公式通过不会冒称完整图通过。Builder通过也不等于Decoder长度/词表可直接使用。

## 本机命令

```powershell
. .\scripts\enter_cuda_v04.ps1
# 下载始终使用新输出根或复用已哈希的缓存；不会替换已存在文件。
python scripts/fetch_reported_headgroup_sources_v01.py
# LSG克隆保留在raw/LSG，版本需固定到上述commit；文献转录表和PubChem响应随私有数据包提供。
python scripts/prepare_reported_headgroup_inputs.py
python scripts/collect_reported_headgroups_v01.py
python scripts/audit_reported_headgroups_v01.py
python scripts/query_reported_headgroups_v01.py --name PPA --compact
python scripts/query_reported_headgroups_v01.py --id <实际Polar-ID>
```

Excel通过`scripts/export_reported_headgroups_v01.mjs`从审计生成的
`excel_payload_private.json`创建，使用桌面应用提供的Artifact Tool和独立authoring/node_modules junction。
原始CSV是完整机器数据。Excel保留全部主表、名称、来源和待审行，只选择人工检查需要的列。
名称输入不区分大小写；匹配序号可切换同名不同图，绝不按目标相似度挑选结构。
使用已验证的VLOOKUP精确匹配，实际检查PC结构及PPA两图切换，不只检查公式文本。

## 用户文献库追加

输入支持CSV或JSONL，必需字段：

```text
full_name,original_smiles,source,source_id_or_doi,source_url,structure_input_kind,source_kind
```

可选`abbreviation`、分号分隔或JSON列表`aliases`、`msms_status`、`literature_verified`、`evidence_locator`。
`structure_input_kind`仅接受`full_lipid`或带明确连接标记的`standard_polar`。
不能从用户给出的普通多P小分子酸中自行猜连接位点。
`literature_verified`默认为false，只有明确审核后才设true。

```powershell
python scripts/import_reported_headgroups_v01.py --input <文献库.csv或.jsonl> --output-root outputs/reported_headgroups_v02
```

工具复制来源包到新目录、追加导入台账后重建，旧目录与旧来源不覆盖。
重复图合并，新增名称和来源继续保留；不合法或不能确定边界的条目进入待审。
之后对新目录运行审计和Excel导出。源码公开不代表私有人工转录数据已经上传。

## 发布

GitHub只含收集/校验/查询/导入代码、回归测试、方法和聚合统计。
完整结构库、下载、原始预测、checkpoint、人工转录、Excel和逐条来源映射留在G盘。
实际结果见[聚合报告](../results/REPORTED_HEADGROUPS_V01_SUMMARY.md)。
