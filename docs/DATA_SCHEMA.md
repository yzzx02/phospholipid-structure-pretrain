# 数据接口 v0.1

真实数据构建尚未实施。本接口接收调用方提供的结构和监督标签，不下载数据库，不从 SMILES 计算分子式，不切割真实磷脂。

## 记录输入

`StructureRecord(structure_id, smiles, neutral_formula)`：

- `structure_id`：唯一字符串；检索库不允许重复 ID。
- `smiles`：完整、单组分、总电荷为 0 的结构；允许中性两性离子。不接收 adduct 片段或 wildcard。
- `neutral_formula`：10 个非负整数，元素顺序 `[C,H,N,O,P,S,F,Cl,Br,I]`。**模板分子式必需**；query 的记录可为 `None`。

`TargetRecord(structure, architecture, chain_tokens, headgroup_smiles, formula_available)`：

- `architecture`：`0=mono_phosphatidyl`，`1=bis_phosphatidyl`。
- `chain_tokens`：1–4 个外部标注的 A/O/P token，可重复；adapter canonical sort 后 NULL pad 到 4。不代表 sn 位点，也不预测 bis 架构内的磷脂单元配对。
- `headgroup_smiles`：外部切割得到的 headgroup fragment，例如 `[*:1]OCCN`。切割算法和 attachment 语义不在本阶段确定。
- `formula_available`：query 分子式可用时为 True；True 时 `structure.neutral_formula` 必须存在。

结构可以带有输入库给定的双键位置；模型不输出它的位置，不校正或替换该位置。未来采用 MS-DIAL 代表结构规则时，由外部数据构建阶段统一输入结构版本。

## 检索接口

`StructuralRetrieval.retrieve(target: StructureRecord) -> list[TemplateHit]`，每个 hit 只有 `template` 和 `structure_similarity`。默认实现使用 Morgan radius=2、2048 bits、无 chirality、Tanimoto；最多 10 个、阈值 0.4。排除同 ID 和 canonical 等价结构；库内部重复结构只保留确定性的一个条目。canonical identity 忽略 atom mapping、立体化学，但保留二维键连接关系；当前不做 tautomer/salt 的数据库级标准化。

真实阶段的 train/val/test 划分、候选库可用范围和泄漏防控规则须由数据方案明确。训练、验证和测试只依赖各自合法的检索 bank；不能为了提高分数把隐藏目标放入模板库。当前 dummy bank 仅用于程序验证，不提供独立评估证据。

## Dataset item：补齐前

```python
{
    "target_id": str,
    "target_architecture": int,
    "target_chain_tokens": list[int],          # 长度 4，已经 canonical sort
    "target_headgroup_token_ids": list[int],   # BOS + lexical tokens + EOS
    "target_formula": Tensor[10],             # 缺失时 0；模型不读取此字段
    "template_ids": list[str],                # 长度 K，1 <= K <= 10
    "template_smiles": list[str],
    "template_formulas": Tensor[K, 10],        # 未标准化的模板中性整数计数
    "delta_formulas": Tensor[K, 10],           # 原始 signed difference；缺失时 0
    "structure_similarity": Tensor[K],        # [0,1]
    "formula_available": BoolTensor[K],
    "template_mask": BoolTensor[K],            # 补齐前全为 True
    "structures": list[PyG.Data | Tensor[2048]],
    "encoder": "graph" | "fingerprint",
}
```

分子式以浮点 tensor 保存，但输入契约要求整数计数。0 个 hit 时 dataset 跳过该 target，记录 `skipped_target_ids`。真实标注的词表由外部提供给 `ChainVocabulary(tokens)`；只接受词表中的标签，不自行扩张。SMILES bracket atom（含 `[*:1]`）作为整个 token；额外 bracket token 由调用方传给 `SmilesTokenizer(extra_tokens=...)`。

`collate_structure_examples` 将模板容量补齐到 10，`template_mask=False` 为 padding，数值补 0。图和 FP **只展平真实模板**，顺序对应 `template_mask` 的 row-major True 项。labels 的 headgroup 字符串按 batch 最长长度 PAD，链槽的 NULL 仍参与分类 loss。

## 接入示例

```python
from structure_pretrain.config import RetrievalConfig
from structure_pretrain.data.structure_dataset import StructurePretrainingDataset
from structure_pretrain.retrieval.structural_retrieval import MorganStructuralRetriever
from structure_pretrain.utils.formula import FormulaStandardizer

# library/targets/chains/tokenizer 全部由你提供的规则构建。
retriever = MorganStructuralRetriever(library, RetrievalConfig())
train_dataset = StructurePretrainingDataset(
    targets, retriever, chains, tokenizer, encoder="graph", rwse_steps=8
)
statistics = FormulaStandardizer.fit(train_dataset.examples)
# 将同一 statistics 传给模型；val/test 不能重新 fit。
```

第一版 adapter 预处理并缓存所有检索结果和 graph/RWSE，适用于 dummy 和小型接口验证。真实库规模确定后，决定磁盘缓存、批量检索和 streaming dataset 的方案，避免在大库上反复线性扫描或缓存整个数据集。

## 仍待提供的规则

1. phospholipid structure library 的来源、范围、结构标准化和版本。
2. molecular formula library 的来源、neutral 化规则、缺失与错误处理、ID 对齐。
3. architecture 标签的可靠来源及特殊结构的处理。
4. A/O/P 链标注、P 链双键数口径、真实词表及未见 token 策略。
5. headgroup 切割、attachment 编号、合法 fragment 和 tokenizer 词表。
6. 代表双键位置策略与结构库之间的一致性。
7. train/val/test 分组、重复/同系物处理、允许访问的检索 bank。
8. 库规模、真实训练入口、缓存格式和效果评估方案。

以上目前均未使用未经你确认的数据规则填补。
