# 真实结构预训练数据契约 v0.3

schema：`structure-pretraining-v0.3-two-output`。完整结构只输入编码器；标签来自冻结builder的polar/chain关联表，不重新切割完整结构。

`StructureRecord(structure_id, smiles, neutral_formula)` 的中性式必填：元素顺序 `C H N O P S F Cl Br I`，10个非负整数。模板为单连通、无dummy、净电荷零的完整分子。`TargetRecord(structure, chain_tokens, polar_fragment_smiles)`用于小数据adapter，正式`RealStructureDataset`从磁盘读取同一语义。

链标签两个canonical槽位，按builder的O→P→A、C数、双键数排序，保留重复链；lyso尾部补NULL，不表示sn。冻结A177/O84/P61+NULL。Polar边界为`glycerol-O | P-containing polar fragment`；唯一mapped dummy连接P，由`validate_polar_fragment`和`attachment`校验，不能根据SMILES开头判断。

```python
{
    "target_id": str,
    "canonical_full_smiles": str,
    "polar_fragment_smiles": str,
    "target_chain_tokens": list[int],       # 两槽，NULL参与监督
    "target_polar_token_ids": list[int],    # BOS+lexical+EOS，最长84
    "target_formula": Tensor[10],          # 必填中性元素计数
    "template_ids": list[str],              # K=1..10，严格train bank
    "template_smiles": list[str],
    "template_polar_smiles": list[str],      # 只用于生成评价
    "template_formulas": Tensor[K,10],
    "delta_formulas": Tensor[K,10],         # target-template，允许负数
    "template_mask": BoolTensor[K],
    "structures": list[PyG.Data | Tensor[2048]],
    "encoder": "graph" | "fingerprint",
}
```

没有architecture监督、formula_available或模型相似度输入。Tanimoto单独保留在检索cache。模型验证目标式必填且有效模板的`template+delta=target`。padding数值零只对应mask=False，不代表真实缺失公式。

Collate补至K=10，`template_counts=[B]`；真实图/FP按mask的row-major顺序展平M条，不生成padding假图。图atom为`[N,7]`、bond为`[E,3]`。结构编码`[M,256]`，公式20D独立编码128D，384D融合到512D；集合memory加TCLS后`[B,11,512]`。输出`chain_logits=[B,2,323]`和`polar_logits=[B,L-1,63]`。Polar CE只忽略PAD，监督EOS；总loss为两项CE等权相加。

Canonical polar graph分组、seed42、约90/5/5；所有alias同组。三个split的references均来自train bank，train排除自身/等价完整图。跨头基模式额外排除相同canonical polar graph。Validation/Test天然跨头基。初始普通/跨头基混合.5可配置，确定性模式选择不依赖worker调度。零模板目标保留在覆盖分母，Dataset明确失败，不跳过或降低阈值。行级self排除依赖已审计的完整canonical图唯一前提。

`labels_manifest.json`绑定源表、旧索引、派生数组和shards哈希；`split_manifest.parquet`保存行→ID→polar graph→split。源`canonical_smiles`只在新shards显式映射为`canonical_full_smiles`，原表不改动。模板行通过冻结`row_ids.npy`映射ID。

Dataset不展开2M Python examples/graphs。默认缓存4个Arrow mmap shards、256个模板特征；spawn worker重开mmap。公式统计用流式矩，仅使用train targets/train references。普通/跨头基等权pair合并；两模式都Top10时对应等权sample mixture。常数维std=1。

Checkpoint绑定v0.3、标签、词表、划分、检索、公式统计哈希；恢复model/AdamW/scaler/RNG/消费样本计数。Sampler按epoch seed和消费游标恢复，DataLoader用独立RNG。旧checkpoint不兼容。

Teacher forcing与BOS-only greedy/beam Top5分开评价。自由生成记录真实EOS、长度耗尽、解析/attachment/polar图合法性、polar exact、链数和无序链集合、template-copy、原builder组装、完整式/图匹配及Morgan。未组装样本的总体相似度记0，另报成功组装条件均值。没有后验修复或公式筛选。
