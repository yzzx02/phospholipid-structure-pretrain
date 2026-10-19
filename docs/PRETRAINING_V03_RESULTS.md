# v0.3 真实结构预训练接入与本机 smoke 结果

本轮只执行真实结构数据的工程验证。GPU验证待运行，没有完整多轮预训练或MS/MS微调。

1. **文件和保留模块**：新增全库准备、train-only检索、lazy Dataset、真实训练/自由生成/迁移核验入口。保留3层GINE(128内部宽度、7/3特征、sum pooling)或Morgan MLP、2层模板集合Transformer与2层polar decoder。
主要新增/修改文件见[聚合JSON](results/PRETRAINING_V03_RESULTS.json)的`principal_added_or_modified_files`；另更新六份`configs/pretraining_v03_*.yaml`、README、数据契约、方法和WSL文档及对应回归测试/CPU CI。原builder、冻结链词表和双键代表结构、7/3化学特征、dummy回归入口保留。
2. **两个输出**：Chain Head `[B,2,323]` 与含P Polar Decoder `[B,L−1,63]`。ArchitectureHead及四槽路径已移除；槽位不代表sn。
3. **融合**：结构256D、公式/差值20→128D、concat384→512D已实现。
4. **公式/相似度**：中性式必填、target−template有符号差值；缺失直接失败。模型不接收formula_available或Tanimoto；分数留在检索cache，统计仅用合法train pairs。
流式公式统计使用36,269,000个合法train pairs，普通/跨头基两模式按初始50/50配置合并；validation/test不参与拟合。
5. **标签/tokenizer**：2,015,000行、195,306个有效polar，OOV=0，截断=0，max=84（含BOS/EOS）。长度min/P50/P90/P95/P99/max为12/45/66/69/74/84，63词。
6. **冻结划分**：

|split|polar groups|完整结构|
|---|---:|---:|
|train|175,772|1,813,451|
|validation|9,767|100,792|
|test|9,767|100,757|

三对split的polar ID与canonical polar graph交集全部为0。Test不用于训练/调参。跨组analog/scaffold关系作为泛化难度单独报告。

7. **合法检索**：仅train bank，普通允许同头基，cross排除相同canonical polar graph及aliases。Morgan参数保持radius2/2048/无chirality、阈值.4/K≤10。下表覆盖分母包含全部目标：

|mode/split|targets|≥1|≥5|10|zero|同头基template ratio|top1中位数|
|---|---:|---:|---:|---:|---:|---:|---:|
|normal/train|1,813,451|1,813,451|1,813,451|1,813,451|0|0.609684|1.000000|
|normal/validation|100,792|100,792|100,792|100,791|0|0.000000|0.864078|
|normal/test|100,757|100,757|100,757|100,757|0|0.000000|0.865385|
|polar_excluded/train|1,813,451|1,813,451|1,813,450|1,813,446|0|0.000000|0.864865|
|polar_excluded/validation|100,792|100,792|100,792|100,791|0|0.000000|0.864078|
|polar_excluded/test|100,757|100,757|100,757|100,757|0|0.000000|0.865385|

ANN候选内RDKit精确重算；不足10均对全train bank精确回查。另作固定全bank精确样本benchmark，不将近似HNSW称为全库穷举TopK。详细分布见聚合JSON。
两模式持久化缓存独立检查共40,299,978个template pairs，train权限、自身/重复图及禁止同头基违规均为0。

8. **标签重建**：671条，包括全部71条特殊结构，组装、canonical完整图和分子式全部100%匹配。此为标签核验，不是模型预测率。

9. **本机真实训练**：每种encoder的固定可用池为512 train/128 validation，先FP后GINE。各运行10步，再从checkpoint实际恢复到11步；batch2，累计各消费22条、22个不同train目标，不是训练遍历了全部512条。以下为工程检查：

|encoder|实际累计steps|first→last train loss|val loss|greedy有效polar|EOS|初始10步RAM峰值 GiB|
|---|---:|---|---:|---:|---:|---:|
|fingerprint|11|10.0758→7.8511|7.8231|0.0000|0.0000|1.699|
|graph|11|10.1363→8.0704|7.8727|0.0000|0.0000|1.678|

所有模块梯度有限且参数确有更新，checkpoint重载logits一致，恢复后继续完成一次optimizer更新。

|encoder|实际参数数|teacher-forced polar token accuracy|初始10步 steps/s|初始10步 samples/s|resume进程RAM峰值 GiB|
|---|---:|---:|---:|---:|---:|
|fingerprint|16,766,021|0.3738|2.269|4.538|1.756|
|graph|15,738,824|0.3820|2.647|5.294|1.728|

吞吐只计训练更新时间，不包含初始化、验证及greedy/beam生成；加载计时另列JSON。RAM为主进程峰值，本机workers=0；CPU FP32运行，没有GPU显存测量。

自由生成固定16个validation目标，greedy和beam5从BOS起步，按原始EOS/长度状态评价；未修复或后验筛式。Teacher forcing、Top5/链/完整结构指标及吞吐详见JSON。两模型16/16均耗尽长度且无EOS，polar合法率、Top5命中、完整组装及公式匹配均为0。EOS停止代码路径有单测，但真实11步模型尚未学会停止，不能宣称已通过生成质量验收。10–11个优化步骤不足以评价科学效果；本轮是structure-oracle检索条件，不能称作MS/MS鉴定准确率。

10. **GPU待运行**：HIP/GPU、GINE和Transformer backward、FP32/BF16/FP16、checkpoint恢复、VRAM/吞吐与20k/2k/1000-step真实GPU smoke，命令见[WSL_ROCm.md](WSL_ROCm.md)。
在另一台电脑已经正确安装HIP PyTorch、迁移私有数据并进入仓库后执行：

```bash
python scripts/verify_pretraining_v03.py
python scripts/check_rocm.py --output outputs/pretraining_v03/rocm_preflight
python train_structure_real.py --device rocm --config configs/pretraining_v03_rocm_fp.yaml --output outputs/pretraining_v03/rocm_smoke/fingerprint
python train_structure_real.py --device rocm --config configs/pretraining_v03_rocm_gine.yaml --output outputs/pretraining_v03/rocm_smoke/graph
```


11. **正式预训练门槛**：真实RX9070XT算子与GPU smoke尚未通过，这是目前必须完成的验收。标签/重建/无泄漏/合法bank/本机管线检查见上表。生成合法性与chain表现须在有意义的GPU训练后重新评价；beam长度策略、采样混合比例及吞吐为后续实验优化项，不能从tiny smoke推断最优设置。

12. **执行边界**：未执行完整2,015,000结构库的多轮预训练，未执行MS/MS微调，未更改原库、双键代表、P链配额或冻结输入哈希。

聚合数值：[PRETRAINING_V03_RESULTS.json](results/PRETRAINING_V03_RESULTS.json)。私有产物在`data/private/pretraining_v03/`与`outputs/pretraining_v03/local_smoke/`；结构、逐样本预测、索引和checkpoint不在GitHub。
全部冻结输入哈希及原索引哈希保存在聚合JSON；派生数据另存，原始2M与v0.3原库完整保留。
