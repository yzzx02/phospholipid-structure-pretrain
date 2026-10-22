# 固定 5,000 结构模板训练：执行契约

本实验训练以结构模板和分子式为条件的 Polar 生成器。主实验为 5,000 模板、K3、seed42、随机初始化；
这次结构训练即第一版模型训练，不自动加入 MS/MS 微调或 PGCN。

## 论文的选模规则

[MetGenX 原文](https://doi.org/10.1038/s41467-026-72149-6)的 “The training of MetGenX”
记载主实验 batch32，使用交叉熵和验证 loss 早停，连续 5 个 epoch 无改善即停止。
原文结构训练在 33 epoch 收敛；这不是本实验的强制训练轮数。
其另一组 MassSpecGym benchmark 的 batch64，以及仓库中的相关示例，不能当作主实验参数。
核验用开放全文 XML 保存在 G 盘私有输出 `paper/MetGenX_PMC13279990.xml`；
[Europe PMC 开放全文](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC13279990/fullTextXML)提供相同方法内容。

本实验每完整 epoch 对全部符合范围的验证头基计算 token 交叉熵，PAD 不计入、EOS 计入。
只有严格降低验证 loss 才更新最佳 checkpoint；相等也计入无改善轮数。
最佳 checkpoint 的选择不使用测试集、化学有效率、Similarity@10、官能团或模板收益。
这些生成指标在模型选定后独立评价，不能用来换选另一个 checkpoint。

## 当前网络与执行预算

|项目|设置|
|---|---|
|网络|沿用现有 v0.4 fingerprint 配置，Morgan r2/2048、256D 结构、128D 公式、512D 融合|
|编码/解码|现有 Template Set Transformer/TCLS 和 Polar Decoder；Tokenizer/Builder/标签不变|
|优化器|现有 AdamW，lr1e-4、weight decay0.01、clip1、累积1|
|batch|有界实测16/32，优先使用论文主实验32；必须通过最长序列反传|
|精度|FP32、Math SDPA、TF32关闭|
|本地上限|50 epochs、300,000 optimizer steps、单作业48小时，先达到者为界|
|早停/选模|完整验证集最小 token CE，连续5个完整epoch无改善停止|
|checkpoint|每1,000步及每个完整epoch原子保存；最佳和最近指针分开|
|加速|模板FP和小型训练张量一次驻留GPU；确定性长度分桶减少PAD；每个核心每epoch一次|
|可复现|固定seed、epoch/游标和随机状态；CPU精确恢复，GPU允许预先声明的FP32数值容差；记录环境及SHA|

50 epochs 是本地有界预算，不是声称论文使用该上限。
按用户前述要求保留现有网络与优化器；论文的 lr/weight decay/warmup 与本项目不同，
不宣称完整参数复现。此次对齐的是主实验 batch 和模型选择/早停规则。

## 独立数据与模板池

冻结195,306图、旧split和恢复目录只读。派生范围保持600 Da，去掉明显额外脂质锚点，
并遵守现有84-token Decoder标签上限。目标按头基核心计数，每核心固定选择一个真实完整脂质行，
保持可信中性全式和已知链组合，推导出的 Polar 公式逐行与冻结缓存相同。

正式参考池仅从原训练核心补选，加入经过人工审核且通过入口检查的外部先验，
排除验证/测试及其磷酸末端质子化等价图。原全库小池提案仍保留，本正式池另建。
每个目标的自体和等价核心均从输入模板检索结果排除；同一模板FP按单/双链占位匹配。

模板补选采用训练集第六近邻密度、MW/P/N/S/卤素分层的平方根总体分配，
以及对已选图的Tanimoto球排除（新图到已有最大相似度不超过0.85）。
先得到嵌套500/2,000/5,000/10,000/20,000池，保留来源和人工种子标记；算法补选不冒充人工批准或文献已报道。
主用5,000池逐个验证连接、Builder及独立含H元素平衡，沿用原chainless Morgan入口。

检索使用HNSW候选和精确位向量重排，并进行随机精确top1分数核查。
每目标保留最多10个参考供后续K实验；当前训练只用前3，结构阈值0.2沿用项目设置。
公式标准化统计只来自当前5,000池的训练配对，旧检索缓存和标准化统计不复用。
各池对全库结构和分子式距离的覆盖属于选池完成后的诊断，不参与验证/测试调参。

## 执行与保护

在原生Windows PowerShell中 dot-source `scripts/enter_cuda_v04.ps1`。所有文件留在G盘。

```powershell
python scripts/prepare_template_guided_fp5000.py
python scripts/gate_template_guided_fp5000.py --job cpu --output outputs/template_guided_fp5000_20261022_r1/cpu_gate_run
python scripts/supervise_template_guided_fp5000.py --job benchmark --run-name benchmark_batch16_32 --timeout-seconds 1800
python scripts/supervise_template_guided_fp5000.py --job gpu_gate --run-name cuda_backward_resume --timeout-seconds 1800
python scripts/supervise_template_guided_fp5000.py --job train --run-name train_main_k3_seed42 --timeout-seconds 172800
```

CPU/CUDA gate执行真实数据反传与磁盘保存/重载，对比连续4步与2+2步的全部模型、优化器、RNG、损失和logits，
并检查全部学习模块更新。正式恢复还固定epoch、batch游标、样本消费量、完整早停状态及随机状态。
probe与正式训练各自输出，正式模型从随机初始化开始，不继承probe权重。

监督器监测System4101、WER LiveKernelEvent141/117、可访问dump和NVIDIA遥测，
核验历史权重SHA；失败或超时后锁定，不自动重试。WATCHDOG受保护目录不可读的限制会记录。
主模型训练与随后模板规模、K、随机种子和最终生成评价是独立作业，不能把主作业启动称为全计划完成。

首次GPU测速因新增严格确定性开关与torch2.9 CUDA nll_loss2d不兼容，在第一步loss处退出。
无驱动重置或新GPU dump，12份历史checkpoint SHA不变；原目录和GPU_STOP均保留。
已恢复历史的非严格确定性运行设置，并在`outputs/template_guided_fp5000_20261022_r1`重新CPU验收；
其data junction只读取原目录已校验的数据。该修订在人工明确同意GPU重试前，不得自动启动GPU。

实际执行状态与聚合数值见 `docs/results/TEMPLATE_GUIDED_FP5000_START_20261022.md`。
