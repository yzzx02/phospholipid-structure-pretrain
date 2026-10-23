# 模板引导结构生成：正式 V2.0 执行约定

用户 2026-10-23 的 V2.0 原文和储存补充意见已在 G 盘私有实验目录留存并校验。
本约定取代上一轮 50 epoch 的资源上限，仅适用于独立的 V2 实验。

原 50 epoch 运行标为探索性实验，全部数据、代码、317 份主训练权重、探针、
日志和停止锁保留。旧模型不得作为正式实验的初始化参数。

冻结原 195,306 核心及划分；研究范围 190,553，训练171,478、验证9,527、测试9,548。
原5,000模板、K3、Batch32、84-token、当前 v0.4 Fingerprint 架构及标签保持一致。
训练前独立检查身份/质子化等价排除、同核心不同链的划分、118个人工种子来源、
全部5,000模板的 Builder/H 与10,000个 chainless 指纹、模板分子式和缓存身份。

两种表示均采用 Morgan radius2/2048/no-chirality。Polar 用于选择与覆盖；chainless
包括甘油、磷酸和脂肪酸位点101/102，作为实际模型输入。单链保留一个标记与一个OH。
同一批190,553个查询分别做Polar、双链chainless、单链chainless精确检索，
都排除自身及当前磷酸质子化比较规则定义的等价项。USearch exhaustive 二进制
Tanimoto加RDKit重排；候选边界出现同分时全池RDKit回退，模板ID字典序处理并列。
独立512样本的Top3身份和分数必须100%吻合。阈值0.2，缺少参考使用mask，不丢目标。
公式统计在修正后的TRAIN参考对上重新拟合，不更改原缓存与标签。

已审核127个兼容参考头基对训练库的非自身覆盖单独报告，不能称作MS/MS微调集。
逐核心统计与低相似度结构、来源保持私有；公开聚合统计和PNG/SVG/PDF。
结构簇诊断采用HNSW top32稀疏图和RDKit核验边，簇数量属于近似诊断；
每个测试核心另外对全部训练核心做精确最近邻检查，剔除含近训练成员的整个诊断簇。
该子集不替换原测试集，不参与学习率选择。

三组均按seed42规则随机初始化，AdamW weight_decay0.01、clip1、FP32/Math/TF32-off：

|组|学习率|上限|
|---|---|---|
|E1|固定1e-4，无scheduler|80完整epoch|
|E2|4000次更新warmup到1e-4，cosine到1e-5|80完整epoch|
|E3|4000次更新warmup到3e-4，cosine到1e-5|80完整epoch|

每轮5359次更新，总428,720次。调度按有效optimizer update推进一次。
只用完整9,527验证核心的最低Token CE选择权重和优化方案；连续5完整轮不改善早停。
第80轮仍改善也只报告到达资源上限，不宣称收敛。测试集不用来选学习率。
CPU与CUDA均验证真实模型、优化器、scheduler、RNG、批次游标和patience的磁盘恢复；
CPU精确一致，CUDA预设绝对容差1e-5。另用真实优化器跨4000次warmup边界测试恢复。

## 新实验的精简保留

所有文件及缓存留G盘。12个不变输入数组用硬链接共享，禁止对其原地写入。
每组只保留最佳验证权重与最近两份完整恢复断点，去重后最多3份。
先原子写入、CPU重新加载、非有限检查、SHA256确认和更新索引，再删除本组已登记的
过时文件。拒绝junction/symlink、路径越界、未知文件、缺失文件和变更的旧断点。
所有步数、权重哈希、删除记录、验证曲线和失败原因保留为小体积元数据。

约0.6GiB权重/组；三组约1.8GiB。新V2独占输出软预算8GiB、硬预算12GiB，
共享输入不重复计费。超限停止并记录，不自动删除其他目录。
真实门槛恢复探针在成功恢复、记录哈希后删除自身临时权重；门槛报告和日志保留。

推断输出用gzip JSONL存每个目标的原始候选、EOS、分数、规则结果和接受候选索引；
逐目标指标分块写Parquet。聚合器不保存全批候选；不存logits、attention、
重复的完整组装结构和全量相似度矩阵。无效候选、空候选目标仍留在分母。

## 生成评价和队列

先固定分层300验证核心，再扩至完整验证和完整测试。普通Beam20和元素/语法约束
Beam20使用相同权重与候选预算，报告Top1/5/10。模型只从BOS生成。
接受、去重和排序接口不接收真实目标图；真实图仅在已排序候选上计算指标。
完整核心用同占据数chainless Morgan；Polar Morgan与可变头基另报。
可变头基只移除连接标记对应P、标记及其末端O/S配位原子，保留其他P与剩余分量。
Template Gain在对应的同一表示内比较模板和生成物，不跨指纹比较。
Builder不支持的合法类别标为out_of_scope，不能一概称为化学错误。

队列顺序：独立CUDA门槛→E1→E2→E3→完整验证CE选方案→300验证试评→完整验证→完整测试。
每个GPU子任务最多48小时，持续监测System4101、WER141/117、可读dump、NVIDIA资源。
WATCHDOG目录无读取权限，必须如实记录。任何故障锁禁止自动重试或启动后续任务。

Formula-only与额外随机种子属于后续条件阶段，须先依据正式模板模型和生成结果
评估稳定性、冻结公平对照输入/预算/参数量；本队列不提前启动。不得把全False模板mask
当作Formula-only模型。本轮不加入真实MS/MS或已知头基适应微调。

## 运行与检查

```powershell
. scripts/enter_cuda_v04.ps1
python scripts/prepare_template_guided_v2.py
python scripts/audit_template_guided_v2_coverage.py
python scripts/check_template_guided_v2_generation.py
python scripts/gate_template_guided_v2.py --device cpu --output outputs/template_guided_v2_20261023/gate_cpu
python scripts/run_template_guided_v2.py
python scripts/report_template_guided_v2.py
```

硬门槛和运行目录必须使用新名称；不覆盖旧门槛或移除故障锁来重试。
当前执行文件根目录为`outputs/template_guided_v2_20261023`。
报告命令只读实验输入并写聚合结果，不能触发训练。
