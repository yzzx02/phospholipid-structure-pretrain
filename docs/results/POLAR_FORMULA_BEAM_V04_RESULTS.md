# v0.4 Polar-only 分子式约束推断实测

状态：**stopped_gpu_fault**。本轮只复用 3,000 步权重做推断，没有训练。
固定 512 个不同的 validation Polar 头基；Morgan/FP 与 GINE 共用同一批目标。
本实验仍使用 structure-oracle 检索与可信中性式、给定链，不能称为 MS/MS 鉴定准确率。

正式作业完成 14/36，中断 1 个；完整 512 目标搜索组合 7/18。
Beam 5 的两模型四方法对照完整；Morgan Beam 20 仅普通 Beam 及其后验过滤完整。
Morgan Beam 20 元素约束的 241 条中断记录仅作单独诊断；其他 Beam 20 与正式 Beam 50 未运行。

方法、计数规则、限制与复现命令见 [实验方法](../experiments/POLAR_FORMULA_BEAM_V04.md)，
精确计数、失败类别、候选数量分布、分层指标和哈希见 [聚合 JSON](POLAR_FORMULA_BEAM_V04_RESULTS.json)。

## 原始输出与最终输出

每个方法都先保存未修复的原始 beams。下表“原始合法”按其原始候选数计算；
“普通 Beam”行保留原始 Decoder 排序和输出，其他行仅返回通过完整检查、按 canonical Polar graph 去重的候选。
“覆盖”始终表示至少一个候选同时满足 Polar 元素式（含 H）、Builder 及完整中性式的目标比例。
完整组合的 Top-k 以全部 512 目标为分母；不完整组合以表中实际目标数为分母。Beam 5 的 Top-10 不适用；无合格候选不移出分母。

|Encoder|Beam|方法|目标数|原始合法|严格式覆盖|无合格候选|合法异构体均数|Polar Top-1|Top-5|Top-10|
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|fingerprint|5|普通 Beam|512|42.46%|11.72%|88.28%|0.166|0.39%|0.59%|—|
|fingerprint|5|普通 Beam＋后验式过滤|512|42.46%|11.72%|88.28%|0.166|0.59%|0.59%|—|
|fingerprint|5|重原子计数＋最终检查|512|15.08%|27.15%|72.85%|0.465|1.37%|1.56%|—|
|fingerprint|5|重原子＋基础语法＋最终检查|512|59.06%|38.09%|61.91%|0.660|1.37%|1.56%|—|
|graph|5|普通 Beam|512|41.02%|9.38%|90.62%|0.127|0.00%|0.20%|—|
|graph|5|普通 Beam＋后验式过滤|512|41.02%|9.38%|90.62%|0.127|0.20%|0.20%|—|
|graph|5|重原子计数＋最终检查|512|13.12%|23.24%|76.76%|0.373|0.59%|0.78%|—|
|graph|5|重原子＋基础语法＋最终检查|512|53.16%|30.27%|69.73%|0.502|0.59%|0.98%|—|
|fingerprint|20|普通 Beam|512|43.63%|19.73%|80.27%|0.504|0.39%|0.78%|1.17%|
|fingerprint|20|普通 Beam＋后验式过滤|512|43.63%|19.73%|80.27%|0.504|0.98%|1.37%|1.37%|

后验式过滤直接使用同一次普通 Beam 的候选，没有额外搜索。约束方法的原始输出 Top-k、
最终过滤后的 Top-k 分开保存在 JSON 的 `raw` / `strict_post_filter` 中；表中约束方法展示最终输出。
合法异构体均数包含无候选目标的 0；同一 canonical 图的不同 SMILES 不重复计数。

## 原始输出的分母与失败

约束可能让所有前缀死亡，返回少于 beam 宽度的候选，甚至空列表。因此候选合法率的分母会改变，不能单凭该比例判断改善。
下表同时给出原始候选数及按全部目标计算的原始 Top-1 合法率。这些原始指标不受最终过滤影响。

|Encoder|Beam|搜索|原始候选数|原始 Top-1 合法率|无存活 beam 目标|仅有未完成 beam 目标|
|---|---:|---|---:|---:|---:|---:|
|fingerprint|5|raw|2560|49.61%|0|0|
|fingerprint|5|elements|2560|29.10%|0|1|
|fingerprint|5|elements_grammar|1087|51.56%|99|9|
|graph|5|raw|2560|46.68%|0|0|
|graph|5|elements|2560|25.78%|0|4|
|graph|5|elements_grammar|980|45.31%|120|24|
|fingerprint|20|raw|10240|55.86%|0|0|

## 分子式正确仍可能结构错误

下表错误率的分母为通过全部检查的唯一候选，Top-1 错误率的分母为有合格输出的目标。
分子式约束不决定连接方式；生成了合法异构体并不等于找回目标结构。

|Encoder|Beam|搜索|合格唯一候选数|式正确但 Polar 图错误|合格 Top-1 图错误|训练未见合法候选数|
|---|---:|---|---:|---:|---:|---:|
|fingerprint|5|raw|85|96.47%|95.00%|53|
|fingerprint|5|elements|238|96.64%|94.96%|187|
|fingerprint|5|elements_grammar|338|97.63%|96.41%|284|
|graph|5|raw|65|98.46%|97.92%|43|
|graph|5|elements|191|97.91%|97.48%|153|
|graph|5|elements_grammar|257|98.05%|98.06%|214|
|fingerprint|20|raw|258|97.29%|95.05%|178|

“训练未见”仅表示 canonical Polar graph 不在 TRAIN 标签集合中，不是全球化学新颖性声明。
候选验收未使用该集合过滤，未要求头基必须存在于冻结数据库。

## 资源与故障监控

每个搜索/编码器/beam 组合分成两个 256 目标的串行 GPU 作业。每作业上限 1,200 秒，
20 秒无进展停止；Windows watchdog dump 每 500 ms、System 4101 每 5 秒检查。
进程结束及下一作业开始还会复查自本实验 epoch 以来的故障，防止遗漏迟到转储。

|Encoder|Beam|搜索|推断＋验证秒|含加载宿主秒|峰值 PyTorch 分配显存 MiB|峰值进程 RAM MiB|
|---|---:|---|---:|---:|---:|---:|
|fingerprint|5|raw|100.44|123.83|147.2|2054.7|
|fingerprint|5|elements|124.23|138.97|150.9|2060.7|
|fingerprint|5|elements_grammar|166.94|183.23|150.9|2115.6|
|graph|5|raw|107.40|122.39|143.7|2021.9|
|graph|5|elements|128.64|144.06|146.1|2031.1|
|graph|5|elements_grammar|176.95|195.57|146.1|2090.4|
|fingerprint|20|raw|218.16|232.98|171.3|2088.4|

显存为 PyTorch allocator 的最大 allocated 值，不包含 Windows 显示占用或全部驱动开销。
推断时间含模板编码、解码、RDKit/Builder 验证和逐目标落盘，不含模型/库的初始加载；宿主时间包含加载。
普通 Beam 的后验过滤复用其解码，耗时不重复计入；JSON 另列 decode / validation 时间。

## GPU 故障与停止

2026-10-20 10:44（北京时间），Windows 新增 `WATCHDOG-20261020-1044.dmp`。
匹配该转储的两条 WER 事件均为 `LiveKernelEvent 141`，用户确认发生黑屏、随后恢复。
该码表示显示引擎未及时响应，见 [Microsoft 定义](https://learn.microsoft.com/en-us/windows-hardware/drivers/debugger/bug-check-0x141---video-engine-timeout-detected)。
监督器因新 dump 写入停止锁，终止推断；外层作业循环以失败退出，没有重试或后续 GPU 作业。
停止处理后 WSL 子进程退出码为 15；stderr 为空，未观察到 PyTorch 数值异常记录。
Windows System 4101 为 0 仍发生了真实故障，不能只靠该事件判断 GPU 是否正常。
精确根因未定位；本次结果说明 Math Attention＋FP32 仍未消除这台主机的黑屏风险。

|中断作业|已保存目标/计划|新 GPU dump|退出码|停止原因|
|---|---:|---:|---:|---|
|fingerprint_elements_b20_s0|241/256|1|15|new_windows_gpu_fault_dump|

中断作业的分母不是 512，因此未纳入主对照；JSON 单独保留其实际前缀目标的诊断聚合。
此外保留两个各 2 目标的 GPU 预检（含 GINE Beam 50），不把预检当作正式比较。

## 本轮结论

- fingerprint Beam 5：严格分子式覆盖率由 11.72% 增至 38.09%；Polar Top-5 exact 由 3/512 增至 8/512。
- graph Beam 5：严格分子式覆盖率由 9.38% 增至 30.27%；Polar Top-5 exact 由 1/512 增至 5/512。
- 元素计数能提高公式候选覆盖，但单独使用会降低原始化学合法率；基础语法和最终含 H 验证仍必要。
- 同式合法候选大部分仍是错误目标图，约束不能替代结构识别与排序；不能据覆盖提升宣称鉴定准确率高。
- 不能根据未完成的实验判断 Beam 20/50 是否优于 Beam 5，或判断哪种 encoder 普遍更优。
- 本轮没有启动 5,000 步、50k 池、全量训练或 MS/MS 微调；故障停止锁保留，后续 GPU 工作停止。

## 保护与限制

GPU 前 CPU 测试：353 项通过；两个 checkpoint 各审计 512 个真实标签的元素式及语法前缀。
最终回归：358 项测试，另有 Ruff、PowerShell 语法与 UTC 时间转换回归检查。
旧实验保护审计：passed；保护文件 1355 个，checkpoints 54 个。
继续使用 ROCm 7.2.1、Math Attention、FP32；Windows 显卡驱动及 Python 依赖未调整。
停止锁：`{'reason': 'new_windows_gpu_fault_dump', 'utc': '2026-10-20T02:44:17.0169950Z', 'run': 'fingerprint_elements_b20_s0'}`。历史宿主黑屏记录保留；即使本实验未观察到新故障，也不能证明此前故障已解决。
Beam 128 未运行；Beam 50 仅有 2 目标预检，没有完成固定 512 目标正式比较。
私有结构、逐目标预测及 checkpoints 留在 WSL Linux 文件系统；GitHub 仅包含代码、配置和聚合结果。

原始基线 commit：`2914c9b6fdce273ae2677637b9618a98a1fd064b`。检查点、标签、tokenizer、检索及样本哈希见聚合 JSON。
