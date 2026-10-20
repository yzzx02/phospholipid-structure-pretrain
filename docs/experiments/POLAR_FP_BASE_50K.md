# v0.4-FP 稳定结构预训练：50k，最多 5,000 steps

本实验按 2026-10-20 用户的新计划执行，独立于已冻结的 20k/3,000-step 模型。
新模型从 seed=42 随机初始化；保留现有标签、`[*:1]`、词表、Builder、冻结 split 和 GINE 历史。
所有私有数据、逐目标输出、权重和缓存均留在 G 盘。仅发布源码、配置和聚合指标。

## 固定模型与优化

|模块|配置|
|---|---|
|参考结构|去链 glycerol–Polar；给定目标完整中性式和外部链|
|结构编码|Morgan radius 2 / 2048 bits，MLP 2048→512→256|
|公式编码|模板 Polar 式 10D + Polar 差值 10D →128|
|融合|384→512→512|
|Template Set|2 层，8 heads；TCLS；memory 为 TCLS+K 个模板|
|Decoder|2 层，8 heads，512D，FFN 2048；最大长度 84，含 BOS/EOS|
|正则|模型 dropout 0.1，template dropout 0.2|
|参数量|16,468,671；FP 1,181,952，projector 463,744，set 6,306,304，decoder 8,516,671|
|优化|AdamW，lr=1e-4，weight decay=0.01，batch=16，clip=1.0|
|运行|FP32，Math SDPA，禁用 TF32；最多 5,000 optimizer steps|

不增加 Chain Head、CL 或其他输出，不重建结构库，不开展 MS/MS 微调。

## 目标池和检索

入口为 `python -m structure_pretrain.polar_only_v04.fp50k_data`，固定 seed=42。
从冻结 TRAIN 的 175,772 个 canonical Polar graph 中，每图选择一个真实完整脂质行。
先保留 TRAIN 中有稀有乙烯醚连接的 graph 及已有 special-case graph；
其余按 chemistry family、16-token 长度档和元素集合轮询，各档内部优先覆盖不同分子式。
这是一项刻意增加头基多样性的抽样，分布变化在结果中报告，不能把结果单独归因于池大小。

最终池为 50,000 行、50,000 个不同 Polar graph、31,206 种 Polar 式。
包括 5,374 单链、44,626 双链；A/A 31,067、O/A 13,558、P/A 1。
冻结库中 P/A 只有极少数已允许组合，本轮保留可用 TRAIN graph，不复制该图凑数量。
原 20k 池只有 18,112 个不同图，保持原样。

K=3、Morgan 阈值 0.2、H/HC 请求比例 80%/20% 不变。
复用冻结全 TRAIN bank 的缓存及相同 K/H/seed 的 TRAIN-only 公式统计；
统计由 5,439,380 个 TRAIN-target/TRAIN-template 配对拟合，没有 validation/test 拟合。
实际 149,967 个配对全部重新核对 TRAIN-only、异图、非自身、H/HC 链规则和精确 Tanimoto。
H 配对占 80.1996%，5 个目标采用已有合法 fallback；49,976 个目标有 3 个模板，
15 个有 2 个、9 个有 1 个，零模板目标为 0。

私有 `pool/manifest_private.json` 保存行 ID 数组哈希、各分布及输入文件哈希。
池和模型契约均位于独立 `outputs/polar_fp_base_50k_20261020/`，不写入历史 junction。
构池时发现首次随机选链未保留稀有 P/A，正式训练前已加入保留规则重新构池；
首次未用的池留在 `pool_initial_no_rare_links_private/`，没有用于任何训练或选模。

## 本机入口和安全门槛

```powershell
. .\scripts\enter_cuda_v04.ps1
python -m structure_pretrain.polar_only_v04.fp50k_data
python scripts/check_polar_fp50k.py
python scripts/supervise_polar_fp50k.py --job train --probe --steps 20 --run-name gpu_probe_20 --timeout-seconds 600
python scripts/supervise_polar_fp50k.py --job train --probe --steps 25 --resume outputs/polar_fp_base_50k_20261020/gpu_probe_20/checkpoint/step_000020.pt --run-name gpu_probe_resume_25 --timeout-seconds 600
python scripts/check_polar_fp50k.py --gpu-result
python scripts/supervise_polar_fp50k.py --job train --steps 1000 --run-name train_A_1000 --timeout-seconds 1800
```

CPU gate 实际训练 8 步，比较连续 4 步与 2+2 真实恢复，要求权重、优化器、CPU/Python RNG、
样本位置和 loss 序列完全一致；另有完整 pytest 和 pip check。
CUDA probe 实际执行 20+5 步，验证有限 loss、非零梯度、四模块更新、优化器步数和重载 logits。
probe checkpoint 与正式训练 checkpoint 由契约字段隔离，不能互相恢复。
正式训练仍从随机初始化开始。

监督器复用已验证的 Windows 事件和 NVIDIA 遥测查询，使用本实验独立连续监控 epoch。
System 4101、Application WER LiveKernelEvent 141/117、可读 GPU dump 目录、nvidia-smi 均监控。
当前普通进程无法读取受保护 WATCHDOG，逐作业记录该限制。
GPU 故障、子进程/CUDA 失败、超时、监控异常均停止并写锁，不自动重试。
旧 FP/GINE 权重及每次恢复/评价输入的 checkpoint 均在作业前后核验 SHA256。

## 阶段和恢复

训练入口为 `train_polar_fp50k.py`，每 200 步在冻结 diagnostic 512 上记录 teacher-forced CE/token accuracy。
每 1,000 步保存完整模型、AdamW、CPU/CUDA/NumPy/Python RNG、累计消费位置、历史和严格配置契约。
文件先写 `.partial` 再原子改名；存在的目录或 checkpoint 不覆盖。
DataLoader 预取不计入消费进度，恢复由已提交的消费位置重建 epoch permutation。

阶段 A 为 1,000；通过工程门槛后记录 `stage_A_decision.json`，真实恢复到阶段 B 的 3,000。
阶段 B 在同一 512 目标、K=3、Beam20 elements_grammar 下比较旧 20k/3,000。
只有中期结果支持继续，才写 `stage_B_decision.json` 并恢复至最多 5,000。
监督器强制检查这些决策及契约。保留各整千步模型；可根据中期证据提前结束。

评价例子（`STEP`、`RUN` 替换为实际 checkpoint 和全新运行名）：

```powershell
python scripts/supervise_polar_fp50k.py --job eval --checkpoint STEP --mode raw --beam-size 1 --run-name RUN
python scripts/supervise_polar_fp50k.py --job eval --checkpoint STEP --mode elements_grammar --beam-size 20 --run-name RUN
python scripts/supervise_polar_fp50k.py --job eval --checkpoint STEP --mode elements_grammar --beam-size 50 --run-name RUN
python scripts/analyze_polar_fp50k.py --include-baseline
```

## 评价口径

Greedy 是原始无约束生成；正式 Beam20/50 使用既有元素+语法实现。
保存验证前的 raw token beam 和验证后的候选两个私有文件，聚合 JSON 分开报告 raw / strict。
约束模式的 raw 指“该约束搜索的检查前输出”，不冒充无约束 Beam。
H 仅由最终 RDKit/Builder 的化学计算严格核对；不修补或改变目标式。
严格通过 Polar 式、连接、已知链组装和完整中性式后按 canonical Polar graph 去重、Decoder 分数排序。
无候选目标继续计入分母。

结构分析计算 Polar-only Morgan radius2/2048 非手性 Tanimoto，包括 internal dummy。
同时给出所有合格候选、错误合格候选的分布，以及全目标 strict Top1/best；后两者无候选记 0。
比较输入模板复制、与 Top1 模板/目标的相似性、TRAIN 已见/新图、Beam 内目标名次，
按 family、长度、目标-模板 Polar 相似性分层。
错误标志使用 graph 上连接 P / 全部 P 的二跳邻域、官能团 SMARTS 及到连接 P 的最短距离、
末端原子及键、非 P 分支度和环组成。标志可重叠，只描述图差异，不推断唯一原因；
无法由这些特征可靠区分的错误归为 unclassified。没有使用 SMILES 编辑距离。

这 512 条是反复使用的诊断集，调参后的结果不属于独立泛化估计。
TEST 不进入训练、评价或选模；structure-oracle 检索和给定链/式不能替代真实 MS/MS 证据。
20k/3,000 对 50k/3,000 是同步数比较，但抽样分布也变了；对 50k/5,000 仅描述性比较。

实际完成阶段、资源、指标和结论见 [结果](../results/POLAR_FP_BASE_50K_RESULTS.md)
与 [聚合 JSON](../results/POLAR_FP_BASE_50K_RESULTS.json)。
