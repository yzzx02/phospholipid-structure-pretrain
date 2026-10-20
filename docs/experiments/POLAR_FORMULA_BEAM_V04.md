# v0.4 Polar-only：复用 3,000 步权重的分子式约束解码

本实验仅改变推断搜索，在固定 512 个 validation 头基上比较 Morgan/FP 与 GINE 的 Beam 5、20、50。
不训练，不修改 encoder、Formula Encoder、Template Set Transformer、TCLS、Tokenizer 或训练标签。
冻结库、检索、数据划分、已有报告和全部旧 checkpoints 保留；新输出独立位于
`outputs/formula_constrained_v04_20261020/`。

结果见 [实测报告](../results/POLAR_FORMULA_BEAM_V04_RESULTS.md) 与
[聚合 JSON](../results/POLAR_FORMULA_BEAM_V04_RESULTS.json)。

**当前状态：GPU 故障停止。** 2026-10-20 10:44 新增 Windows watchdog，WER 为 LiveKernelEvent 141，
用户确认黑屏后恢复。Beam 5 对照完成，Beam 20 只完成部分，正式 Beam 50 未运行。
停止锁保留，以下命令仅记录原执行配置，不表示现在可重启本机 GPU 作业；本轮没有自动重试。

## 方法与可比性

参考 [MetGenX 论文](https://www.nature.com/articles/s41467-026-72149-6)
及[作者仓库](https://github.com/ZhuMetLab/MetGenX)的分子组成约束思路，独立实现元素预算与前缀约束。
原文采用可计入隐式 H 的标签设计，本项目本轮保留原有 SMILES Tokenizer，因此只在搜索过程中约束可可靠计数的重原子。
这不是逐 token 约束氢数的完全复现；氢数由最终 RDKit 和 Builder 严格核对。
论文中的 Beam 128 为可选参考，本轮原计划比较 5、20、50，实际完成范围以结果报告为准。

四种方法使用同一 checkpoint、同一目标和同一模板检索分配：

1. **普通 Beam**：直接调用未修改的 `PolarDecoder.generate`，保留原始输出与原始排序。
2. **普通 Beam＋后验分子式过滤**：过滤上一步同一批候选，完全不增加搜索。
3. **重原子计数约束**：生成过程中拒绝超出目标重原子预算的 token，重原子未齐时禁止 EOS；完成后严格验证。
4. **重原子计数＋基础语法约束**：在方法 3 上增加 SMILES 前缀状态检查；完成后使用同一严格验证。

原始 Decoder 对 PAD/BOS/UNK 的屏蔽和 `log_softmax` 不变。额外约束施加在原始 log probability 之后，
**不对剩余 token 重新归一化**。每步沿用原始实现的平均 log probability 排序，
最终分数为 `sum(log p(token)) / (生成 token 数，包括真实 EOS，不包括 BOS)`。
不强制补 EOS，不修复 SMILES，不改写分子式，不使用目标结构参与候选排序。
耗尽最大长度或所有可行前缀被剪掉时，明确返回无合格候选。

## 元素和语法规则

可信完整中性式减去固定甘油骨架与外部给定链组合，沿用 v0.4 的 lyso 氢修正，得到 Polar 目标式。
公式顺序保持 `C H N O P S F Cl Br I`；逐步预算使用其中九种重元素。

- `Cl`、`Br` 各计一个原子；`c/n/o/s` 计相应元素。
- 方括号 atom token 只计其原子，例如 `[NH3+]` 计一个 N，`[13CH2:99]` 计一个 C。
- 同位素数字、map 数字、环号、键和括号不是额外原子；`[*:1]` 不消耗实际元素。
- 不根据字符、价态猜测或 token 后缀推算总 H。目标 H 改变不会改变生成中重原子预算。
- 可信十元素 schema 外的元素 token 无法表示目标式，按不可行处理。

基础语法检查维护当前原子、分支栈、成对环号、待连接键与已连接边。
避免空分支、悬空键、未闭合环/分支、自环、重复边、明显冲突的环键、重复连接标记等操作。
环号闭合后可以复用，支持 `%nn`；不实现完整的价态或芳香性求解器。
语法模式采用单连通分量的 SMILES 表示，不生成 `.` 分隔形式；这是保守的前缀限制，
不能声称接受全部 RDKit 支持的 SMILES 表示。两个模型各对固定 512 个真实标签的全部前缀做了接受性审计。

## 连接位置与严格验收

现有 checkpoint 的标签仍含 `[*:1]`。内部继续以该标记的单个 P 邻居定位甘油连接处，
不选取 SMILES 中任意出现的第一个 P；多个 P 原子合法。最终继续使用原有 Builder 的图连接和化学检查。
固定 512 个目标的额外 CPU 往返审计全部通过：188 个目标含多个 P，
130 个目标的连接 P 不是其 SMILES 中第一个 P；完整中性式（含 H）及完整 canonical 图均精确重建。
不增加连接位置预测网络、损失或数据库头基分类器。

每个候选均需：

1. 实际生成 EOS，并通过 RDKit 化学检查及原有 Polar 边界约束。
2. Polar 的全部十元素计数与目标 Polar 式完全一致，包含 H。
3. 用固定甘油和给定链通过现有 Builder 组装成功。
4. 组装后的完整中性式十元素计数与目标完整式完全一致，包含 H。

对合格候选按 canonical Polar graph 去重，保留不同连接图的合法同分异构体，
按原始 Decoder 分数排序。未在冻结头基查找表中的合法图会走原有自由生成 Builder 规则，
不会因不在数据库中而被拒绝。TRAIN 图集合仅用于事后的“训练未见”统计，不参与搜索或验收。
canonical 图沿用现有非立体化学表示；本实验不区分立体异构体，也不预测链的 sn 位置。

未来若重新训练无标记 Decoder，需另建以连接 P 为根的统一标签并验证往返重建；
本轮未删除输出 token，也不宣称现有输出层兼容无标记标签。

## 固定数据与指标

基线 commit：`2914c9b6fdce273ae2677637b9618a98a1fd064b`。

- FP 权重：`outputs/pretraining_v04/runs/fp_smoke_3000/checkpoint/step_003000.pt`。
- GINE 权重：`outputs/pretraining_v04/runs/gine_smoke_3000/checkpoint/step_003000.pt`。
- 固定样本：原有 `diagnostic_512.npy`，512 个不同的 validation Polar 组。
- 样本行 SHA256：`694d0d4c0c12330f4c555724c63dd8eeb026ff499f332108e6d4f6aae87c4fcd`。

Dataset 先用完整 512 个目标实例化，再取分段，保持既有 H/HC 检索模式分配。
生成前显式移除 `target_polar_token_ids`；目标 Polar/完整图只用于评价，未传入 Decoder。
checkpoint 合约逐项核对 3,000 步、v0.4 smoke 配置、标签、词表、特征、检索和公式标准化统计哈希。

原始候选和最终候选分别保存指标，固定全目标分母；不能把过滤后的 100% 合法率当作原始生成质量。
Beam 5 的 Top-10 标为不适用，而非当作 Top-5 重复填报。
覆盖率指至少有一个严格合格候选的目标比例，失败目标保留在 Top-k 和平均候选数的分母。
另报同式错误连接图比例、条件 Top-1 错误率、各失败阶段、候选数量分布、推断/验证耗时和资源。

这些条件包括 structure-oracle 模板检索、可信完整式和已知链；不能解释为未知 MS/MS 的鉴定能力。
同式异构体的合法性不等于其物理稳定性、生物学存在性或正确鉴定。

## 运行与故障保护

所有数据、模型、预测和日志位于 WSL Linux 文件系统。
使用原有 `/home/administrator/miniconda3/envs/torch-rocm721/bin/python`、ROCm 7.2.1、FP32、Math Attention；
没有重装或更换 Windows 驱动、ROCm、Python 依赖。

CPU 阶段先执行测试及两个 checkpoint 的真实公式/前缀审计，再写本轮独立 `cpu_gate.json`。
GPU 预检通过后，以单作业 256 目标、总时限 1,200 秒、无进展时限 20 秒串行运行。
监督器每 500 ms 检查 WATCHDOG/AMD_WATCHDOG/AMD_REPORT_UM 转储，每 5 秒检查 System 4101。
持续 epoch 覆盖作业间隔；新故障将写入停止锁并结束已核对命令行的推断进程，禁止自动重试。
任何用户报告黑屏都需立即停止后续 GPU 作业。保持旧实验的停止锁检查与 checkpoint 保护。

Windows PowerShell 7.5+ 的 JSON 时间自动转换可能导致 UTC 时间二次解析偏移，
新监督器使用 `ConvertFrom-Json -DateKind String` 保留 ISO 时间。
首次预检在 GPU 进程启动前因该时间处理问题误计历史转储，已保存误报证据；
核对原始 epoch 后没有真实新转储，修正解析后继续，原 epoch 未重置。
这次预检误报不表示发生了新的 GPU watchdog。
与该预检误报不同，10:44 的新 dump 与用户黑屏是实际故障，并已触发停止。
停止后另修正 System 4101 查询：先转换为 Windows 本地时间查询，再按 UTC 精确过滤，
避免查询范围意外包含旧事件。实际执行版本与发布版本的监督器 SHA 分开保存；
Decoder 和评估模块在正式作业期间及停止后均未改变。

## 复现

实验参数见 [配置](../../configs/polar_formula_beam_v04.yaml)。以下命令均在仓库根目录执行。
私有冻结数据和检查点不随 GitHub 提供，需先具备与聚合 JSON 相同的哈希。

CPU 测试、真实 checkpoint 审计和 gate 建立（WSL）：

```bash
/home/administrator/miniconda3/envs/torch-rocm721/bin/python \
  scripts/check_formula_beam_v04.py
```

该脚本为 CPU 子进程关闭 GPU，执行 pytest、pip check 和两份真实 checkpoint 审计，
另用 `scripts/audit_formula_targets_v04.py` 验证全部 512 个目标的完整式与完整图重建，
全部通过后写 gate；已有 gate 与日志受保护。`--verify-existing` 只读复核已有证据，不重写 gate。
GPU 作业使用 Windows PowerShell 7.5+：

```powershell
$supervisor = '\\wsl.localhost\Ubuntu-24.04\home\administrator\projects\phospholipid-structure-pretrain\scripts\supervise_formula_beam_v04.ps1'
foreach ($beam in @(5,20,50)) {
  foreach ($encoder in @('fingerprint','graph')) {
    foreach ($mode in @('raw','elements','elements_grammar')) {
      foreach ($startIndex in @(0,256)) {
        pwsh -NoProfile -ExecutionPolicy Bypass -File $supervisor `
          -Encoder $encoder -Mode $mode -BeamSize $beam `
          -Start $startIndex -Count 256 -TimeoutSeconds 1200 `
          -RunName "${encoder}_${mode}_b${beam}_s${startIndex}"
        if ($LASTEXITCODE -ne 0) { throw 'Stop all GPU jobs; no automatic retry' }
      }
    }
  }
}
```

监督器要求全新作业目录；复现实验须单独保存既有结果，不能覆盖此轮输出或绕过停止锁。
`ExecutionPolicy Bypass` 仅作用于该子 PowerShell 进程，不更改机器持久策略。
聚合脚本检查全部 36 个作业、每个目标行、哈希与 host monitor；默认拒绝缺失/重复/错误目标。
只有显式 `--allow-incomplete` 才能发布标明不完整的中断报告。

```bash
/home/administrator/miniconda3/envs/torch-rocm721/bin/python \
  scripts/report_formula_beam_v04.py
```

本轮结束后再根据结果讨论继续 5,000 步或 50k 训练池；没有启动这些后续训练。
