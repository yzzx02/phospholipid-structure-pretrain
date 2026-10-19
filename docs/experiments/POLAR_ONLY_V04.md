# v0.4 Polar-only：已知链条件下的结构预训练

任务契约来自 [Issue #1](https://github.com/yzzx02/phospholipid-structure-pretrain/issues/1)。
v0.4 是独立入口、数据缓存和 checkpoint 契约；v0.3 的训练入口和冻结资产保留。
实测结果与阶段状态见 [结果报告](../results/POLAR_ONLY_V04_RESULTS.md)。

## 模型与化学边界

输入是若干参考结构模板、可信的目标中性完整分子式、外部给定的一个或两个链 token。
输出只有 P-containing Polar SMILES。模型没有 Chain Head，也没有 Chain Loss。
训练时使用真实链作为已知条件；推理必须由外部给出链假设，不能输入目标 Polar 结构。
teacher forcing 中的目标 Polar 只作为解码标签；`generate()` 从 BOS 开始，忽略目标标签。

模板由冻结 builder 的 `SCAFFOLD = [*:1]OCC(O[*:2])CO[*:3]` 和冻结 Polar 片段拼接。
Polar 的 map1 dummy 与 scaffold 的 map3 dummy 连接后移除，形成 glycerol–O–P 边界。
真实占用的链位保留标准 dummy101/102；lyso 的第二位删除 dummy、保留真实 OH。
从不启发式切割完整脂质；AA/OA/PA 在相同头基与占位下具有同一个标准无链图。
链连接类型由给定链和分子式扣除、最终重建决定。单链与双链显式区分。

计数顺序固定为 `C,H,N,O,P,S,F,Cl,Br,I`。对冻结实际片段执行 AddHs 后计数，dummy 不计质量：

```text
Ffull = Fscaffold + sum(Fattached_chain) + Fattached_polar + I(lyso) * eH
Fpolar,target = Ffull,target - Fscaffold - sum(Fattached_given_chain) - I(lyso) * eH
formula_input_k = concat(Fpolar,template_k, Fpolar,target - Fpolar,template_k)  # 20D
```

这里的链是冻结的 `representative_fragment_smiles`，不是游离脂肪酸。
不另行套用任意脱水公式；空链位增加的 H 由 builder 的真实 OH 约定决定。
目标公式缺失、非整数、负推导结果或无法形成含 P 的 Polar 会抛错并停止。
非负约束针对反推的 `Fpolar,target` 元素计数；条件输入中的 target−template 差值保留正负号，不裁剪。
模板与差值的标准化只拟合 TRAIN-target / TRAIN-reference 配对，保存 K、H 比例、seed 与审计 SHA。

GINE 使用独立的 fragment parser，允许并约束 dummy101/102；不调用禁止 dummy 的 full-molecule validator。
原子特征额外有占位角色 embedding。3 层 GINE、hidden128、sum pooling、输出256D。
另一对照为重新计算的无链 Morgan radius2/2048 → MLP256D；不读取 v0.3 完整分子指纹缓存。
20D 公式 → MLP128D；与结构256D 拼接后融合为512D。
TemplateSetTransformer 保留 TCLS 与全部模板，输出 `[B,K+1,512]`，2 层 PolarDecoder 对全部 memory 做 cross-attention。
没有模板位置编码；padding 被 mask，模板置换测试覆盖 K3/5/10。
损失仅 Polar token CE：忽略 PAD、保留 EOS。

## 新检索与数据契约

沿用冻结 canonical Polar graph split：TRAIN 1,813,451 / validation100,792 / test100,757。
参考库始终 TRAIN-only，排除自身、同完整图和相同 canonical Polar。
v0.4 独立建立 chain-combination postings、head/occupancy postings 和无链 Morgan 索引。
H 表示同一精确链组合、异头基；HC 表示不同链组合、异头基；不刻意构造 C。

默认以固定 seed42 对目标／模板组作稳定的80% H、20% HC 选择。
H 无合法结果时显式 fallback HC；HC 无结果时 fallback H；两者都空时停止，不删除目标。
少于 K 时保留真实数量与 mask，不伪造链、不用混合模式填满。
阈值为新无链 Morgan Tanimoto≥0.2。阈值及覆盖的实测统计必须一起解读。
小链组合 postings（≤4096）穷举；大 postings 和 HC 使用新 TRAIN 索引的128候选、精确 Tanimoto 重排。
不足10个时做穷举确认；另以64个固定验证目标对精确最优 top1 分数作独立检查，要求≥95% 命中。
对会排除大量近邻的常见 lyso 链组合，HC 先建立排除该链组合后的合法 TRAIN 候选索引，避免反复穷举。
指纹碰撞与分数并列不视作结构相同。缓存保留最多10条，当前实验 K3。

这属于 **目标结构已知的离线预训练 oracle 检索**。未来 MS/MS 查询并不知道目标图，必须另建谱图参考检索。
本项目当前结果不是 MS/MS 鉴定准确率。

未来链候选接口 `independent_candidate_batch()` 保留 `[B,N,K]`，分别推导各候选的 Polar 公式。
仅将 B,N 展平为 B*N；每个候选具有自己的 K 模板与 memory，禁止跨 N 聚合。
没有虚构谱图或链预测器。

## 本地数据与阶段门槛

Linux 文件系统内新缓存位于 `data/private/pretraining_v04/`；实验位于 `outputs/pretraining_v04/`。
390,612个 head/occupancy 变体使用2048条 Arrow shard；大数组 mmap；图缓存最多512个、shard最多4个。
不常驻201.5万个 RDKit Mol 或 Python 图对象。
源 labels/split/chain library、特征、TRAIN bank、检索与标准化均绑定 SHA256。

1. 全库201.5万公式恒等式、原671（含全部71特殊结构）、额外10,000跨家族/lyso/AA/OA/PA结构重建。
2. 全量参考合法性、split 不交叉、相似度重算、H/HC覆盖与fallback、Ruff和pytest。
3. 真实数据 FP/GINE CPU forward、backward、参数更新、无标签 greedy、worker加载。
4. 32个不同 TRAIN 头基、同一已知链组合、同链异头基参考，随机初始化，200–500 steps。
   EOS≥95%、合法 Polar≥90%、训练 Polar graph exact≥80%；真实保存、加载并继续优化器状态。
5. 两种模型分别从**新随机权重**开始，匹配20,000目标、有效 batch16，先1000 steps。
   每500 steps保留 checkpoint、评估2,000条 teacher forcing 与固定512个不同验证头基的 raw greedy / beam5。
   后续3000/5000 steps仅在中间结果支持时有界执行；不能自动切换全库训练。

两模型复用 v0.3 的20k/2k/512目标 row IDs以便核查，但不复用权重或完整分子指纹。
过拟合单独设 dropout0、template dropout0、batch8、lr3e-4；smoke见配置文件。
GINE smoke batch8×accum2，FP batch16×accum1；seed42、Math attention、FP32。

报告包括原始 EOS、RDKit/Polar合法率、Polar graph/formula exact、给定链下完整 formula/exact、template copy、
TF-EOS、beam5召回、可选公式过滤前后覆盖与准确率，以及家族/长度/检索相似度分层。
原始预测与结构、检索缓存、checkpoints、kernel dump均不公开。
有效性采用分阶段验收：先要求真实 EOS，再检查 RDKit、attachment 与 Polar 化学规则；
各合法率分母仍为全部评估目标，未终止输出不算有效候选。
v0.3 使用预测链，v0.4 使用给定链，完整分子指标只能作带此条件的描述性比较。

## 复现命令

命令从 Linux 仓库根目录执行。`scripts/run_rocm.sh` 使用本机已验证的 `torch-rocm721` 解释器。

```bash
scripts/run_rocm.sh -u -m structure_pretrain.polar_only_v04.prepare --stage features
scripts/run_rocm.sh -u -m structure_pretrain.polar_only_v04.audit_features
scripts/run_rocm.sh -u -m structure_pretrain.polar_only_v04.retrieval
scripts/run_rocm.sh -m ruff check .
scripts/run_rocm.sh -m pytest -q
scripts/run_rocm.sh -u -m structure_pretrain.polar_only_v04.audit_cpu
```

Windows PowerShell 用 `scripts/supervise_polar_only_v04.ps1` 执行每个 GPU 阶段，独立 RunName：

```powershell
.\scripts\supervise_polar_only_v04.ps1 -Encoder fingerprint -Kind overfit -Action train -Steps 200 -RunName fp_overfit_0200
.\scripts\supervise_polar_only_v04.ps1 -Encoder fingerprint -Kind overfit -Action evaluate -RunName fp_overfit_eval_0200 -Checkpoint outputs/pretraining_v04/runs/fp_overfit_0200/checkpoint/step_000200.pt
.\scripts\supervise_polar_only_v04.ps1 -Encoder fingerprint -Kind overfit -Action train -Steps 201 -RunName fp_overfit_resume_0201 -Checkpoint outputs/pretraining_v04/runs/fp_overfit_0200/checkpoint/step_000200.pt
```

graph重复同样步骤；过拟合门槛均通过后，每种模型先 smoke500、评估，再续训1000、评估。
所有 GPU 子进程使用 Math attention 和 FP32；监控新 WATCHDOG/AMD dump、System4101、20秒无进度和总时限。
触发后立即终止本任务进程并写 `GPU_STOP.json`，后续 GPU 入口拒绝自动重试。
黑屏通知同样应立即停止。监控不能证明显示驱动问题已永久修复。
不修改驱动、核心依赖、TDR，不启动完整201.5万预训练或 MS/MS fine-tune。

如需 K5/10 或不同 H 比例，先生成匹配的 TRAIN-only 统计，再修改独立配置：

```bash
scripts/run_rocm.sh -m structure_pretrain.polar_only_v04.retrieval --statistics-only --k 5 --h-fraction 0.8 --seed 42
```
