# v0.4-FP：Polar 相似度、模板收益与文件整理

本轮只评价，不改变模型、标签、连接规则、训练数据或冻结划分，不执行训练或 MS/MS 微调。
三个模型均为 16,468,671 参数：20k/3,000、50k/3,000、50k/5,000。
使用反复使用的 validation `diagnostic_512`，不称为独立泛化测试。

## 研究问题和参考

核心问题是：给定结构参考和目标式，生成的合法 Polar 是否比参考更接近目标？
参考 [MetGenX 原论文 Fig.3f](https://doi.org/10.1038/s41467-026-72149-6)
的 Top-N 最大结构相似度累计评价思路（[作者项目](https://github.com/ZhuMetLab/MetGenX)）。
本轮指纹、结构边界和数据任务沿用本项目，不能将数值直接与该论文的谱图鉴定结果横比。

统一使用 canonical Polar、内部 `[*:1]`、Morgan radius2/2048 bit、非手性 Tanimoto。
目标、模板、候选全部是 Polar-only；不混入 glycerol–Polar 或完整脂质指纹。
图一致性另行比较；Tanimoto=1 不在定义上等于图 Exact。

## 原始数据复用和唯一补充推断

复用已保存的五组真实逐目标结果，并复核来源哈希、固定目标、模板契约、候选排序和化学判定。
旧 native 结果继续通过原来的完整块审计复用，不拼接中断前缀。
唯一缺项是 50k/3,000 的 Beam50，使用原独立 CPU gate、FP32/Math/关闭TF32和 Windows 监督器补齐：

```powershell
. .\scripts\enter_cuda_v04.ps1
python scripts/supervise_polar_fp50k.py --job eval --checkpoint outputs/polar_fp_base_50k_20261020/train_B_3000/checkpoint/step_003000.pt --mode elements_grammar --beam-size 50 --run-name eval_similarity_3000_beam50_512 --timeout-seconds 1800
python scripts/evaluate_polar_similarity_v04.py
```

不要重用已有作业名。新补充评价不改写旧 50k 报告或原来 12 作业的历史回执。
全部新分析位于 `outputs/polar_similarity_eval_v04_20261020/`。
固定目标行哈希为 `694d0d4c0c12330f4c555724c63dd8eeb026ff499f332108e6d4f6aae87c4fcd`。
实际模板从相同冻结 H/HC、K3、H0.8、seed42 契约重建，记录模板行及私有 context 哈希；
逐项检查 TRAIN-only、排除目标及相同 canonical Polar 的原契约。
K=3为容量上限：511个目标有3个实际模板，1个目标只有1个；未用位置沿用原mask。
基线只使用真实输入模板，不补检索或复制补齐；六组模型使用相同模板集合。

CPU 对每个严格合格候选再次调用未经修改的 RDKit/Builder，核对 Polar 式含H、连接、
外部给定链组装及完整中性式。Raw 有效率复用原候选判定，严格候选与原排名及去重结果必须完全一致。
不重复执行已有 GPU 推断，不通过目标相似度重新排序。

## 相似度和收益定义

对模型分数排序、严格筛选并 canonical 图去重的候选列表，先截取前N，再计算
`S_N=max(T(candidate_i,target), i<=N)`，N=1/3/5/10。
候选少于N时使用全部已有候选；完全没有候选时 S_N=0。
另报告全部严格候选的最大值；Beam50这一项仅为既定搜索候选集合的上限诊断。
S_N 是事后评价，真实未知目标上不能按该值选候选。

分别报告全部512目标和有候选目标的条件统计，不混分母。
阈值0.5/0.6/0.7/0.8/0.9和=1同时记录数量、分母和比例。
均值、中位数和0/10/25/50/75/90/100分位数均保留。

`template_top1=T(ref1,target)`；`template_best=max_j T(ref_j,target)`。
template_best 只用于分析，不能作为未知目标时可用的模板选择器。
`Delta S_N=S_N-template_best`，N=1/5/10；abs(delta)<=1e-12归为0，
其余按正负分类。无候选目标仍以S_N=0参与。
模型间差异采用同目标配对；2,000次seed42 bootstrap仅描述此重复诊断集的重采样区间，
不代表独立泛化置信度，也不对大量探索分层作未经校正的显著性声明。

按最佳模板相似度<0.4、[0.4,0.7)、>=0.7，以及family、16-token长度档、P数量、
CORE/EXTENDED、来源和 TRAIN 分子式图频次0/1–5/6–50/>50分层。
补充最佳模板是否匹配目标式和公式L1距离分层，因为参考模板允许不同式，生成候选则要求正确式。
因此负Delta说明相对这个有利结构基线的不足，不能单独证明没有学习目标分子式条件。

## 结构变换和化学指标

记录候选模板复制、全 TRAIN 图已见/未见、三方相似度关系和新颖候选的目标相似度。
TRAIN未见不代表天然产物新发现；严格候选的合法性来自筛选定义，不能据此声称原始新颖输出100%合法。
Raw RDKit/Polar有效率以原始搜索候选为分母；未EOS候选沿用原判定记为无效。
Polar式覆盖、完整中性式覆盖和联合严格覆盖分开，Exact沿原分数排名统计。
目标进入Beam、进入后排10名以外、多合法连接图目标数也同时报告计数和比例。

保留连接P/全部P二跳邻域、官能团及图距离、末端、分支和环差异标志；可重叠，无法解释的记未分类。
共同含P核心使用保守描述代理：最佳参考与目标的连接P二跳邻域相同，候选仍保持这一邻域，
同时其他官能团/末端/分支/环图特征改变。该代理不证明唯一反应路线、合成可行性或生物真实性。

## 图表与隐私

单独 G 盘绘图环境安装 `requirements-similarity-plots.txt`，不修改 CUDA 环境：

```powershell
..\environments\polar-similarity-plots\Scripts\python.exe scripts/plot_polar_similarity_v04.py
python scripts/draw_private_similarity_cases_v04.py
```

Figure A：两种 Beam、三模型的 Top1/3/5/10 累计超越曲线（P(S>=t)）。
Figure B：模板与生成相似度的配对二维密度，保留全512目标及无候选0。
Figure C：最佳模板相似度与Delta的配对密度。
公开图只含无标识的统计密度/曲线，导出PNG、SVG、PDF。
Figure D 从真实 Top10 中按预先写明的规则事后选择改进、长链、多P及失败案例，
保留原候选排名并标注真实式/相似度/图特征差异；缺候选时中间留空，不画假分子。
仅保存在 G 盘 `private_Figure_D/`，不上传结构图、context或逐目标预测。

## 磁盘整理与保护

```powershell
python scripts/audit_experiment_storage.py --label initial
# 所有评价、绘图和测试完成且workspace Python均退出后：
python scripts/audit_experiment_storage.py --label before_cleanup --prepare-cleanup
.\scripts\apply_storage_cache_cleanup.ps1 -ValidateOnly
.\scripts\apply_storage_cache_cleanup.ps1
python scripts/verify_similarity_checkpoints_v04.py --label after_cleanup
python scripts/audit_experiment_storage.py --label after_cleanup
```

扫描物理树，跳过junction并处理硬链接，避免三处project junction重复计数。
保留恢复树、原始迁移包、所有checkpoint、数据划分、词表、索引、预测、审计和GPU停止锁。
清理只接受已哈希的__pycache__/pytest/ruff缓存，以及不完整的遗留pip wheel临时文件；
其完整离线wheel必须先匹配历史SHA。删除前逐路径验证工作区边界和无reparse point，
验证全部文件未变且无占用；任何workspace Python仍运行或进程路径不可确认则停止。
所有目标先输出dry-run，再独立执行验证和删除。其他checkpoint及大预测只列人工清单，不自动删除。
删除后CPU加载三模型及AdamW状态、核对权重和预测哈希，记录逻辑删除量及实测磁盘空闲变化。

结果见 [聚合报告](../results/POLAR_PRETRAIN_SIMILARITY_EVAL_V04_RESULTS.md)、
[统计JSON](../results/POLAR_PRETRAIN_SIMILARITY_EVAL_V04_RESULTS.json) 和
[磁盘审计](../results/EXPERIMENT_STORAGE_AUDIT.md)。
