# 验证记录

2026-10-13，在本地 Windows CPU 环境完成简化为三层 GINE 后的结构预训练 v0.1 程序核验。Linux CPU 通过 GitHub Actions 单独核验。RX 9070 XT / WSL / ROCm 尚待另一台机器实测。

本地核心版本：Python 3.13.2、torch 2.9.1 CPU、torch-geometric 2.8.0.post1、RDKit 2026.03.3、Lightning 2.6.6、OmegaConf 2.4.0、NumPy 2.2.3、pytest 9.0.3、Ruff 0.15.17。

## 已执行的验证

- pytest：31 项，覆盖 graph/FP forward、atom/bond fields、图节点顺序、结构去重、Top-K、空检索跳过、K<10 padding、模板集合置换、dropout 至少留一个、缺失公式及标准化遮蔽、train-only statistics、1/2/3/4 chain 标签、attachment tokenizer、teacher forcing 因果 mask、PAD loss 忽略、两种模型 backward 和 optimizer 更新；新增 3+2 模板、M=5 的 graph batching 与 padding 顺序检查。
- Ruff lint、format check 和 compileall。
- graph 和 fingerprint 默认完整模型各执行 2 个实际 optimizer steps，随后的 validation loss 有限，checkpoint 保存成功。
- 新 GINE checkpoint 恢复 model/optimizer/Trainer，继续到累计 step 3；改造前的图模型 checkpoint 不用于恢复。
- 参数脚本使用实际模型实例与去重后的 parameters 统计，graph=15,626,956，FP=16,654,153。词表为示例 chain 6 类、SMILES 47 类。
- 小 batch 诊断脚本对两种模型均执行 forward、三项 loss computation、backward；所有可训练参数都产生梯度，没有非有限输出或梯度。实际 shape 见 [encoder_batch_shapes.json](encoder_batch_shapes.json)。

实际运行概要见 [verification_summary.json](verification_summary.json)。本地完整 checkpoint 和日志放在忽略的 `outputs/` 中，不上传这些合成训练权重。

## 解释边界

示例只是 toy structures 和人为目标标签，用来保证框架能训练。它不提供真实化学标签、真实磷脂分布、数据库覆盖率、生成有效率或准确率证据；不把 dummy 的 loss 下降或 token accuracy 解释成预测能力。

validation 用两个未参与统计拟合的 toy target，retrieval bank 仍为 toy bank。真实评估的库使用和数据划分规则目前没有确定。

断点恢复包含 optimizer 和 Trainer 状态；不保存 dummy DataLoader 的 epoch 内游标，不声明完全相同的 batch replay。

当前 Python 3.13 下出现 PyG inspector 的第三方 typing deprecation warnings，不是测试失败。WSL 推荐 Python 3.12，以匹配参考 AMD wheels。
