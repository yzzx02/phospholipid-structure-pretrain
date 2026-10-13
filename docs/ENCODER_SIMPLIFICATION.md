# Structure Encoder 简化报告

2026-10-13。当前仅保留 **Morgan FP → MLP** 与 **3-layer GINE → sum pooling → MLP** 两条结构表示路径，输出均为 `[M,256]`。

## 修改文件

| 文件 | 修改 |
|---|---|
| `src/structure_pretrain/config.py` | 去掉图 attention/RWSE 配置；固定 GINE、3 层、128 hidden、sum、256 输出 |
| `configs/structure_pretrain_graph.yaml` | 同步简化后的 graph 参数 |
| `src/structure_pretrain/models/graph_encoder.py` | 复用 atom/bond Embedding，三层 PyG GINEConv 替换 GPS |
| `src/structure_pretrain/data/molecular_graph.py` | 仅移除随机游走计算和输出；原有 7/3 个化学特征完全保留 |
| `src/structure_pretrain/data/structure_dataset.py` | 移除随机游走参数及传递 |
| `tests/test_graph_encoder.py` | 特征检查改为当前 graph 数据格式 |
| `tests/test_graph_batch_padding.py` | 新增 B=2、3+2 模板的 shape、pooling 和 padding 顺序检查 |
| `tests/test_structure_pretrain_backward.py` | 检查所有可训练参数均有有限梯度 |
| `scripts/check_encoder_batch.py` | 可复跑的两条路径小 batch / loss / backward 诊断 |
| `.github/workflows/cpu.yml` | Linux CPU CI 增加小 batch 诊断并保存 JSON |
| `README.md`、`docs/DATA_SCHEMA.md`、`docs/WSL_ROCm.md` | 更新当前架构、接口和运行说明 |
| `docs/VERIFICATION.md`、`docs/verification_summary.json` | 更新实际验证和参数记录 |
| `docs/ENCODER_SIMPLIFICATION.md`、`docs/encoder_batch_shapes.json` | 本报告及逐项实际 shape 证据 |

`fingerprint_encoder.py`、`batching.py`、`structure_pretrain_model.py`、TemplateTokenProjector、下游 Template Set Transformer、Headgroup Decoder、prediction heads、训练入口、loss 和 metrics 均未改变逻辑。数据划分、seed=42、K=10、标签和监督目标保留原实现。YAML/CLI 模式仍使用现有名称 `graph` / `fingerprint`；`graph` 只表示 GINE。

## 已移除的部分

- Graph Encoder 中的 GPSConv、分子内部 MHSA/global attention 及其 heads 设置。
- `rwse_encoder`、`rwse_steps`，随机游走邻接矩阵/转移矩阵/return probability 的计算与 `Data.rwse` 输出。
- 图编码器中额外 positional/structural encoding 路径；不提供旧图模型兼容分支。

下游模板 Transformer 的 attention 和 Headgroup Decoder 的序列位置 embedding 保留，二者不是本次删除的分子图编码器功能。没有添加任何 GPS/RWSE/heads 结构消融。改造前的图模型 checkpoint 需重新训练，不能当作新 GINE checkpoint 续训。

## 原始 atom/bond 编码：以预处理代码为准

`x` 的 dtype 为 `torch.long`，全部是**离散类别索引**，不是已有的连续向量：

| 列 | 内容 | 实际编码 | 类别容量 |
|---|---|---|---:|
| 0 | atomic number | `GetAtomicNum()` | 119 |
| 1 | degree | `min(GetDegree(),8)` | 9 |
| 2 | formal charge | `GetFormalCharge()+4`，支持 −4…4 | 9 |
| 3 | hybridization | `min(int(GetHybridization()),8)` | 9 |
| 4 | aromatic | 0/1 | 2 |
| 5 | in ring | 0/1 | 2 |
| 6 | number of hydrogens | `min(GetTotalNumHs(),8)` | 9 |

当前 x 没有 chirality 列。`edge_attr` 同样为 `torch.long`：

| 列 | 内容 | 编码 | 类别容量 |
|---|---|---|---:|
| 0 | bond type | single=0、double=1、triple=2、aromatic=3 | 4 |
| 1 | conjugated | 0/1 | 2 |
| 2 | in ring | 0/1 | 2 |

复用已有 `CategoricalFeatures`：每列用一个独立 `Embedding(cardinality,128)`，再逐维相加。七列 atom 得到 `[N_atoms,128]`；三列 bond 得到 `[N_edges,128]`。没有把类别 ID 当连续值送入 `Linear(7,128)` 或 `Linear(3,128)`。GINE 实际使用的是 **128D edge embedding**。

## 三层 GINE 与 pooling

每层是标准 `PyG GINEConv(train_eps=True)`，内部 MLP 为：

```text
Linear(128,128) → GELU → Linear(128,128)
```

每次 message passing 后：

```text
GELU → LayerNorm(128) → Dropout(0.1)
```

没有加入新的复杂 residual block。三层输出仍为 `[N_atoms,128]`。随后 `global_add_pool(state, graphs.batch, size=graphs.num_graphs)` 对每个分子的原子逐维求和，输出 `[M,128]`；沿用 readout：

```text
Linear(128,256) → GELU → Dropout(0.1) → LayerNorm(256)
```

最终 graph 输出 `[M,256]`。Morgan 原 MLP 保留 `2048→512→256`，输出同样为 `[M,256]`。参数量分别：GINE encoder **154,755**、Morgan encoder **1,181,952**；完整模型分别 **15,626,956** 与 **16,654,153**，基于示例 6 类 chain / 47 类 SMILES 词表。

## M 与 B×K 的索引关系

`data/batching.py:28` 展平真实模板列表；`:45` 用 `Batch.from_data_list(structures)` 合并这些图，不产生 padding 分子图。

`models/structure_pretrain_model.py:44` 分配 `[B,K,256]`，`:45` 以 `structure[mask] = flat` 按 row-major 的有效 mask 位置填入 `[M,256]`。这些代码没有改动。因此 M 是每个样本真实模板数的总和，而不是 B×K。

## 实际小 batch

采用原 dummy dataset，sample 1 截取 3 个模板，sample 2 截取 2 个；B=2、K=10、M=5，真实合并图含 46 个原子、82 条有向边。实际记录来自 forward hooks，不是只按公式推算。

| 张量 | 实际 shape |
|---|---|
| x / edge_index / edge_attr / batch | `[46,7]` / `[2,82]` / `[82,3]` / `[46]` |
| node_h / edge_h | `[46,128]` / `[82,128]` |
| GINE 1/2/3 输出 | 每层 `[46,128]` |
| pooled | `[5,128]` |
| Graph output | `[5,256]` |
| Morgan input / output | `[5,2048]` / `[5,256]` |
| padding 后 | `[2,10,256]` |
| template tokens / memory | `[2,10,512]` / `[2,11,512]` |
| architecture / chain logits | `[2,2]` / `[2,4,6]` |
| headgroup logits | `[2,6,47]` |

两条路径均核验如下 mask，并逐项验证 `padded[mask] == flat`、padding 向量为 0：

```text
sample1: 1 1 1 0 0 0 0 0 0 0
sample2: 1 1 0 0 0 0 0 0 0 0
```

## forward / loss / backward 结果

两条路径的小 batch 均完成原有三项 CE loss 和 backward，**未使用参数数=0、非有限梯度数=0、非有限输出数=0、shape mismatch=0**。GINE 总 loss=6.651118，Morgan 总 loss=6.413305；它们是人工标签 dummy 的数值检查，不是模型效果比较。

31 项 pytest 通过；两种完整默认模型各完成 2 个实际 optimizer steps、validation 和 checkpoint 保存。新 GINE checkpoint 可继续恢复训练。结构预训练流程保留；9070 XT / WSL ROCm 尚未在目标机器实测。

复跑命令：

```bash
python -m pytest -q
python scripts/check_encoder_batch.py
python scripts/count_parameters.py
```

小 batch 的完整原始记录见 [encoder_batch_shapes.json](encoder_batch_shapes.json)，训练与验证摘要见 [verification_summary.json](verification_summary.json)。
