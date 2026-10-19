# v0.3 真实结构预训练方法与执行

本轮仅接入冻结 v0.3，不改原库、链代表双键、polar pool、原索引或旧输入哈希。由完整结构模板预测两个canonical chain槽位和含P polar SMILES，再用原builder组装。没有MS/MS、bis-phosphatidyl或额外预训练任务。

## 审计与划分

`python -m structure_pretrain.data.prepare_v03` 验证结构SHA256为 `22eb617b25532743a29a6d6e5d3fdacf804e3a4b5289d7ca6bcf746f7395d38a`。逐一解析195,306个polar，按原builder检验元素、价态、净电荷、单dummy与P连接。多位点PubChem来源已经在原builder中转为不同连接位点的合法单连接fragment；不假定原始来源只有一个磷酸位点。保留phosphonate/polyphosphate真实局部结构。

遍历2,015,000行的ID/SMILES哈希、polar关联、chain排序、组合许可、family/tier/origin及中性式。每行元素计数与冻结骨架+polar+chain的加和核对；lyso未占用的glycerol氧补H。该步骤检查全库关联与公式，不声称逐一重新RDKit组装2M。另独立重建600个随机结构加71个全部特殊结构，671条无重复样本的sanitize/canonical/full-formula全匹配。

Tokenizer将bracket atom及mapped dummy保留为整体token；对全部规范polar执行严格encode/decode，不使用UNK替代或截断。63词，长度含BOS/EOS：min12/P50=45/P90=66/P95=69/P99=74/max84；26,098个头基超过旧64。真实入口从审计manifest设定max84，dummy配置64仅用于短测试fixture。

对canonical polar graph统一dummy map1、去除当前任务不预测的立体信息，建立图组并交叉核查polar ID。按family/tier/origin/32-token长度分箱进行seed42的组级90/5/5分配，stratum四舍五入。当前195,306个ID与独立图一一对应。所有同图aliases的设计处理仍保留。输出：

| split | independent polar groups | full structures |
|---|---:|---:|
| Train | 175,772 | 1,813,451 |
| Validation | 9,767 | 100,792 |
| Test | 9,767 | 100,757 |

三对split的polar ID、canonical graph交集均为0。4,064个scaffold跨组，analog-parent关系单独统计。对固定各256个val/test头基与全部175,772个train头基精确比较Morgan，最近头基相似度中位数约.7901/.7931；即使指纹Tanimoto=1也不代表规范图相同。这是元数据泛化难度分析，没有用test模型成绩调参。

## 合法检索

`python -m structure_pretrain.retrieval.train_only --threads 2` 为train的1,813,451个结构及其1,276,195个独立FP groups建立单独HNSW。全库冻结指纹只作query特征和行ID映射；references从索引构建前已经限定train。不从全库Top10过滤。

使用原Morgan radius2、2048、无chirality、阈值.4、K≤10。HNSW connectivity32、expansion_add256。先在固定32个train目标上精确扫描全部train bank校准搜索强度，要求Top1分数召回≥.995、tie-aware Top10召回≥.99；当前选择expansion_search1024、候选512组。排除头基前扫描足够postings和边界ties，最终RDKit重算分数。任何不足10的结果对整个train bank精确回查，不降阈值。精确回查采用全train packed-bit扫描，保留所有边界ties，并以最大头基行数+K+1的组界保证合法TopK，最终对照RDKit。另固定样本做full-train-bank exact score benchmark。校准为小样本工程检查，不声称全库穷举最优邻居。

普通模式排除self/等价完整图，允许同头基；cross模式额外排除全部相同canonical polar graph（包括ID alias）。原库规范图全局唯一已审计，行self排除因此等价于完整图排除。Validation/Test只来自未见头基，合法模板天然cross。初始train模式按稳定的row/seed计数选择约50/50，比例可配，不宣称最优。

两种cache保存template row和Tanimoto，ID通过冻结row_ids映射。report逐split记录≥1/≥5/10、零模板、同头基ratio和top1/top10分布。Test只作要求的检索覆盖审计，不进入训练。原全库结果继续作为历史对照保存，但不供训练调用。

持久化完成后再独立遍历所有reference pairs，核验train权限、self/重复排除、canonical polar排除、mask前缀、降序及保存分数与原始2048 bits的精确Tanimoto一致。该审计不把ANN选择本身称为全局精确TopK。

## 数据、公式和模型

Arrow IPC shards每8192行，磁盘Numpy数组保存公式、链标签、polar group和split。默认最多4个shards、256个图/FP；mmap在spawn worker重新打开，不pickle整个指纹矩阵。目标仅按整数索引shuffle，不构造2M Python examples/graphs。PyG Batch只含真实templates。

目标与模板中性式必填，元素 `C H N O P S F Cl Br I`。在原始计数上求有符号target-template；缺失/非有限/全零目标式或不一致差值直接失败。公式统计用流式Welford合并合法train pairs；validation/test不拟合。普通与cross两组pairs等权合并，两种都有10模板时即等权sample mixture。常数元素std=1。

模型保留原3层GINE（hidden128、7/3化学类别、sum pooling→256）或Morgan2048→MLP→256，两者普通实验二选一。公式20→128 GELU/LayerNorm。结构256+公式128=384→512→512融合。保留2层8头FFN2048模板集合Transformer、TCLS及.2模板dropout；保留2层8头FFN2048 polar decoder。只有两个输出，chain `[B,2,323]` / polar `[B,L-1,63]`。CE等权，PAD从polar loss忽略、EOS保留，NULL从chain loss不忽略。没有architecture head、公式可用性、相似度特征或旧权重复用。

## 当前电脑的真实smoke

`train_structure_real.py` 的local配置默认512 train/128 validation、batch2、FP32、10 optimizer steps、上限50。抽样遵守冻结groups，优先包含分配到train/val的特殊结构、既有罕见P链和长度两端，再增加family/tier/origin/chain多样性。不会从test借样本。先FP后GINE，并各实际保存、重载、恢复至少一次。

入口逐步检查finite loss/gradient、各模块更新、shape/masks和OOV。验证teacher forcing覆盖128目标；greedy和beam Top5在固定16个validation目标上独立执行，仅从BOS开始，不传真实前缀、不强制EOS。报告EOS/长度耗尽、解析/attachment/polar exact、template-copy、链数/集合、原builder组装、完整式/图及Morgan。没有后验修复或公式筛选。预测全部无效时如实报告0，不能用标签重建率冒充预测成功率。

`--steps`是累计optimizer steps。检查点包括模型、AdamW、scaler、CPU/GPU/Numpy/Python RNG、已消费样本游标和全部数据契约。Sampler按epoch seed和消费游标重建shuffle；DataLoader使用独立RNG。默认每500步原子保存，smoke末尾也保存。旧checkpoint不兼容。

## 产物与迁移

正式派生目录 `data/private/pretraining_v03/` 包含input/labels manifests、polar/chain词表、标签/token/reconstruction/split audits、磁盘shards/数组、split manifest、独立train bank、两种retrieval caches及coverage、train-only formula stats。临时SSD计算目录在执行记录中登记；结果按哈希归档回正式目录。`scripts/verify_pretraining_v03.py` 对迁移的源库/索引/全部派生数组、shards及retrieval验证哈希。

`outputs/pretraining_v03/local_smoke/{fingerprint,graph}` 保存resolved config、training/generation/resource JSON和checkpoint。逐样本预测、结构、索引、checkpoint只在本地。GitHub只提交代码、配置、测试、方法文档及聚合统计。

9070 XT的待执行预检、GPU smoke和独立full命令见[WSL_ROCm.md](WSL_ROCm.md)。Full命令要求显式full配置、额外allow flag和两种encoder真实GPU smoke≥1000steps/20k train/2k val的合格报告。Smoke不会转入full。完整实验只训练train组，验证/测试组保持其权限。
