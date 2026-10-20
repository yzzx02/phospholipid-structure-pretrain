# G盘实验文件占用与安全清理审计

扫描G:/yzx/linzhiyuce的项目、恢复副本、迁移包、独立环境和缓存。按物理目录扫描并跳过junction；文件大小为逻辑字节，不冒充NTFS分配块大小。
所有checkpoint、数据、预测、配置、审计、故障日志与停止锁保留。

|类别（清理前）|GiB|
|---|---:|
|installed_runtime_retained|4.5892|
|protected_frozen_restore_checkpoints_data_and_audits|3.5591|
|retained_offline_install_wheel|2.6655|
|checkpoint_keep_manual_review_only|1.8427|
|temporary_download_or_export_review|1.0637|
|protected_transfer_archive_and_receipts|1.0362|
|prediction_keep_evaluation_or_manual_review|0.2707|
|ordinary_python_or_test_cache|0.1770|
|download_cache_retained_for_reproducibility|0.1178|
|experiment_output_retained|0.0437|
|source_or_version_history_retained|0.0069|
|protected_audit_contract_monitor_or_stop|0.0006|

## 实际清理

先生成并验证dry-run，确认无workspace Python进程及文件占用后删除 **10,173个文件，逻辑大小1,324,621,340字节**（1.234 GiB）。
清理瞬间实测G盘空闲增加 1,345,335,296字节（1.253 GiB）；模型重载会重新生成部分pyc，磁盘空闲也可能受其他写入影响，故与逻辑删除量分别报告。
重载与最终盘点后，较清理前净空闲增加 1,278,648,320字节（1.191 GiB）。

|删除类别|文件数|逻辑字节|用途与可重建性|
|---|---:|---:|---|
|cache_directory|10172|190062108|仅Python/pytest/ruff临时缓存，可由导入/测试重建|
|abandoned_partial_wheel|1|1134559232|不完整pip下载，保留完整同版本wheel及校验记录|

完整Torch离线安装wheel保留并匹配历史SHA256；pip HTTP缓存保留，便于复现依赖。恢复树和原迁移包均未清理。没有删除任何模型权重、逐目标预测或错误实验目录。

## 人工确认清单：本轮全部保留

以下列出实验checkpoint；除关键20k/3,000、50k/3,000、50k/5,000外，其他整千步、CPU/GPU探针也不自动删除。大预测及历史错误运行仍有审计/复现价值。

|Checkpoint物理路径|MiB|用途/处理|
|---|---:|---|
|G:/yzx/linzhiyuce/restored_v04/phospholipid-structure-pretrain/outputs/pretraining_v04/runs/gine_smoke_3000/checkpoint/step_003000.pt|176.97|保留，后续删除需人工确认|
|G:/yzx/linzhiyuce/restored_v04/phospholipid-structure-pretrain/outputs/pretraining_v04/runs/fp_smoke_3000/checkpoint/step_003000.pt|188.69|保留，后续删除需人工确认|
|G:/yzx/linzhiyuce/phospholipid-structure-pretrain/outputs/polar_fp_base_50k_20261020/train_C_5000/checkpoint/step_004000.pt|188.83|保留，后续删除需人工确认|
|G:/yzx/linzhiyuce/phospholipid-structure-pretrain/outputs/polar_fp_base_50k_20261020/train_C_5000/checkpoint/step_005000.pt|188.89|保留，后续删除需人工确认|
|G:/yzx/linzhiyuce/phospholipid-structure-pretrain/outputs/polar_fp_base_50k_20261020/train_B_3000/checkpoint/step_002000.pt|188.72|保留，后续删除需人工确认|
|G:/yzx/linzhiyuce/phospholipid-structure-pretrain/outputs/polar_fp_base_50k_20261020/train_B_3000/checkpoint/step_003000.pt|188.77|保留，后续删除需人工确认|
|G:/yzx/linzhiyuce/phospholipid-structure-pretrain/outputs/polar_fp_base_50k_20261020/train_A_1000/checkpoint/step_001000.pt|188.66|保留，后续删除需人工确认|
|G:/yzx/linzhiyuce/phospholipid-structure-pretrain/outputs/polar_fp_base_50k_20261020/gpu_probe_resume_25/checkpoint/step_000025.pt|188.61|保留，后续删除需人工确认|
|G:/yzx/linzhiyuce/phospholipid-structure-pretrain/outputs/polar_fp_base_50k_20261020/gpu_probe_20/checkpoint/step_000020.pt|188.61|保留，后续删除需人工确认|
|G:/yzx/linzhiyuce/phospholipid-structure-pretrain/outputs/polar_fp_base_50k_20261020/cpu_split_2/checkpoint/step_000002.pt|188.60|保留，后续删除需人工确认|
|G:/yzx/linzhiyuce/phospholipid-structure-pretrain/outputs/polar_fp_base_50k_20261020/cpu_resume_4/checkpoint/step_000004.pt|188.60|保留，后续删除需人工确认|
|G:/yzx/linzhiyuce/phospholipid-structure-pretrain/outputs/polar_fp_base_50k_20261020/cpu_continuous_4/checkpoint/step_000004.pt|188.60|保留，后续删除需人工确认|

|大型预测文件（保留）|MiB|
|---|---:|
|G:/yzx/linzhiyuce/phospholipid-structure-pretrain/outputs/polar_fp_base_50k_20261020/eval_similarity_3000_beam50_512/predictions_private.jsonl|15.06|
|G:/yzx/linzhiyuce/phospholipid-structure-pretrain/outputs/polar_fp_base_50k_20261020/eval_C_5000_beam50_512/predictions_private.jsonl|19.06|
|G:/yzx/linzhiyuce/phospholipid-structure-pretrain/outputs/polar_fp_base_50k_20261020/eval_C_5000_beam50_512/raw_beams_private.jsonl|5.02|
|G:/yzx/linzhiyuce/phospholipid-structure-pretrain/outputs/polar_fp_base_50k_20261020/eval_C_5000_beam20_512/predictions_private.jsonl|6.97|
|G:/yzx/linzhiyuce/phospholipid-structure-pretrain/outputs/polar_fp_base_50k_20261020/eval_B_3000_beam20_512/predictions_private.jsonl|5.37|
|G:/yzx/linzhiyuce/phospholipid-structure-pretrain/outputs/formula_constrained_v04_rtx4070_20261020_r1/graph_raw_b50_s32/predictions_private.jsonl|9.36|
|G:/yzx/linzhiyuce/phospholipid-structure-pretrain/outputs/formula_constrained_v04_rtx4070_20261020_r1/graph_raw_b50_s256/predictions_private.jsonl|10.56|
|G:/yzx/linzhiyuce/phospholipid-structure-pretrain/outputs/formula_constrained_v04_rtx4070_20261020_r1/graph_elements_grammar_b50_s32/predictions_private.jsonl|7.62|
|G:/yzx/linzhiyuce/phospholipid-structure-pretrain/outputs/formula_constrained_v04_rtx4070_20261020_r1/graph_elements_grammar_b50_s256/predictions_private.jsonl|7.85|
|G:/yzx/linzhiyuce/phospholipid-structure-pretrain/outputs/formula_constrained_v04_rtx4070_20261020_r1/graph_elements_b50_s32/predictions_private.jsonl|9.86|
|G:/yzx/linzhiyuce/phospholipid-structure-pretrain/outputs/formula_constrained_v04_rtx4070_20261020_r1/graph_elements_b50_s256/predictions_private.jsonl|10.96|
|G:/yzx/linzhiyuce/phospholipid-structure-pretrain/outputs/formula_constrained_v04_rtx4070_20261020_r1/fingerprint_raw_b50_s256/predictions_private.jsonl|10.58|
|G:/yzx/linzhiyuce/phospholipid-structure-pretrain/outputs/formula_constrained_v04_rtx4070_20261020_r1/fingerprint_elements_grammar_b50_s32/predictions_private.jsonl|8.02|
|G:/yzx/linzhiyuce/phospholipid-structure-pretrain/outputs/formula_constrained_v04_rtx4070_20261020_r1/fingerprint_elements_grammar_b50_s256/predictions_private.jsonl|8.62|
|G:/yzx/linzhiyuce/phospholipid-structure-pretrain/outputs/formula_constrained_v04_rtx4070_20261020_r1/fingerprint_elements_b50_s256/predictions_private.jsonl|10.99|
|G:/yzx/linzhiyuce/phospholipid-structure-pretrain/outputs/formula_constrained_v04_rtx4070_20261020/fingerprint_raw_b50_s32/predictions_private.jsonl|9.31|
|G:/yzx/linzhiyuce/phospholipid-structure-pretrain/outputs/formula_constrained_v04_rtx4070_20261020/fingerprint_elements_b50_s32/predictions_private.jsonl|9.89|

## 清理后验证

三模型在CPU严格加载成功，AdamW状态与步数/样本消费位置完整，没有执行optimizer step。五份50k正式checkpoint哈希与原报告一致，所有本轮评价源预测哈希一致；清理前后缺失的原有文件仅为授权缓存或已确认临时下载。
详细dry-run逐文件哈希、删除回执、完整占用清单及进程检查证据留在G盘本轮storage目录。
