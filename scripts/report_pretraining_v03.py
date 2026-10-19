"""Export aggregate-only real v0.3 audit and smoke results; never sample predictions."""

import argparse
import json
from pathlib import Path

from structure_pretrain.data.prepare_v03 import read_json


def report(prepared, smoke, destination):
    prepared, smoke, destination = map(Path, (prepared, smoke, destination))
    destination.mkdir(parents=True, exist_ok=True)
    labels = read_json(prepared / "label_audit.json")
    tokens = read_json(prepared / "tokenization_audit.json")
    split = read_json(prepared / "split_audit.json")
    reconstruction = read_json(prepared / "reconstruction_audit.json")
    coverage = read_json(prepared / "retrieval_coverage_by_split.json")
    models = {}
    for name in ("fingerprint", "graph"):
        path = smoke / name
        training = read_json(path / "training_metrics.json")
        models[name] = {
            k: training[k]
            for k in (
                "optimizer_steps_total",
                "optimizer_steps_this_run",
                "resumed_from_step",
                "train_targets",
                "train_examples_consumed",
                "unique_train_targets_seen",
                "validation_targets",
                "test_targets_used_for_training_or_tuning",
                "validation_teacher_forced",
                "gradient_checks",
                "parameters_changed",
                "checkpoint_reload_identical_logits",
                "parameters",
                "module_parameters",
                "shapes",
            )
        }
        models[name].update(
            {
                "first_step_loss": training["train_history"][0],
                "last_step_loss": training["train_history"][-1],
                "generation": read_json(path / "generation_metrics.json"),
                "resources": read_json(path / "resource_metrics.json"),
            }
        )
        initial_resources = path / "initial_run/resource_metrics.json"
        if initial_resources.exists():
            models[name]["resume_resources"] = models[name]["resources"]
            models[name]["resources"] = read_json(initial_resources)
            models[name]["resource_measurement"] = (
                "initial 10-step run; resume resource metrics separate"
            )
    result = {
        "schema": "pretraining-v03-aggregate-results",
        "version": "v0.3",
        "structures_sha256": labels["structures_sha256"],
        "model": {
            "outputs": ["chain_2_canonical_slots", "P_containing_polar_fragment"],
            "dimensions": {"structure": 256, "formula": 128, "fusion": 512},
            "loss_weights": {"chain": 1, "polar": 1},
            "removed": [
                "ArchitectureHead",
                "four_chain_slots",
                "formula_available",
                "similarity_neural_input",
                "missing_formula_substitution",
            ],
        },
        "labels": {
            k: labels[k]
            for k in (
                "rows_checked",
                "valid_polar_ids",
                "valid_canonical_polar_graphs",
                "invalid_labels",
                "used_chain_counts",
            )
        },
        "tokenizer": {
            k: tokens[k]
            for k in (
                "oov",
                "truncated",
                "roundtrip_failures",
                "length",
                "max_polar_length",
                "vocabulary_size",
                "bracket_tokens",
            )
        },
        "split": {
            k: split[k]
            for k in (
                "seed",
                "split_counts",
                "intersections",
                "cross_split_scaffolds",
                "analog_parent_edges_by_split",
            )
        },
        "reconstruction": {
            k: reconstruction[k]
            for k in (
                "sample_count",
                "special_structures_checked",
                "all_special_structures",
                "assembly_success_rate",
                "canonical_graph_match_rate",
                "formula_match_rate",
            )
        },
        "retrieval": {
            k: coverage[k]
            for k in (
                "train_structures",
                "train_fp_groups",
                "held_out_reference_rows",
                "threshold",
                "k",
                "approximate_retrieval",
                "candidate_groups",
                "search_calibration",
                "splits",
                "ann_note",
            )
        },
        "retrieval_full_bank_exact_sample": coverage["exact_full_train_bank_benchmark"],
        "exact_low_count_results": {
            "confirmations": len(coverage["exact_low_count_confirmations"]),
            "result_counts": [x["count"] for x in coverage["exact_low_count_confirmations"]],
        },
        "head_difficulty": read_json(prepared / "head_difficulty_audit.json")["results"],
        "retrieval_integrity": read_json(prepared / "retrieval_integrity_audit.json"),
        "smoke_sampling": read_json(prepared / "smoke_sampling_audit.json")["splits"],
        "models": models,
        "gpu_verification": "pending execution on RX9070XT/WSL/ROCm",
        "full_pretraining_executed": False,
        "msms_finetuning_executed": False,
        "scope": "real-data engineering smoke, structure-oracle retrieval; not MS/MS accuracy",
    }
    (destination / "PRETRAINING_V03_RESULTS.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# v0.3 真实结构预训练接入与本机 smoke 结果",
        "",
        "本轮只执行真实结构数据的工程验证。GPU验证待运行，没有完整多轮预训练或MS/MS微调。",
        "",
        "1. **文件和保留模块**：新增全库准备、train-only检索、lazy Dataset、"
        "真实训练/自由生成/迁移核验入口。保留3层GINE(128内部宽度、7/3特征、sum pooling)"
        "或Morgan MLP、2层模板集合Transformer与2层polar decoder。",
        "2. **两个输出**：Chain Head `[B,2,323]` 与含P Polar Decoder `[B,L−1,63]`。"
        "ArchitectureHead及四槽路径已移除；槽位不代表sn。",
        "3. **融合**：结构256D、公式/差值20→128D、concat384→512D已实现。",
        "4. **公式/相似度**：中性式必填、target−template有符号差值；缺失直接失败。"
        "模型不接收formula_available或Tanimoto；分数留在检索cache，统计仅用合法train pairs。",
        f"5. **标签/tokenizer**：{labels['rows_checked']:,}行、"
        f"{labels['valid_polar_ids']:,}个有效polar，"
        f"OOV={tokens['oov']}，截断={tokens['truncated']}，max={tokens['max_polar_length']}（含BOS/EOS）。"
        "长度min/P50/P90/P95/P99/max为12/45/66/69/74/84，63词。",
        "6. **冻结划分**：",
        "",
        "|split|polar groups|完整结构|",
        "|---|---:|---:|",
    ]
    for name, counts in split["split_counts"].items():
        lines.append(f"|{name}|{counts['polar_groups']:,}|{counts['structures']:,}|")
    lines += [
        "",
        "三对split的polar ID与canonical polar graph交集全部为0。Test不用于训练/调参。"
        "跨组analog/scaffold关系作为泛化难度单独报告。",
        "",
        "7. **合法检索**：仅train bank，普通允许同头基，"
        "cross排除相同canonical polar graph及aliases。"
        "Morgan参数保持radius2/2048/无chirality、阈值.4/K≤10。下表覆盖分母包含全部目标：",
        "",
        "|mode/split|targets|≥1|≥5|10|zero|同头基template ratio|top1中位数|",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, splits in coverage["splits"].items():
        for name, values in splits.items():
            # summarize_metrics uses one consistent naming scheme across retrieval reports.
            five = round(values["at_least_5_fraction"] * values["targets"])
            ten = round(values["ten_fraction"] * values["targets"])
            lines.append(
                f"|{mode}/{name}|{values['targets']:,}|{values['at_least_1']:,}|"
                f"{five:,}|{ten:,}|"
                f"{values['zero_template_targets']:,}|{values['same_polar_template_ratio']:.6f}|"
                f"{values['max_similarity']['quantiles']['0.5']:.6f}|"
            )
    lines += [
        "",
        "ANN候选内RDKit精确重算；不足10均对全train bank精确回查。"
        "另作固定全bank精确样本benchmark，不将近似HNSW称为全库穷举TopK。详细分布见聚合JSON。",
        "",
        f"8. **标签重建**：{reconstruction['sample_count']}条，"
        f"包括全部{reconstruction['all_special_structures']}条特殊结构，"
        "组装、canonical完整图和分子式全部100%匹配。此为标签核验，不是模型预测率。",
        "",
        "9. **本机真实训练**：每种encoder使用512 train/128 validation，先FP后GINE，"
        "含保存、重载和实际恢复更新。以下为工程检查：",
        "",
        "|encoder|实际累计steps|first→last train loss|val loss|greedy有效polar|EOS|peak RAM GiB|",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for name, metrics in models.items():
        gen, resources = metrics["generation"], metrics["resources"]
        lines.append(
            f"|{name}|{metrics['optimizer_steps_total']}|{metrics['first_step_loss']['loss']:.4f}→"
            f"{metrics['last_step_loss']['loss']:.4f}|{metrics['validation_teacher_forced']['loss']:.4f}|"
            f"{gen['valid_polar_rate']:.4f}|{gen['eos_rate']:.4f}|"
            f"{resources['peak_RAM_bytes'] / 2**30:.3f}|"
        )
    lines += [
        "",
        "自由生成固定16个validation目标，greedy和beam5从BOS起步，按原始EOS/长度状态评价；"
        "未修复或后验筛式。Teacher forcing、Top5/链/完整结构指标及吞吐详见JSON。"
        "10–11个优化步骤不足以评价科学效果，0合法率也如实保留。",
        "",
        "10. **GPU待运行**：HIP/GPU、GINE和Transformer backward、FP32/BF16/FP16、checkpoint恢复、"
        "VRAM/吞吐与20k/2k/1000-step真实GPU smoke，命令见[WSL_ROCm.md](WSL_ROCm.md)。",
        "",
        "11. **正式预训练门槛**：真实RX9070XT算子与GPU smoke尚未通过，这是目前必须完成的验收。"
        "标签/重建/无泄漏/合法bank/本机管线检查见上表。生成合法性与chain表现须在有意义的GPU训练后重新评价；"
        "beam长度策略、采样混合比例及吞吐为后续实验优化项，不能从tiny smoke推断最优设置。",
        "",
        "12. **执行边界**：未执行完整2,015,000结构库的多轮预训练，未执行MS/MS微调，"
        "未更改原库、双键代表、P链配额或冻结输入哈希。",
        "",
        "聚合数值：[PRETRAINING_V03_RESULTS.json](results/PRETRAINING_V03_RESULTS.json)。"
        "私有产物在`data/private/pretraining_v03/`与`outputs/pretraining_v03/local_smoke/`；"
        "结构、逐样本预测、索引和checkpoint不在GitHub。",
    ]
    (destination.parent / "PRETRAINING_V03_RESULTS.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", default="data/private/pretraining_v03")
    p.add_argument("--smoke", default="outputs/pretraining_v03/local_smoke")
    p.add_argument("--destination", default="docs/results")
    a = p.parse_args()
    report(a.prepared, a.smoke, a.destination)
