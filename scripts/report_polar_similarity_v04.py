"""Publish aggregate similarity/storage reports; no private structures or target rows."""

import hashlib
import json
from pathlib import Path

ROOT = Path("outputs/polar_similarity_eval_v04_20261020")
OUT = Path("docs/results")
FIG = "POLAR_PRETRAIN_SIMILARITY_EVAL_V04_FIGURES"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write(path, text):
    path.write_text(text, encoding="utf-8", newline="\n")


def rate(value):
    fraction = value["fraction"]
    return (
        f"{value['count']}/{value['denominator']} ({100 * fraction:.2f}%)"
        if fraction is not None
        else "0/0 (—)"
    )


def storage():
    folder = ROOT / "storage"
    before = read(folder / "summary_before_cleanup.json")
    after = read(folder / "summary_after_cleanup.json")
    receipt = read(folder / "cleanup_receipt.json")
    inventory = read(folder / "inventory_before_cleanup_private.json")
    after_inventory = read(folder / "inventory_after_cleanup_private.json")
    paths = {x["path"] for x in after_inventory}
    dry = read(folder / "cleanup_dry_run_private.json")
    removed = [f for t in dry["targets"] for f in t["files"]]
    # Post-load imports may regenerate some pyc paths; distinguish these from lost assets.
    missing = [f for f in inventory if f["path"] not in paths]
    assert all(
        f["category"] in ("ordinary_python_or_test_cache", "temporary_download_or_export_review")
        for f in missing
    )
    checkpoints = [
        f
        for f in inventory
        if Path(f["path"]).suffix in (".pt", ".pth", ".ckpt") and "outputs" in Path(f["path"]).parts
    ]
    for f in checkpoints:
        assert Path(f["path"]).exists() and Path(f["path"]).stat().st_size == f["bytes"]
    model_before = read(ROOT / "checkpoint_load_before_cleanup.json")
    model_after = read(ROOT / "checkpoint_load_after_cleanup.json")
    assert model_before == model_after
    predictions = read(ROOT / "aggregate.json")["source_predictions_sha256"]
    assert all(digest(Path(p)) == sha for p, sha in predictions.items())
    protected = read(Path("docs/results/POLAR_FP_BASE_50K_RESULTS.json"))["checkpoints"]
    for path in Path("outputs/polar_fp_base_50k_20261020").glob("train_*/checkpoint/step_*.pt"):
        step = str(int(path.stem.split("_")[1]))
        assert digest(path) == protected[step]["sha256"]
    return {
        "before": before,
        "after": after,
        "cleanup_receipt": receipt,
        "dry_run_manifest_sha256": digest(folder / "cleanup_dry_run_private.json"),
        "removed_by_kind": {
            kind: {
                "files": sum(len(t["files"]) for t in dry["targets"] if t["kind"] == kind),
                "logical_bytes": sum(
                    f["bytes"] for t in dry["targets"] if t["kind"] == kind for f in t["files"]
                ),
            }
            for kind in ("cache_directory", "abandoned_partial_wheel")
        },
        "manual_confirmation_checkpoints_retained": checkpoints,
        "large_predictions_retained": [
            f for f in inventory if "prediction_keep" in f["category"] and f["bytes"] >= 5_000_000
        ],
        "all_three_models_and_AdamW_CPU_reloaded": model_after,
        "five_formal_checkpoint_hashes_unchanged": True,
        "all_evaluation_prediction_hashes_unchanged": True,
        "all_missing_preexisting_files_are_authorized_cache_or_temp": True,
        "net_observed_drive_free_increase_after_model_reload": after["drive_free_bytes"]
        - receipt["free_bytes_before"],
        "deleted_file_snapshot_count": len(removed),
    }


def storage_markdown(s):
    receipt = s["cleanup_receipt"]
    lines = [
        "# G盘实验文件占用与安全清理审计",
        "",
        "扫描G:/yzx/linzhiyuce的项目、恢复副本、迁移包、独立环境和缓存。"
        "按物理目录扫描并跳过junction；文件大小为逻辑字节，不冒充NTFS分配块大小。",
        "所有checkpoint、数据、预测、配置、审计、故障日志与停止锁保留。",
        "",
        "|类别（清理前）|GiB|",
        "|---|---:|",
    ]
    for key, value in s["before"]["category_bytes"].items():
        lines.append(f"|{key}|{value / 2**30:.4f}|")
    lines += [
        "",
        "## 实际清理",
        "",
        f"先生成并验证dry-run，确认无workspace Python进程及文件占用后删除"
        f" **{receipt['deleted_files']:,}个文件，逻辑大小{receipt['deleted_logical_bytes']:,}字节**"
        f"（{receipt['deleted_logical_bytes'] / 2**30:.3f} GiB）。",
        f"清理瞬间实测G盘空闲增加 {receipt['observed_drive_free_bytes_increase']:,}字节"
        f"（{receipt['observed_drive_free_bytes_increase'] / 2**30:.3f} GiB）；"
        "模型重载会重新生成部分pyc，磁盘空闲也可能受其他写入影响，故与逻辑删除量分别报告。",
        f"重载与最终盘点后，较清理前净空闲增加"
        f" {s['net_observed_drive_free_increase_after_model_reload']:,}字节"
        f"（{s['net_observed_drive_free_increase_after_model_reload'] / 2**30:.3f} GiB）。",
        "",
        "|删除类别|文件数|逻辑字节|用途与可重建性|",
        "|---|---:|---:|---|",
    ]
    for key, value in s["removed_by_kind"].items():
        reason = (
            "仅Python/pytest/ruff临时缓存，可由导入/测试重建"
            if key == "cache_directory"
            else "不完整pip下载，保留完整同版本wheel及校验记录"
        )
        lines.append(f"|{key}|{value['files']}|{value['logical_bytes']}|{reason}|")
    lines += [
        "",
        "完整Torch离线安装wheel保留并匹配历史SHA256；pip HTTP缓存保留，便于复现依赖。"
        "恢复树和原迁移包均未清理。没有删除任何模型权重、逐目标预测或错误实验目录。",
        "",
        "## 人工确认清单：本轮全部保留",
        "",
        "以下列出实验checkpoint；除关键20k/3,000、50k/3,000、50k/5,000外，"
        "其他整千步、CPU/GPU探针也不自动删除。大预测及历史错误运行仍有审计/复现价值。",
        "",
        "|Checkpoint物理路径|MiB|用途/处理|",
        "|---|---:|---|",
    ]
    for f in s["manual_confirmation_checkpoints_retained"]:
        lines.append(
            f"|{f['path'].replace(chr(92), '/')}|{f['bytes'] / 2**20:.2f}|保留，后续删除需人工确认|"
        )
    lines += ["", "|大型预测文件（保留）|MiB|", "|---|---:|"]
    for f in s["large_predictions_retained"]:
        lines.append(f"|{f['path'].replace(chr(92), '/')}|{f['bytes'] / 2**20:.2f}|")
    lines += [
        "",
        "## 清理后验证",
        "",
        "三模型在CPU严格加载成功，AdamW状态与步数/样本消费位置完整，没有执行optimizer step。"
        "五份50k正式checkpoint哈希与原报告一致，所有本轮评价源预测哈希一致；"
        "清理前后缺失的原有文件仅为授权缓存或已确认临时下载。",
        "详细dry-run逐文件哈希、删除回执、完整占用清单及进程检查证据留在G盘本轮storage目录。",
        "",
    ]
    return "\n".join(lines)


def markdown(r):
    c = r["cases"]
    main = c["50k_5000_b50"]["summary"]
    lines = [
        "# v0.4-FP：结构相似度与参考模板收益评价",
        "",
        "**结论：存在局部条件生成信号，但整体尚未稳定超过结构参考，暂不支持盲目扩大预训练。**",
        f"参考Top1平均相似度{main['template_top1']['mean']:.4f}，"
        f"最佳参考模板平均相似度{main['template_best']['mean']:.4f}；"
        "50k/5,000 Beam50的Top10最大相似度均值"
        f"{main['all_targets']['10']['similarity']['mean']:.4f}。"
        f"Top10正收益为{rate(main['all_targets']['10']['delta_vs_template_best']['positive'])}，"
        f"平均Delta={main['all_targets']['10']['delta_vs_template_best']['mean']:+.4f}。",
        "",
        "使用冻结且反复使用的512 validation诊断目标，structure-oracle检索、可信式和外部给定链；"
        "不是独立泛化测试或MS/MS鉴定评价。模型/标签/架构/连接规则未改，没有重新训练。",
        "复用五组结果，仅补充50k/3,000 Beam50一次受监控推断。六组共3,072目标次，"
        "14,928个严格去重候选再次通过原Builder、连接、Polar含H式和完整中性式检查。",
        "K=3是模板容量上限；沿用冻结检索结果，511个目标实际有3个模板，1个目标只有1个，"
        "其余位置按原流程mask。基线仅比较实际输入模板，不补检索、不复制补齐。",
        "",
        "## Top-N相似度与覆盖",
        "",
        "先按原模型分数截取严格候选Top-N，再计算与目标最大Tanimoto；不按目标重新排序。"
        "候选少于N用全部已有候选，无候选S_N=0。以下以全部512为分母。",
        "",
        "|模型/Beam|覆盖|S1均值|S3均值|S5均值|S10均值|全部严格候选最大值均数|",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for key, case in c.items():
        s = case["summary"]
        vals = "|".join(
            f"{s['all_targets'][k]['similarity']['mean']:.4f}" for k in ("1", "3", "5", "10")
        )
        lines.append(
            f"|{key}|{rate(s['coverage'])}|{vals}|{s['all_targets']['all_strict_candidates']['similarity']['mean']:.4f}|"
        )
    lines += [
        "",
        "## 50k/5,000 Beam50累计阈值",
        "",
        "|Top-N|>=0.5|>=0.6|>=0.7|>=0.8|>=0.9|=1.0|",
        "|---|---|---|---|---|---|---|",
    ]
    for n in ("1", "3", "5", "10"):
        values = "|".join(
            rate(main["all_targets"][n]["thresholds"][t])
            for t in ("0.5", "0.6", "0.7", "0.8", "0.9", "1.0")
        )
        lines.append(f"|{n}|{values}|")
    lines += [
        "",
        "Tanimoto=1与canonical图Exact分别计算；本配置的指纹相同不在定义上保证完整图相同。",
        "",
        "|统计范围|Top-N|相似度均值|Delta均值|Delta中位数|正收益|零收益|负收益|",
        "|---|---:|---:|---:|---:|---|---|---|",
    ]
    for scope in ("all_targets", "covered_targets"):
        for n in ("1", "5", "10"):
            v = main[scope][n]
            d = v["delta_vs_template_best"]
            lines.append(
                f"|{scope}|{n}|{v['similarity']['mean']:.4f}|{d['mean']:+.4f}|{d['median']:+.4f}|"
                f"{rate(d['positive'])}|{rate(d['zero'])}|{rate(d['negative'])}|"
            )
    lines += [
        "",
        "全部六组的条件阈值、均值/中位数/分位数、Delta分布与配对重采样区间均在JSON。"
        "不将有候选目标的条件均值当成全512指标。",
        "",
        "## 相对模板的收益与分层",
        "",
        "template_best是在实际输入K个Polar中事后按目标取最大值，只是评价基线。"
        "模板允许不同式，生成候选严格匹配目标式；公式相同/不同与式L1差的分层也保存在JSON。",
        "最佳模板仅48/512（9.38%）与目标Polar式相同，464/512不同式，不能直接作为严格候选使用。"
        "两组Top10平均Delta分别为−0.4498与−0.1709；这区分了化学组成正确与结构接近，"
        "不把负Delta简单解释成完全没有学习公式条件。",
        "",
        "|模型/Beam|Top1正收益|Top5正收益|Top10正收益|Top10平均Delta|Top10 Delta中位数|",
        "|---|---|---|---|---:|---:|",
    ]
    for key, case in c.items():
        s = case["summary"]["all_targets"]
        gains = "|".join(rate(s[n]["delta_vs_template_best"]["positive"]) for n in ("1", "5", "10"))
        lines.append(
            f"|{key}|{gains}|{s['10']['delta_vs_template_best']['mean']:+.4f}|{s['10']['delta_vs_template_best']['median']:+.4f}|"
        )
    lines += [
        "",
        "最终50k/5,000 Beam50分层；完整Top1/3/5/10及条件统计在JSON：",
        "",
        "|分层|目标数|覆盖|S10均值|Delta10均值|正收益|Top10 Exact|",
        "|---|---:|---|---:|---:|---|---|",
    ]
    for key, s in c["50k_5000_b50"]["strata"].items():
        if not key.startswith(
            ("template_best:", "family:", "length:", "P_count:", "formula_TRAIN_frequency:")
        ):
            continue
        v = s["all_targets"]["10"]
        d = v["delta_vs_template_best"]
        lines.append(
            f"|{key}|{s['targets']}|{rate(s['coverage'])}|{v['similarity']['mean']:.4f}|"
            f"{d['mean']:+.4f}|{rate(d['positive'])}|{rate(v['exact'])}|"
        )
    low = c["50k_5000_b50"]["strata"]["template_best:low_<0.4"]["all_targets"]["10"][
        "delta_vs_template_best"
    ]
    lines += [
        "",
        f"低模板组Top10平均Delta={low['mean']:+.4f}，中位数{low['median']:+.4f}，"
        f"正收益{rate(low['positive'])}；均值诊断bootstrap区间{low['paired_diagnostic_bootstrap_mean_95_percentile']}跨0。"
        "因此不能声称低模板条件下已取得稳定平均收益。小样本分层只描述，不作显著性结论。",
        "收益集中在短头基：16–31 token组Top10平均Delta=+0.0725，70/112（62.50%）正收益；"
        "64–79 token组为−0.3860，仅18/120（15.00%）正收益。多磷酸家族正收益56/138（40.58%），"
        "高于磷酸酯/膦酸酯组约24%，但三个家族平均Delta仍均为负。P=3组正收益28/73（38.36%），"
        "仍不足以说明多P条件普遍优于模板。TRAIN未见分子式组仅14/87（16.09%）正收益。"
        "这些是探索性描述差异，未作多重校正的独立显著性验证。",
        "",
        "## 新颖性、核心保留与化学辅助指标",
        "",
        "TRAIN已见指冻结TRAIN bank的图集合，并不表示每个图都曾作为该模型训练目标。"
        "TRAIN未见不等于天然产物新发现；下表新颖候选均为已严格筛选子集，其合法性不能反推原始输出100%合法。",
        "",
        "|模型/Beam|严格候选数|模板复制|TRAIN未见|新颖候选目标T均值|新颖候选T>=0.7|",
        "|---|---:|---|---|---:|---|",
    ]
    for key, case in c.items():
        a = case["auxiliary"]
        lines.append(
            f"|{key}|{a['strict_unique_candidates']}|{rate(a['candidate_template_copy'])}|"
            f"{rate(a['candidate_novel_to_full_TRAIN'])}|{case['novel_candidate_similarity']['mean']:.4f}|"
            f"{rate(case['novel_candidate_thresholds']['0.7'])}|"
        )
    lines += [
        "",
        "三方相似关系、候选相对模板/目标的完整分布、共同P核心保留及官能团等变化的计数见JSON。"
        "共同核心采用连接P二跳邻域一致的保守描述代理；不是反应路线或合成/生物真实性判据。"
        "P邻域、官能团/距离、末端、分支、环差异可重叠，未强制归为单一化学错误。",
        "最终Beam50有2,215个候选机会对应参考与目标共同的连接P邻域；其中1,024（46.23%）保持该核心，"
        "992（44.79%）同时改变其他图特征，149（6.73%）还取得正相似度收益。"
        "这是候选级/共同核心机会分母，不能当作512个目标的成功比例。",
        "",
        "|模型/Beam|Raw RDKit有效|Raw Polar有效|Polar式覆盖|完整中性式覆盖|联合严格覆盖|",
        "|---|---|---|---|---|---|",
    ]
    for key, case in c.items():
        a = case["auxiliary"]
        lines.append(
            f"|{key}|"
            + "|".join(
                rate(a[k])
                for k in (
                    "raw_RDKit_valid",
                    "raw_Polar_chemistry_valid",
                    "target_Polar_formula_coverage",
                    "target_full_neutral_formula_coverage",
                    "strict_joint_formula_builder_coverage",
                )
            )
            + "|"
        )
    lines += [
        "",
        "Raw有效率以原始搜索候选为分母，沿原流程要求EOS；式覆盖以目标为分母。"
        "Raw中可能重复，严格候选canonical图去重；二者不混淆。完整中性式匹配候选率也在JSON单独记录。",
        "",
        "|模型/Beam|Exact Top1|Exact Top5|Exact Top10|无候选|目标进Beam|"
        "目标进Beam但排10名后|合法异构体均数|",
        "|---|---|---|---|---|---|---|---:|",
    ]
    for key, case in c.items():
        a = case["auxiliary"]
        exact = "|".join(rate(a["exact"][n]) for n in ("1", "5", "10"))
        lines.append(
            f"|{key}|{exact}|{rate(a['no_qualified_candidate'])}|{rate(a['true_graph_in_strict_beam'])}|"
            f"{rate(a['true_graph_present_below_rank10'])}|{a['legal_isomer_count']['mean']:.3f}|"
        )
    lines += [
        "",
        "## 图表",
        "",
        f"![Figure A]({FIG}/Figure_A_TopN_Cumulative.png)",
        "",
        f"![Figure B]({FIG}/Figure_B_Template_Generated_Beam50.png)",
        "",
        f"![Figure C]({FIG}/Figure_C_Template_Gain.png)",
        "",
        "另有Beam20配对图；公开PNG/SVG/PDF均不含结构或目标标识。",
        f"Figure D已从真实预测制作{r['private_case_count']}个事后案例，含改进、多P/长头基及失败。"
        "保留原排名，无候选处留空。结构图和真实式/相似度/差异标注仅在G盘"
        " `outputs/polar_similarity_eval_v04_20261020/private_Figure_D/`，未上传GitHub。",
        "",
        "## 判断与下一步",
        "",
        "1. 50k/5,000相对两个3,000步模型的相似度分布改善，但20k/3,000与50k/5,000同时改变池和步数；"
        "同步数的池效应较弱且随Beam不同，不能将联合改善全部归因于头基覆盖。",
        "2. 部分TRAIN未见且严格合法的生成图超过了最佳模板，说明有局部生成潜力；"
        "但大多数目标Delta为负，低模板组均值收益接近0且区间跨0，尚不能声称稳健条件结构变换。",
        "3. 没有随机模板/无模板消融，正Delta也不能单独证明收益由模板信息因果驱动。"
        "模板结构使用、公式条件和Decoder先验的贡献仍需控制实验。",
        "4. 暂不盲目扩大预训练池或堆叠网络。先评估模板利用的匹配随机模板/无模板对照，"
        "再开展有真实谱图、严格划分的小规模MS/MS条件实验及预训练vs随机初始化比较。"
        "若核心变换仍弱，再独立设计编辑模型对照，本轮未改架构。",
        "5. checkpoint可作为后续初始化候选，但结构相似度、正确式与真实MS/MS鉴定是不同层次。"
        "本轮全为structure-oracle有利模板条件，未提供真实谱图鉴定性能证据。",
        "",
        "## 后续控制方案：未执行",
        "",
        "A：20k/3,000权重、AdamW、RNG字段和消费位置均通过CPU静态加载，具备进一步适配条件。"
        "旧训练入口仍有ROCm门槛，不能用拒绝旧契约的FP50k入口直接续训。未来应建独立native20k控制入口，"
        "保留旧3000及数据/配置契约，先完成受监控的真实反向/恢复门槛，再追加2000步至5000，不能宣称已验证可直接安全续训。",
        "本机相同架构额外2000步实测约92–95秒（含阶段验证/保存；不含加载和新安全门槛）；"
        "Beam20约5分钟、Beam50约10–18分钟。以20–30分钟级窗口和1–2GiB新增输出预算规划，"
        "显存参考PyTorch峰值约535MiB、主进程RAM约2.4GiB，额外驱动/worker开销仍需监控。",
        "B：真实MS/MS上固定相同划分、优化预算和检索流程，比较预训练与随机初始化；"
        "正常/随机模板均TRAIN-only且排除目标，随机对照尽量匹配链型/公式差值分布。"
        "无模板对照需未来明确定义合法占位/TCLS-only输入，不能把当前缺模板报错当实验。"
        "记录目标结构是否在预训练池中出现；不再用真实目标图帮助谱图阶段检索。",
        "C：若上述消融证明模板利用不足，再单独设计保留连接P核心的编辑式生成对照。"
        "这些控制、额外训练与MS/MS微调本轮均未启动。",
        "",
        "## 清理和复现",
        "",
        f"删除{r['storage']['cleanup_receipt']['deleted_files']:,}个授权缓存/临时文件，"
        f"逻辑大小{r['storage']['cleanup_receipt']['deleted_logical_bytes'] / 2**30:.3f}GiB；"
        "清理时磁盘空闲实测增加"
        f"{r['storage']['cleanup_receipt']['observed_drive_free_bytes_increase'] / 2**30:.3f}GiB。"
        "所有checkpoint及评价源预测保留并复核哈希；三模型CPU重载通过。",
        "详见 [磁盘审计](EXPERIMENT_STORAGE_AUDIT.md) 与本报告JSON。",
        "372项测试通过；核心CUDA环境版本不变；补充推断无新可见GPU故障，WATCHDOG目录权限限制照实保留。",
        "代码/输入/预测/图表哈希、配对比较和全部分层见JSON；最终commit SHA由交付回执提供。",
        "",
    ]
    return "\n".join(lines)


def main():
    r = read(ROOT / "aggregate.json")
    assert r["complete"] and len(r["cases"]) == 6
    for name, sha in r["analysis_source_sha256"].items():
        assert digest(Path(name)) == sha
    r["storage"] = storage()
    for case in r["cases"].values():
        a = case["auxiliary"]
        denominator = a["strict_unique_candidates"] - a["true_graph_in_strict_beam"]["count"]
        case["wrong_candidate_graph_difference_rates"] = {
            name: {
                "count": int(count),
                "denominator": denominator,
                "fraction": count / denominator if denominator else None,
            }
            for name, count in case["wrong_candidate_graph_difference_flags"].items()
        }
        core = case["shared_core_descriptive_proxy"]
        opportunities = core.get("shared_reference_target_rooted_P_core_candidate_opportunities", 0)
        case["shared_core_descriptive_proxy_rates"] = {
            key: {
                "count": count,
                "denominator": opportunities,
                "fraction": count / opportunities if opportunities else None,
            }
            for key, count in core.items()
            if key != "shared_reference_target_rooted_P_core_candidate_opportunities"
        }
    r["environment"] = read(ROOT / "analysis_environment.json")
    r["private_case_count"] = len(read(ROOT / "private_Figure_D/case_manifest_private.json"))
    r["figure_sha256"] = read(OUT / FIG / "FIGURE_SHA256.json")
    host = read(
        Path("outputs/polar_fp_base_50k_20261020/eval_similarity_3000_beam50_512/host_monitor.json")
    )
    assert host["stop_reason"] is None and host["protected_checkpoints_unchanged"]
    r["only_supplemental_gpu_job"] = host
    j = OUT / "POLAR_PRETRAIN_SIMILARITY_EVAL_V04_RESULTS.json"
    md = OUT / "POLAR_PRETRAIN_SIMILARITY_EVAL_V04_RESULTS.md"
    write(j, json.dumps(r, ensure_ascii=False, indent=2) + "\n")
    write(md, markdown(r))
    write(OUT / "EXPERIMENT_STORAGE_AUDIT.md", storage_markdown(r["storage"]))
    write(
        OUT / "EXPERIMENT_STORAGE_AUDIT.json",
        json.dumps(r["storage"], ensure_ascii=False, indent=2) + "\n",
    )
    write(
        OUT / "POLAR_PRETRAIN_SIMILARITY_EVAL_V04_SHA256.json",
        json.dumps(
            {
                p.as_posix(): digest(p)
                for p in (
                    j,
                    md,
                    OUT / "EXPERIMENT_STORAGE_AUDIT.md",
                    OUT / "EXPERIMENT_STORAGE_AUDIT.json",
                )
            },
            indent=2,
        )
        + "\n",
    )
    print(
        json.dumps(
            {"reports_written": True, "models": 3, "cases": 6, "private_figures_published": False}
        )
    )


if __name__ == "__main__":
    main()
