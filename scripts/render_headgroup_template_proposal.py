"""Private visual review of integrated approved heads and nested template proposals."""

import argparse
import base64
import json
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[2]/'cache'/'matplotlib'))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

from build_headgroup_structure_review_v01 import depict
from structure_pretrain.reported_headgroups.single_anchor_review import analyze_single_anchor
from structure_pretrain.reported_headgroups.template_selection import digest


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_page(out, payload):
    template = Path("scripts/headgroup_template_proposal.html").read_text(encoding="utf-8")
    image = base64.b64encode((out / 'template_coverage_comparison.png').read_bytes()).decode()
    html = template.replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False).replace("</", "<\\/"))
    html = html.replace("__COVERAGE_PNG__", image)
    (out / "index.html").write_text(html, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/headgroup_template_proposal_20261022"))
    parser.add_argument("--html-only", action="store_true", help="Reuse checked drawings and only rebuild the page")
    args = parser.parse_args()
    out = args.output
    if args.html_only:
        payload = json.loads((out / 'authoring/workbook_payload_private.json').read_text(encoding='utf-8'))
        assert payload['proposal_input_sha256'] == digest(out/'new_template_proposals_ranked.csv')
        write_page(out, payload)
        return
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    coverage = json.loads((out / "coverage_summary.json").read_text(encoding="utf-8"))
    banks = json.loads((out / "proposed_banks_private.json").read_text(encoding="utf-8"))
    approved = json.loads((out / "approved_catalog_private.json").read_text(encoding="utf-8"))
    new = [r for r in banks["350"] if r['selection_status'] != 'human_approved']
    for folder in ("svg", "png", "authoring"):
        (out / folder).mkdir(exist_ok=True)
    rows = []
    for i, original in enumerate(new + approved):
        row = dict(original)
        row["kind"] = "new" if i < len(new) else "approved"
        row["display_name"] = row.get("display_name") or row.get("abbreviation") or row["full_name"]
        row["review_number"] = f"T{i+1:03d}" if row["kind"] == 'new' else f"K{i-len(new)+1:03d}"
        row["first_bank_size"] = row.get("first_bank_size", 0)
        row.update(analyze_single_anchor(row["polar_smiles"]))
        row.update(depict(row, out))
        rows.append(row)
    sizes = summary["proposed_sizes"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    colors = ["#8298a6", "#516db0", "#227e84", "#cc8b31", "#954f85"]
    fig, ax = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    for size, color in zip(sizes, colors):
        table = pq.read_table(out / f"target_coverage_bank_{size}_private.parquet").to_pydict()
        similarities = np.sort(table["nearest_tanimoto"])
        ax[0, 0].plot(similarities, np.arange(1, len(similarities)+1)/len(similarities), label=str(size), color=color)
        distances = np.sort(table["minimum_heavy_element_L1"])
        ax[0, 1].plot(distances, np.arange(1, len(distances)+1)/len(distances), label=str(size), color=color)
    ax[0, 0].set(xlabel="Nearest template Morgan Tanimoto", ylabel="Fraction of 195,306 target cores", title="Structural distance distribution", xlim=(0,1), ylim=(0,1))
    ax[0, 0].legend(title="Bank size")
    ax[0, 1].set(xlabel="Minimum heavy-element count L1 difference", ylabel="Cumulative target fraction", title="Formula composition difference", xlim=(0,20), ylim=(0,1))
    ax[1, 0].plot(sizes, [coverage[str(n)]["fixed_unseen_targets"]['fraction_at_least_0.5']*100 for n in sizes], marker="o", label="Tanimoto >= 0.5", color="#227e84")
    ax[1, 0].plot(sizes, [coverage[str(n)]["fixed_unseen_targets"]['fraction_at_least_0.7']*100 for n in sizes], marker="o", label="Tanimoto >= 0.7", color="#516db0")
    ax[1, 0].set(xlabel="Template bank size", ylabel="Covered target cores (%)", title="Fixed target population outside all proposed banks")
    ax[1, 0].legend()
    for n, color in zip(sizes, colors):
        if n == sizes[0]:
            values = [r['average_mass_Da'] for r in banks[str(n)]]
        else:
            values = [r['average_mass_Da'] for r in banks[str(n)] if r['selection_status'] != 'human_approved']
        ax[1, 1].hist(values, bins=np.arange(0, 751, 75), histtype="step", label=f"{n} ({'known' if n == sizes[0] else 'new only'})", color=color)
    ax[1, 1].set(xlabel="Open Polar average mass (Da)", ylabel="Templates", title="Template mass distribution (known >750 Da omitted)")
    ax[1, 1].legend(fontsize=8)
    for a in ax.flat:
        a.grid(alpha=.18)
    for ext in ('png', 'svg'):
        fig.savefig(out / f"template_coverage_comparison.{ext}", dpi=160)
    plt.close(fig)
    graph_hash = digest(out / "new_template_proposals_ranked.csv")
    payload = {"summary": summary, "coverage": coverage, "rows": rows,
               "proposal_input_sha256": graph_hash}
    write_page(out, payload)
    dump(out / "authoring/workbook_payload_private.json", payload)
    dump(out / "visual_validation.json", {"graphs": len(rows), "svg": len(list((out/'svg').glob('*.svg'))),
          "png": len(list((out/'png').glob('*.png'))), "new_unapproved": len(new),
          "known_kept": len(approved), "source_binding_sha256": graph_hash})
    print(json.dumps({"drawn": len(rows), "new": len(new), "known": len(approved)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
