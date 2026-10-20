"""Figure D: real, posthoc-selected cases; private structures stay under outputs/."""

import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem import Draw

from structure_pretrain.data.prepare_v03 import atom_counts, canonical_polar
from structure_pretrain.library_builder.canonicalize import parse_molecule
from structure_pretrain.library_builder.schemas import ELEMENTS

ROOT = Path("outputs/polar_similarity_eval_v04_20261020")


def main():
    context = json.loads((ROOT / "template_context_private.json").read_text())
    rows = json.loads((ROOT / "target_metrics_private.json").read_text())["50k_5000_b50"]
    out = ROOT / "private_Figure_D"
    out.mkdir(exist_ok=True)
    used, selected = set(), []

    def pick(label, predicate, failure=False):
        options = []
        for i, (c, row) in enumerate(zip(context, rows)):
            if i in used:
                continue
            for pred in row["candidates"][:10]:
                if predicate(c, pred):
                    options.append((pred["gain"], i, pred))
        if options:
            options.sort(key=lambda x: (x[0], x[1], x[2]["rank"]))
            _, i, pred = options[len(options) // 2]
            used.add(i)
            selected.append(
                (label, i, pred, "median qualifying candidate gain; posthoc illustration")
            )

    pick(
        "low_template_improvement",
        lambda c, p: (
            c["template_best"] < 0.4 and p["novel"] and not p["exact"] and p["gain"] >= 0.1
        ),
    )
    pick(
        "Top1_improvement",
        lambda c, p: p["rank"] == 1 and p["novel"] and not p["exact"] and p["gain"] > 0,
    )
    pick(
        "multi_P_improvement",
        lambda c, p: c["P_count"] >= 2 and p["novel"] and not p["exact"] and p["gain"] > 0,
    )
    pick("long_head_improvement", lambda c, p: c["length"] >= 64 and p["novel"] and p["gain"] > 0)
    pick(
        "shared_P_core_modified",
        lambda c, p: (
            p["preserves_shared_rooted_P_core"]
            and p["modifies_other_features"]
            and p["gain"] > 0
            and not p["exact"]
        ),
    )
    pick(
        "high_template_failure",
        lambda c, p: c["template_best"] >= 0.7 and p["rank"] == 1 and p["gain"] < -0.2,
        failure=True,
    )
    missing = [i for i, r in enumerate(rows) if not r["candidate_count"] and i not in used]
    if missing:
        i = missing[len(missing) // 2]
        selected.append(("no_qualified_candidate", i, None, "middle cohort index among failures"))
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 21)
    records = []
    for number, (label, index, candidate, rule) in enumerate(selected, 1):
        c = context[index]
        ref = c["templates"][c["template_best_index"]]
        graph = candidate["graph"] if candidate else None
        texts = [ref, graph, c["target"]]
        molecules = [parse_molecule(s) if s else None for s in texts]
        for original, molecule in zip(texts, molecules):
            if original:
                assert canonical_polar(Chem.MolToSmiles(molecule)) == original
        formula = "".join(
            e + (str(int(n)) if n != 1 else "") for e, n in zip(ELEMENTS, c["formula"]) if n
        )
        ref_formula = "".join(
            e + (str(int(n)) if n != 1 else "")
            for e, n in zip(ELEMENTS, atom_counts(molecules[0]))
            if n
        )
        legends = [
            f"Best input ref #{c['template_best_index'] + 1}\n"
            f"T(ref,target)={c['template_best']:.4f}\nFormula: {ref_formula}",
            (
                f"Generated: original rank {candidate['rank']}\n"
                f"T(pred,target)={candidate['similarity']:.4f}; gain={candidate['gain']:+.4f}\n"
                f"Formula: {formula}"
            )
            if candidate
            else "No strictly qualified candidate\nGenerated similarity recorded as 0",
            f"Frozen target\nPolar formula: {formula} (dummy excluded)",
        ]
        options = Draw.MolDrawOptions()
        options.legendFontSize = 19
        options.annotationFontScale = 0.85
        image = Draw.MolsToGridImage(
            molecules, molsPerRow=3, subImgSize=(620, 480), legends=legends, drawOptions=options
        )
        canvas = Image.new("RGB", (image.width, image.height + 145), "white")
        canvas.paste(image, (0, 145))
        draw = ImageDraw.Draw(canvas)
        title = (
            f"Private Figure D{number}: {label} | 50k/5000, Beam50 | "
            f"target P={c['P_count']}, length={c['length']}"
        )
        draw.text((20, 12), title, fill="#122b43", font=font)
        flags = (
            ", ".join(candidate["difference_flags_vs_target"])
            if candidate
            else "no candidate to compare"
        )
        caption = "Graph feature differences vs target: " + (flags or "exact canonical graph")
        caption += (
            ". Posthoc case selection; original decoder rank retained. Not an MS/MS identification."
        )
        for line, y in zip(textwrap.wrap(caption, width=145), (48, 77, 106)):
            draw.text((20, y), line, fill="#333333", font=font)
        base = out / f"Figure_D{number}_{label}"
        canvas.save(base.with_suffix(".png"))
        svg = Draw.MolsToGridImage(
            molecules,
            molsPerRow=3,
            subImgSize=(620, 480),
            legends=legends,
            drawOptions=options,
            useSVG=True,
        )
        base.with_suffix(".svg").write_text(svg, encoding="utf-8")
        records.append(
            {
                "label": label,
                "cohort_index_private": index,
                "row_private": c["row"],
                "selection_rule": rule,
                "context": c,
                "candidate": candidate,
                "path": str(base.with_suffix(".png")),
            }
        )
    (out / "case_manifest_private.json").write_text(json.dumps(records, indent=2) + "\n")
    print(
        json.dumps(
            {
                "private_cases": len(records),
                "directory": str(out),
                "uploaded_to_public_figures": False,
            }
        )
    )


if __name__ == "__main__":
    main()
