from collections import Counter
from statistics import median

from .schemas import BuildConfig, PolarRule, Rejection


class ReportAccumulator:
    def __init__(self, polars: list[PolarRule]):
        self.per_polar = Counter({polar.polar_id: 0 for polar in polars})
        self.families, self.anchors, self.combos = Counter(), Counter(), Counter()
        self.carbon, self.db = Counter(), Counter()
        self.total, self.lyso, self.formula_missing = 0, 0, 0

    def add(self, row):
        self.total += 1
        self.per_polar[row["polar_id"]] += 1
        self.families[row["linkage_family"]] += 1
        self.combos[row["combination_id"]] += 1
        self.carbon[row["total_chain_carbon"]] += 1
        self.db[row["total_chain_double_bonds"]] += 1
        self.lyso += row["is_lyso"]
        self.formula_missing += not bool(row["neutral_formula"])
        if row["is_anchor_combination"]:
            self.anchors[row["combination_id"]] += 1

    def finish(
        self,
        config: BuildConfig,
        chains: int,
        combos: int,
        rejections: list[Rejection],
        duplicates: int,
    ):
        counts = list(self.per_polar.values())
        reasons = Counter(rejection.reason for rejection in rejections)
        return {
            "status": "complete",
            "schema_version": config.schema_version,
            "seed": config.seed,
            "training_performed": False,
            "max_structures": config.max_structures,
            "total_structures": self.total,
            "unique_canonical_smiles": self.total,
            "duplicates_removed": duplicates,
            "rejected_count": len(rejections),
            "number_of_chains": chains,
            "number_of_chain_combinations": combos,
            "number_of_polar_moieties": len(self.per_polar),
            "structures_per_polar_min": min(counts, default=0),
            "structures_per_polar_median": median(counts) if counts else 0,
            "structures_per_polar_max": max(counts, default=0),
            "structures_per_polar": dict(sorted(self.per_polar.items())),
            "AA_count": self.families["AA"],
            "OA_count": self.families["OA"],
            "PE_PA_count": self.families["PA"],
            "lyso_count": self.lyso,
            "bis_phosphatidyl_count": 0,
            "anchor_combo_counts": dict(sorted(self.anchors.items())),
            "total_carbon_distribution": dict(sorted(self.carbon.items())),
            "total_double_bond_distribution": dict(sorted(self.db.items())),
            "distribution_definition": (
                "Sum of imported chain annotations; excludes polar/scaffold carbons; "
                "P DB convention is external"
            ),
            "neutral_formula_missing_count": self.formula_missing,
            "invalid_valence_count": reasons["invalid_valence"],
            "invalid_smiles_count": reasons["invalid_smiles"],
            "unsupported_element_count": reasons["unsupported_element"],
            "rejection_reasons": dict(sorted(reasons.items())),
            "top_20_most_frequent_chain_combinations": [
                {"combination_id": combo, "count": count}
                for combo, count in sorted(
                    self.combos.items(), key=lambda pair: (-pair[1], pair[0])
                )[:20]
            ],
            "top_20_least_represented_polar_moieties": [
                {"polar_id": polar, "count": count}
                for polar, count in sorted(
                    self.per_polar.items(), key=lambda pair: (pair[1], pair[0])
                )[:20]
            ],
            "lyso_fraction_target": config.lyso_fraction_target,
            "lyso_fraction_actual": self.lyso / self.total if self.total else 0,
            "remaining_structure_budget": config.max_structures - self.total,
            "representative_sn_assignment": (
                "Computational representative placement, not biological sn annotation"
            ),
        }
