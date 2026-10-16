"""Benchmark and audit indexed Morgan template retrieval without training or data splits."""

import argparse
import json
from pathlib import Path

from structure_pretrain.retrieval.indexed_morgan import (
    exact_benchmark,
    full_readiness,
    prepare_cache,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library", default="data/structure_library_v0.2")
    p.add_argument("--output", default="data/private/morgan_readiness_v0.1")
    p.add_argument("--benchmark-targets", type=int, default=10000)
    p.add_argument("--benchmark-reference", type=int, default=20000)
    a = p.parse_args()
    library, output = Path(a.library), Path(a.output)
    report = json.loads((library / "library_report.json").read_text(encoding="utf-8"))
    audit = json.loads((library / "verification_audit.json").read_text(encoding="utf-8"))
    if report["status"] != "complete" or audit["status"] != "passed":
        raise ValueError("Complete library and independent quality audit required")
    if not 10000 <= a.benchmark_targets <= 50000:
        raise ValueError("Formal benchmark requires 10,000-50,000 targets")
    cache = prepare_cache(library / "structures.parquet", output)
    benchmark_path = output / "exact_benchmark.json"
    if benchmark_path.exists():
        benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
        if (
            benchmark["status"] != "passed"
            or benchmark["target_count"] != a.benchmark_targets
            or benchmark["reference_subset_count"] != a.benchmark_reference
        ):
            raise ValueError("Existing benchmark failed or uses different subset sizes")
    else:
        benchmark = exact_benchmark(cache, output, a.benchmark_targets, a.benchmark_reference)
    report = full_readiness(cache, benchmark, output)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
