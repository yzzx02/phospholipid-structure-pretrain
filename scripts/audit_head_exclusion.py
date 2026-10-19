"""Re-query frozen v0.2 with identity-only and entire-polar-id exclusion side by side."""

import argparse
import json

from structure_pretrain.retrieval.head_exclusion import full_head_audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library", default="data/structure_library_v0.2/structures.parquet")
    p.add_argument("--cache", default="data/private/morgan_readiness_v0.1")
    p.add_argument("--output", default="data/private/head_exclusion_audit_v0.1")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--working-cache", help="Optional hash-identical cache on faster local storage")
    p.add_argument("--benchmark-targets", type=int, default=2000)
    a = p.parse_args()
    print(
        json.dumps(
            full_head_audit(
                a.library,
                a.cache,
                a.output,
                a.threads,
                a.benchmark_targets,
                a.workers,
                a.working_cache,
            ),
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
