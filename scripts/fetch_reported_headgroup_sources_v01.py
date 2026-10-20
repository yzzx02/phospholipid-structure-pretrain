"""Cache public source snapshots on G drive; never replace an existing cached file."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path("outputs/reported_headgroups_v01_20261020/raw")
MSDIAL_SHA = "349d7ef6edbd09e34c0eb38b849bd9fd8af79e7e"
FILES = {
    "LM_download.tsv": "https://www.lipidmaps.org/rest/compound/lm_id/LM/all/download",
    "LMGP_download.tsv": "https://www.lipidmaps.org/rest/compound/lm_id/LMGP/all/download",
    "LMSP_download.tsv": "https://www.lipidmaps.org/rest/compound/lm_id/LMSP/all/download",
    "LMSD_extended.sdf.zip": "https://www.lipidmaps.org/files/?file=LMSD_extended&ext=sdf.zip",
    "LipidBlast_Glycerolipids.zip": "https://zenodo.org/records/10953284/files/LipidBlast%20DBs%20fork%20GlyceroLipids%20VS12.zip?download=1",
}
for file, directory in {
    "SmilesInchikeyGenerator.cs": "src/Common/CommonStandard/Lipidomics",
    "LipidClassDictionary.cs": "src/Common/CommonStandard/Lipidomics",
    "LipidClassProperties.csv": "src/Common/CommonStandard/Resources",
    "SmilesInchikeyGeneratorTest.cs": "tests/Common/CommonStandardTests/Lipidomics",
}.items():
    FILES[file] = (
        f"https://raw.githubusercontent.com/systemsomicslab/MsdialWorkbench/{MSDIAL_SHA}/{directory}/{file}"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-root", type=Path, default=ROOT)
    p.add_argument("--proxy", default="http://127.0.0.1:17891")
    args = p.parse_args()
    args.raw_root.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    if args.proxy:
        session.proxies.update({"https": args.proxy, "http": args.proxy})
    manifest = []
    for name, url in FILES.items():
        path = args.raw_root / name
        cached = path.exists()
        if not cached:
            r = session.get(url, timeout=90)
            r.raise_for_status()
            if len(r.content) < 100:
                raise ValueError(f"Unexpected empty/short source payload: {url}")
            path.write_bytes(r.content)
        with path.open("rb") as f:
            sha = hashlib.file_digest(f, "sha256").hexdigest()
        manifest.append(
            {
                "name": name,
                "url": url,
                "sha256": sha,
                "bytes": path.stat().st_size,
                "cache_reused": cached,
                "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
    (args.raw_root / "source_snapshot_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps({"sources": len(manifest), "cached": sum(m["cache_reused"] for m in manifest)})
    )


if __name__ == "__main__":
    main()
