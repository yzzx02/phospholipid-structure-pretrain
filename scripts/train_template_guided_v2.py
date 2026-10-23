"""Entry point for one independent V2 experiment; host supervision is mandatory."""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from structure_pretrain.formal_v2.trainer import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("E1", "E2", "E3"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()

    def event(**row):
        row["unix_time"] = time.time()
        args.output.mkdir(exist_ok=True)
        with (args.output / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    train(args.experiment, args.output, event, args.resume)


if __name__ == "__main__":
    main()
