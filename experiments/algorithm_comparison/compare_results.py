"""Convert JSONL experiment records to category-specific CSV reports."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def convert(source: str | Path, destination: str | Path) -> int:
    rows = [json.loads(line) for line in Path(source).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"No result records found in {source}")
    keys = sorted({key for row in rows for key in row})
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(f"Wrote {convert(args.source, args.destination)} records to {args.destination}")


if __name__ == "__main__":
    main()

