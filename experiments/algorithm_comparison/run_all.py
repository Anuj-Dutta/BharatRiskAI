"""Run selected isolated experiments; no full training is started by default."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .mtl_alternatives.run import run as run_mtl
from .random_forest_alternatives.run import run as run_rf
from .spatiotemporal_alternatives.run import run as run_spatiotemporal


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", choices=("random_forest", "mtl", "spatiotemporal", "all"), default="all")
    parser.add_argument("--dataset", type=Path, help="Required for random-forest alternatives")
    parser.add_argument("--target")
    parser.add_argument("--smoke", action="store_true", help="Use small models and forward-pass validation")
    args = parser.parse_args()
    result = {}
    if args.category in ("random_forest", "all"):
        if args.dataset is None:
            raise SystemExit("--dataset is required for random_forest or all")
        result["random_forest"] = run_rf(args.dataset, args.target, smoke=args.smoke)
    if args.category in ("mtl", "all"):
        result["mtl"] = run_mtl(args.smoke)
    if args.category in ("spatiotemporal", "all"):
        result["spatiotemporal"] = run_spatiotemporal(args.smoke)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()

