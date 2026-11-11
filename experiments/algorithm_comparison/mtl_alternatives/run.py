"""Smoke-run MTL alternatives using existing canonical feature generation."""
from __future__ import annotations

import argparse
import json
import time

import torch

from ..adapters.spatiotemporal_adapter import build_inputs
from ..config import ExperimentConfig


def run(smoke: bool = True) -> list[dict]:
    config = ExperimentConfig()
    torch.manual_seed(config.seed)
    sequence, _ = build_inputs({"iwv": 55.0, "cape_j_kg": 2200.0})
    inputs = torch.from_numpy(sequence).unsqueeze(0)
    from .mmoe import MMoE
    from .mtan import MTAN
    from .ple import PLE
    models = {"mmoe": MMoE, "ple": PLE, "mtan": MTAN}
    results = []
    for name, constructor in models.items():
        started = time.perf_counter()
        model = constructor(hidden_channels=4 if smoke else config.hidden_channels)
        with torch.inference_mode():
            outputs = model(inputs)
        results.append({"model": name, "status": "ok", "output_shape": list(outputs.shape), "parameters": sum(p.numel() for p in model.parameters()), "inference_seconds": time.perf_counter() - started})
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    print(json.dumps(run(parser.parse_args().smoke), indent=2))


if __name__ == "__main__":
    main()

