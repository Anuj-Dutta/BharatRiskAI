"""Smoke-run spatiotemporal alternatives using existing feature extraction."""
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
    sequence, baseline = build_inputs({"iwv": 55.0, "cape_j_kg": 2200.0})
    sequence_tensor = torch.from_numpy(sequence).unsqueeze(0)
    baseline_tensor = torch.from_numpy(baseline).unsqueeze(0)
    from .earthformer import Earthformer
    from .predrnn import PredRNN
    from .simvp import SimVP
    models = {"predrnn": PredRNN, "earthformer": Earthformer, "simvp": SimVP}
    results = []
    for name, constructor in models.items():
        started = time.perf_counter()
        model = constructor(hidden_channels=4 if smoke else config.hidden_channels)
        with torch.inference_mode():
            outputs = model(sequence_tensor, baseline_tensor)
        results.append({"model": name, "status": "ok", "output_shape": list(outputs.shape), "parameters": sum(p.numel() for p in model.parameters()), "inference_seconds": time.perf_counter() - started})
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    print(json.dumps(run(parser.parse_args().smoke), indent=2))


if __name__ == "__main__":
    main()

