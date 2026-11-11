"""Explicit registry of exactly the nine experimental alternatives."""
from __future__ import annotations

from .mtl_alternatives.mmoe import MMoE
from .mtl_alternatives.mtan import MTAN
from .mtl_alternatives.ple import PLE
from .spatiotemporal_alternatives.earthformer import Earthformer
from .spatiotemporal_alternatives.predrnn import PredRNN
from .spatiotemporal_alternatives.simvp import SimVP


def _xgboost(**kwargs):
    from .random_forest_alternatives.xgboost_model import build_model
    return build_model(**kwargs)


def _lightgbm(**kwargs):
    from .random_forest_alternatives.lightgbm_model import build_model
    return build_model(**kwargs)


def _extra_trees(**kwargs):
    from .random_forest_alternatives.extra_trees_model import build_model
    return build_model(**kwargs)


REGISTRY = {
    "random_forest": {
        "xgboost": _xgboost,
        "lightgbm": _lightgbm,
        "extra_trees": _extra_trees,
    },
    "mtl": {"mmoe": MMoE, "ple": PLE, "mtan": MTAN},
    "spatiotemporal": {"predrnn": PredRNN, "earthformer": Earthformer, "simvp": SimVP},
}

if sum(len(models) for models in REGISTRY.values()) != 9:
    raise RuntimeError("The comparison registry must contain exactly nine alternatives")
