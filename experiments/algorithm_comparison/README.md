# Algorithm comparison experiments

This package contains exactly nine isolated alternatives for benchmarking:
XGBoost, LightGBM, Extra Trees; MMoE, PLE, MTAN; and PredRNN, Earthformer,
SimVP. It does not replace or import experimental models from the production
application.

## Verified data contracts

* Tabular experiments reuse `ml.training.train.load_training_data` and
  `time_split`, including the temporal 80/20 split, imputation convention, and
  seven RF feature columns. The current repository's trainer calls its default
  target `target`; a dataset with an explicit `risk_score` column can be
  selected with `--target risk_score`.
* Neural experiments consume the canonical sequence assembled by
  `scripts.datasets.build_sequences.assemble_sequences`: `[7, 13, H, W]`
  inputs and `[5, 3, H, W]` targets. The inference adapter reuses
  `ml.nowcasting.features.build_spatiotemporal_features` and verifies
  `[7, 13, 16, 16]` plus `[6, 16, 16]`.
* Every neural alternative returns `[B, 5, 3, 16, 16]`. Optional masks are
  applied by MTL models without changing the output contract.

## Installation and execution

The primary `requirements.txt` already includes PyTorch and scikit-learn.
XGBoost and LightGBM remain optional; missing packages are reported as skipped,
not silently substituted:

```powershell
python -m pip install xgboost lightgbm
```

Run category smoke tests:

```powershell
python -m experiments.algorithm_comparison.mtl_alternatives.run --smoke
python -m experiments.algorithm_comparison.spatiotemporal_alternatives.run --smoke
python -m experiments.algorithm_comparison.random_forest_alternatives.run --dataset .\path\to\data.csv --smoke
```

Run all nine (RF still requires an explicit user dataset):

```powershell
python -m experiments.algorithm_comparison.run_all --category all --dataset .\path\to\data.csv --smoke
```

Full experiments are opt-in by omitting `--smoke`. JSONL records can be
converted to a comparison report:

```powershell
python -m experiments.algorithm_comparison.compare_results .\results.jsonl .\experiments\algorithm_comparison\results\mtl_comparison.csv
```

Tabular reports use MAE and R². Neural benchmarking should use the existing
per-task loss and forecast metrics (CSI, POD, FAR, HSS, IoU) against real
assembled targets. No synthetic data, labels, checkpoints, or production
results are created by this package. Earthformer is a compact compatible
attention implementation and is not a dependency on the separate upstream
Earthformer project.
