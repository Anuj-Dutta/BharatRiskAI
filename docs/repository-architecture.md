# Repository architecture audit

## Audit snapshot

The repository is a Python/FastAPI service with a Vite/React frontend. The
backend entry point is `backend/main.py`; frontend entry points are
`frontend/src/main.jsx` and `frontend/index.html`. Production imports use
package-qualified paths such as `backend.*`, `ml.*`, `scripts.*`, and
`ingestion.*`.

The existing production boundaries are:

| Area | Current location | Decision |
| --- | --- | --- |
| Frontend | `frontend/` | Already correctly isolated |
| Backend/API | `backend/` | Already correctly isolated |
| Traditional ML | `ml/models`, `ml/training`, `ml/preprocessing` | Preserved |
| Deep learning | `deep_learning/architectures`, `ml/nowcasting` | Architecture separated; compatibility facades preserved |
| Data ingestion | `ingestion/`, `ml/ingestion` | Preserved because both are imported by production code |
| Data pipeline | `scripts/` | Preserved as executable pipeline package |
| Experiments | `experiments/algorithm_comparison/` | Moved out of `ml/` |
| Tests | `tests/` plus root operational probes | Preserved; root probes are not production test modules |
| Configuration | `configs/application/`, backend settings, frontend package config | Centralized application and scientific configuration |
| Deployment | `infra/`, `docker-compose.yml`, `Makefile` | Preserved |
| Generated artifacts | `data/`, `models/`, `reports/`, `ml/artifacts/` | Preserved and kept separate from source |

## Import and path risks

The data pipeline and runtime still intentionally use `scripts.*`, `ingestion.*`,
and `ml.nowcasting.*`. They are imported by tests, training, calibration,
backend services, and live ingestion. Moving those packages wholesale would
require a large compatibility migration and would risk changing data-path
semantics. Instead, the neural architecture modules now have canonical
locations under `deep_learning/architectures/`, while compatibility modules
keep existing callers working.

Raw and generated data directories are gitignored by design. Existing tracked
reference data, model checkpoints, reports, and configuration files were not
deleted or rewritten.

## Canonical structure

```text
frontend/                         React/Vite application
backend/                          FastAPI API and services
ml/                               traditional ML and runtime data facades
deep_learning/                    neural architectures and DL documentation
data/                             raw/interim/processed runtime data
ingestion/                        application ingestion integrations
scripts/                          executable data pipeline stages
experiments/algorithm_comparison/ isolated nine-model benchmarks
tests/                            automated tests
docs/                             system and operational documentation
configs/application/              scientific/model configuration
infra/                            Docker and reverse-proxy definitions
models/, reports/, ml/artifacts/  generated model and report artifacts
```

## Validation commands

```powershell
python -m compileall -q backend ml deep_learning ingestion scripts experiments
python -m pytest -q
Set-Location frontend
npm run build
Set-Location ..
python -c "from backend.main import app; print(app.title)"
```
