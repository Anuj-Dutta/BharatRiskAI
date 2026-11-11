# Deep learning

Canonical neural architectures live under `deep_learning/architectures/`.
The spatiotemporal nowcasting feature and inference facades remain in
`ml/nowcasting/` for backward compatibility with the production API and
training scripts. Compatibility modules preserve existing imports while new
code should import architectures from this package.

The canonical forecasting contract is:

* dynamic sequence: `[B, 7, 13, H, W]`
* baseline: `[B, 6, H, W]`
* prediction: `[B, 5, 3, H, W]`

Training and validation commands remain documented in
`docs/data-pipeline-contract.md` and `docs/data-pipeline-guide.md`.
