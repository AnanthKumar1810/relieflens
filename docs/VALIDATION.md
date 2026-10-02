# ReliefLens validation

## Automated test suite

`pytest -q` → **23 passed** in the final local run.

Coverage includes:

- xBD JSON pixel-polygon parsing and mask rendering
- full MultiPolygon component preservation
- unknown damage-label rejection rather than silent background conversion
- collision-safe pre/post discovery with directory-aware sample IDs
- image-dimension validation
- event-disjoint manifest validation, including duplicate disaster-assignment rejection
- configurable random-crop training coverage and synchronized geometric augmentation
- Siamese ReliefLensNet and six-channel U-Net output shapes
- composite loss forward/backward and class-weighted focal loss
- temperature scaling with positive temperature enforcement and ECE
- segmentation/damage metrics, streaming metric aggregation, and triage ranking
- registration sanity checks
- confidence/entropy-based abstention outputs
- direct image-pair inference shape/priority checks
- self-contained CLI evaluation after generating a synthetic checkpoint
- Python compilation of all source modules

## End-to-end synthetic integration

A four-event xBD-style fixture completed this pipeline locally:

`raw fixture → event-disjoint manifests → dataset patches → Siamese training → validation checkpoint → validation-only temperature calibration → held-out evaluation`

The six-channel baseline completed the same train/evaluate path. The smoke fixture is intentionally tiny; its numerical metrics are **not evidence of real-world model performance**. They are only evidence that the project executes end-to-end.

## Trustworthiness changes

- Dataset discovery no longer keys files by basename alone; relative directory context is retained and same-key duplicates are rejected.
- MultiPolygon annotations retain all polygon components.
- Unknown damage labels raise by default; an explicit `unknown_damage_policy: ignore` mode is available when a downstream workflow chooses warning-and-skip semantics.
- Held-out evaluation never fits temperature on the test set. Calibration temperature is produced from validation data during training and stored with the checkpoint.
- Evaluation retains bounded calibration samples and streaming confusion/triage statistics instead of all test-pixel logits.
- The smoke CLI test generates its own checkpoint and manifest; there is no silent skip when artifacts are missing.

## Dashboard status

The dashboard source and inference core compile and the inference path was exercised with a generated checkpoint. The current execution environment does not have the `streamlit` package installed, so the interactive UI itself was not launched here. The dependency is declared in the project metadata and CI installs the full project.

## CI

`.github/workflows/ci.yml` installs the project plus development tooling, compiles `src` and `tests`, runs Ruff lint, and runs the full pytest suite on every push and pull request.

## Not included

Real xBD imagery and a trained real-data checkpoint are intentionally not bundled. Meaningful project metrics must be generated on downloaded xBD/xView2 data using an event-disjoint held-out test set.
