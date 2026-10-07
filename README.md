# Online Structural Break Detection

Source code of the selected final solution (submission 79) for the
**ADIA Lab Structural Break Challenge: Real-Time Edition** on CrunchDAO.

Author: Yu Suzuki, Ph.D. (entry affiliation: Independent Researcher)

One-page summary: [docs/poster.pdf](docs/poster.pdf) - Code guide: [docs/CODE_GUIDE.md](docs/CODE_GUIDE.md)

> **Scope of this release.** This is an archival, source-only release. It contains the
> Python sources and the recorded dependency lists of the final selected submission.
> It does **not** contain competition data, trained model weights, cached features,
> predictions, or the intermediate artifacts that several training scripts read.
> The serving code therefore cannot run without the trained weights, and the training
> scripts document the recipes rather than forming a one-command retraining pipeline.

## Task

Each series provides a break-free history; new observations then arrive one at a time.
After every observation the model outputs a score for how likely it is that a structural
break has already occurred, using only that series' history and the observations seen so
far. Submissions are scored by TS-AUC, which compares the scores of different series at
the same online time.

## Method

Every component measures the observed prefix against the series' **own break-free
history**, so that evidence is comparable across series. Three complementary views are
combined:

- **A. Statistical ensemble (`p_base`).** Over short to long windows, features describe how
  recent values drift from the normal past (for example level, variability and the
  correlation of successive values), for raw values and for AR residuals, i.e. what
  earlier values fail to predict. Tree classifiers and rankers learn from these features;
  the ranking objective compares series at the same online step. Later layers build on
  this base: calibration and memory corrections, NLinear/GRU networks, frozen Chronos-Bolt
  features, conditional Bayesian evidence, adaptive-variance evidence and causal temporal
  CNNs.
- **B. Adapted Chronos-2 (`p_chronos`).** Used as an encoder, not a forecaster. The last
  1,024 points form 64 ordered 16-point patches whose embeddings are compared with the
  series' own history reference. Only the final two encoder blocks, the final norm and a
  patch-localization head were adapted.
- **C. Predictive-distribution heads (`p_aug`, `p_ctl`).** Per-series AR(12) innovations are
  compared with quantiles predicted by networks trained on normal histories. PIT, z-score
  and pinball-loss features enter an augmented head; a matched control head lacks them.

Final integration, with weights chosen during development and fixed at inference:

```
p_mix = 0.875 * p_base + 0.125 * p_chronos
s_t   = sigmoid( logit(p_mix) + 0.1 * [ logit(p_aug) - logit(p_ctl) ] )
```

Probabilities are clipped to [1e-7, 1 - 1e-7] before taking logits. Chronos-2 and the
contrast heads are refreshed every 4 observations and held in between. Each series is
scored independently and causally: at inference no future values and no other series
are used.

Why combine views with small weights: different views make different ranking errors, so
a blend can let one view offset another's mistakes, while small fixed weights aim to keep
the established ensemble as the anchor and limit the influence of uncertain new evidence
on scores.

## Results

The selected submission 79 has a Public TS-AUC of **64.29%** as displayed (rounded) on
the Crunch Hub. Private results and the final ranking are not known at the time of
writing.

Representative Public milestones during development. Each row is a cumulative
configuration, not an isolated ablation:

| Stage | Cumulative change | Public TS-AUC (%) |
|---|---|---|
| Statistical features | Three-model logit blend | 62.07 |
| | Tree + ranking stack | 62.87 |
| Correction layers | Loss-weighted bagging | 63.00 |
| | Calibration and memory | 63.10 |
| Normal-behavior models | NLinear networks + GRU | 63.61 |
| | Chronos-Bolt embeddings | 63.69 |
| | Bayesian contrast | 63.86 |
| Contrasts + Chronos-2 | Adaptive variance + temporal CNNs | 64.03 |
| | Chronos-2 + feature contrast | 64.25 |
| | Patch head, faster refresh (final) | 64.29 |

In the initial five-fold development comparison, the fixed three-model logit blend
(weights 0.20 / 0.40 / 0.40) scored 62.16% against 61.71% for the best single model and
was higher in all five folds. Development folds were reused throughout the research, so
local development scores are descriptive, not independent validation.

## Repository layout

| Path | Contents |
|---|---|
| `submission/main.py` | Entry point of the final submission; applies the 0.1 logit contrast |
| `submission/incumbent_main.py` | 0.875 / 0.125 mixture with the adapted Chronos-2 component |
| `submission/legacy_gpu_main.py` | Conditional Bayesian, adaptive-variance and temporal-CNN corrections |
| `submission/selected64_main.py`, `submission/selected60_main.py` | Chronos-Bolt and NLinear/GRU blends |
| `submission/baseline/` | Statistical ensemble and its correction layers |
| `submission/feature_serving/` | Predictive-distribution features, augmented and control heads |
| `submission/chronos_*.py` | Chronos-2 runtime and networks |
| `submission/nlinear_detector/` | NLinear residual and distribution-change networks |
| `submission/*training_source/`, `submission/train_extended_tree_submission_v120.py` | Archival training recipes |
| `source_manifest.json` | Byte count, SHA-256 and submitted-file SHA-256 of every source file |
| `tools/verify_source.py` | Source-only verification (no model execution) |
| `LICENSE`, `NOTICE`, `licenses/` | Licenses and upstream notices |
| `docs/` | Poster and code guide |

See [docs/CODE_GUIDE.md](docs/CODE_GUIDE.md) for the serving stack, layer by layer, and
the training-source map.

## Verifying the source release

```
python tools/verify_source.py
```

The script checks every file under `submission/` against `source_manifest.json` (byte
count and SHA-256), parses all 174 Python files, rejects unexpected files, `/Users/`
paths, Google Drive links and common credential patterns, and prints a `PASS` line. It
does not import or run any model code.

The dependency lists and 162 Python files are byte-identical to the submitted files.
Publication changes are limited to ten training-only scripts with machine-specific paths
replaced by `ADIA_*` environment variables and two files with private community links and
participant-specific provenance omitted. Each changed file has a publication comment;
model, feature and training algorithms are unchanged. `source_manifest.json` records both
the submitted and published SHA-256 of every file.

## Training scripts and the reproduction boundary

The training scripts read intermediate artifacts from earlier project stages (cached
feature matrices, out-of-fold predictions and some earlier workspace modules) that are
not part of this release. Their locations are configured with environment variables;
defaults are relative to the current working directory:

| Variable | Default | Read by |
|---|---|---|
| `ADIA_DATA_DIR` | `./data` | `feature_training_source/predictor_source/common.py`, `feature_training_source/training_input_source/common.py` |
| `ADIA_PYTHON` | current interpreter | same two files |
| `ADIA_PRIOR_ROOT` | `./external-artifacts/prior` | same two files |
| `ADIA_LEGACY_ROOT` | `./external-artifacts/legacy` | `feature_training_source/integration_common.py`, `gpu_training_source/{evaluate,train,train_grouped_fixed,train_replicate}.py`, `training_source/{extract_adaptive,fit_adaptive}.py` |
| `ADIA_PREDICTIVE_ROOT` | `./external-artifacts/predictive` | `feature_training_source/integration_common.py` |
| `ADIA_FEATURE_RECIPE` | `./external-artifacts/run_v255_equal_series_logtick_v4.py` | `feature_training_source/integration_common.py` |
| `ADIA_CHRONOS_WORK` | `./external-artifacts/chronos2` | `chronos_patch_training_source/recipe_sources/chronos_localization_data.py` |

Within these roots, several scripts still expect the directory layout of the original
workspace.

## Data

The competition data are not redistributed here. Obtain them through the official
CrunchDAO competition with your own account and under the competition's terms. The
training scripts expect the organizer's training files (for example `X_train.parquet`)
under the configured data locations.

## Dependencies

- Final serving environment: `submission/requirements.txt` (NumPy, LightGBM, XGBoost,
  SciPy, pandas, PyArrow, PyTorch 2.7.1+cu118, chronos-forecasting 2.3.2, transformers,
  accelerate, scikit-learn, threadpoolctl; exact pins in the file).
- Training environment as recorded: `submission/requirements-training.txt`.
- Component lists kept as recorded at earlier stages (pins can differ from the serving
  list): `submission/baseline/`, `submission/baseline/v338/`,
  `submission/baseline/v338/base/` and `submission/feature_serving/`.
- The Chronos-2 runtime requires a CUDA GPU (float16 encoder, CUDA graphs); the other
  components run on CPU.
- Pretrained backbones `amazon/chronos-2` and `amazon/chronos-bolt-mini` are loaded from
  local copies at serving time and are not included.

## References

1. A. Zeng, M. Chen, L. Zhang and Q. Xu. Are Transformers Effective for Time Series
   Forecasting? AAAI, 37(9):11121-11128, 2023. (NLinear)
2. A. F. Ansari et al. Chronos-2: From Univariate to Universal Forecasting.
   arXiv:2510.15821, 2025.
3. A. F. Ansari et al. Chronos: Learning the Language of Time Series. Transactions on
   Machine Learning Research, 2024. arXiv:2403.07815. (Chronos family, including
   Chronos-Bolt)

## Acknowledgements

- One training recipe of the statistical ensemble adopts the equal-series, log-tick row
  sampling, the weighting and the LightGBM profile published in a public repository from
  the earlier ADIA Lab Structural Break Challenge
  (`github.com/mq1234567/ADIA_Lab_Structural_Break_Challenge`, commit `bcdcf16`); see
  `submission/feature_training_source/training_input_source/head_recipe.py`. This release
  includes our implementation of those ideas and hyperparameters, and redistributes no
  source files from that repository.
- The conditional Bayesian change evidence re-implements mathematics that a participant
  described in the competition community; no participant source code is redistributed.
- Chronos-2 and Chronos-Bolt are by Amazon; NLinear is from LTSF-Linear.

## License

Apache License 2.0; see `LICENSE`. Upstream license texts and notices are in `licenses/`
and `NOTICE`. Third-party dependencies and pretrained models keep their own licenses and
are not included in this repository.
