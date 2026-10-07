# Code guide

This guide maps the source files of the final submission (submission 79) to the method
described in the [README](../README.md) and the [poster](poster.pdf). The release is
source-only: trained weights, data and intermediate artifacts are not included.

## Online protocol

The CrunchDAO real-time runner calls two hooks in `submission/main.py`:

- `train(datasets, model_directory_path)` mainly authenticates the pre-trained resources
  (hash checks). All models were fitted offline; no fitting happens in the submission run.
- `infer(datasets, model_directory_path)` is a generator. It yields once to signal
  readiness, then receives each series as `(history, online)` and yields exactly one
  score per online observation.

The serving code is a stack of wrappers. Each wrapper runs its parent's generator through
a proxy that hands over observations one at a time and asserts that the parent never
reads ahead, then combines the parent score with its own term. Every layer keeps
per-series state that is created from the series' history; nothing is shared across
series.

## Serving stack (outermost first)

| Layer | File | Adds | Resources under the model directory |
|---|---|---|---|
| 1 | `main.py` | Final contrast: `logit(p_mix) + 0.1 * [logit(p_aug) - logit(p_ctl)]`, using `feature_serving/` | `feature_integration/` |
| 2 | `incumbent_main.py` | `p_mix = 0.875 * p_base + 0.125 * p_chronos`, using `chronos_adapter_runtime.py`, `chronos_patch_network.py`, `chronos_adapter_network.py`, `chronos_single_group.py` | `chronos_adapter/` |
| 3 | `legacy_gpu_main.py` | Conditional Bayesian evidence (`conditional_bayes_runtime_v199.py`, `discord_bayes_features_v198.py`), adaptive-variance evidence (`adaptive_variance_runtime.py`, `adaptive_features.py`) and causal temporal CNNs (`gpu_temporal_runtime.py`, `gpu_temporal_network.py`) | `conditional_bayes/`, `adaptive_variance/`, `gpu_temporal/` |
| 4 | `selected64_main.py` | Frozen Chronos-Bolt-mini encoder with a tree head, 12.5% blend with normal-history calibration (`encoder_submission_runtime_v108.py`, `encoder_change_v108.py`, `normal_calibration_v124.py`) | `encoder/`, `encoder_blend.json` |
| 5 | `selected60_main.py` | NLinear residual and distribution-change networks with GRU corrections and a time-varying blend weight (`nlinear_detector/`) | `neural/`, `blend.json` |
| 6 | `baseline/main.py` | Statistical ensemble: a chain of `baseline/v*_parent_main.py` wrappers adding calibration, memory, positional and curvature corrections over the tree and ranking core in `baseline/v338/` | `baseline/` |

In poster terms, layers 6 to 3 build `p_base`, layer 2 mixes in `p_chronos`, and layer 1
adds the contrast of `p_aug` and `p_ctl`. The predictive-distribution heads, the
conditional Bayesian, adaptive-variance and temporal-CNN corrections, and Chronos-2 are
refreshed every 4 observations and held in between. The Chronos-2 runtime needs a CUDA
GPU; the other layers run on CPU.

## Training sources (archival recipes)

| Path | Produces |
|---|---|
| `neural_training_source/` | NLinear/GRU detector networks used by layer 5 |
| `train_extended_tree_submission_v120.py` | Tree head on frozen Chronos-Bolt features used by layer 4 |
| `training_source/` | Conditional Bayesian features and adaptive-variance heads used by layer 3 |
| `gpu_training_source/` | Causal temporal CNN residuals used by layer 3 |
| `chronos_training_source/` | Earlier Chronos-2 task adapter |
| `chronos_patch_training_source/` | Chronos-2 localization and patch-localization head used by layer 2 |
| `feature_training_source/` | Quantile predictors, augmented and control heads, and the blend comparison that selected the 0.1 contrast (layer 1) |

These scripts read intermediate artifacts from earlier project stages (cached feature
matrices, out-of-fold predictions and some earlier workspace modules). Those artifacts are
not part of this release, so the scripts document the recipes but do not run end to end
from this repository alone.

## Configuration

Ten training-only scripts read their input locations from `ADIA_*` environment variables
(`ADIA_DATA_DIR`, `ADIA_PYTHON`, `ADIA_PRIOR_ROOT`, `ADIA_LEGACY_ROOT`,
`ADIA_PREDICTIVE_ROOT`, `ADIA_FEATURE_RECIPE`, `ADIA_CHRONOS_WORK`). Defaults are relative
to the current working directory; the README lists which files read which variable. The
competition data must be obtained through the official CrunchDAO competition under its
terms.

## Verification

```
python tools/verify_source.py
```

This checks byte counts and SHA-256 hashes against `source_manifest.json`, parses every
Python file and scans for `/Users/` paths, Google Drive links and common credential
patterns. No model code is imported or executed.
