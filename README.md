# Exoplanet Transit Signal Classification Pipeline
### ISRO Bharatiya Antariksh Hackathon — Problem Statement 7
#### AI-enabled Detection of Exoplanets from Noisy Astronomical Light Curves

---

## Overview

This repository contains a complete, end-to-end machine learning pipeline that:
1. Downloads **real** Kepler light curves from NASA's MAST archive
2. Detrends and phase-folds them on known orbital periods
3. Extracts 12 physically-motivated **shape features** from each light curve
4. Trains an **XGBoost classifier** to distinguish 4 categories of periodic signals
5. Evaluates the model on a **held-out test set** and produces all required plots/metrics
6. Applies the trained model to **genuinely unlabeled TESS targets** (never seen during training)

---

## Signal Classes

| Class | Description | Key Diagnostic |
|-------|-------------|----------------|
| **Transit** | Genuine exoplanet transit | Flat-bottomed, symmetric, no secondary eclipse |
| **Eclipsing Binary** | Stellar eclipse (SS flag) | Odd/even depth difference, secondary eclipse present |
| **Blend** | Contamination from nearby source (CO/EC flags) | Centroid offset indicators, shallow/diluted |
| **Other/Noise** | Not transit-like (NT flag) | Irregular shape, low SNR, asymmetric |

---

## Pipeline Scripts

| Script | Purpose |
|--------|---------|
| `00_setup.py` | Install dependencies, verify imports |
| `01_data_prep.py` | Download KOI table → sample 360 targets → download LCs → extract features → `training_features.csv` |
| `02_train_eval.py` | Train XGBoost → evaluate on test set → generate all plots + metrics |
| `03_predict_unknown.py` | Apply to 5 unlabeled TESS targets → `unknown_target_predictions.csv` |

---

## How to Run

```bash
# Step 1: Install dependencies
python 00_setup.py

# Step 2: Download data and extract features (~45-90 min, mostly download-bound)
python 01_data_prep.py

# Step 3: Train and evaluate the model (~2-5 min)
python 02_train_eval.py

# Step 4: Predict on unknown TESS targets (~5-15 min)
python 03_predict_unknown.py
```

> **Resume support**: If `01_data_prep.py` is interrupted, simply re-run it — it caches downloaded light curves in `lc_cache/` and skips already-processed targets.

---

## Features Extracted

| Feature | Description | Discriminative Power |
|---------|-------------|---------------------|
| `transit_depth` | Depth of the flux dip (fractional) | All classes |
| `transit_duration_hrs` | Width of the transit in hours | EB vs Planet |
| `ingress_duration_hrs` | Time from baseline to minimum flux | Shape asymmetry |
| `egress_duration_hrs` | Time from minimum back to baseline | Shape asymmetry |
| `ingress_egress_ratio` | Ratio of ingress/egress (≈1.0 for true planets) | **Key discriminator** |
| `depth_duration_ratio` | V-shape vs U-shape metric | Grazing EB vs Planet |
| `odd_even_depth_diff` | Depth difference between odd/even transits | **Eclipsing Binary flag** |
| `secondary_eclipse_depth` | Depth at phase 0.5 (secondary eclipse) | **Eclipsing Binary flag** |
| `snr` | Depth / baseline scatter | All classes |
| `flat_bottom_fraction` | Fraction of transit at minimum flux level | Planet vs grazing EB |
| `koi_period` | Orbital period (days) | Period distribution differs by class |
| `koi_prad` | Planet radius (Earth radii, from catalog) | Size distribution |

---

## Output Files

| File | Description |
|------|-------------|
| `training_features.csv` | Extracted features for all training targets |
| `confusion_matrix.png` | 4×4 confusion matrix on held-out test set |
| `feature_importance.png` | XGBoost feature importance (gain) |
| `roc_curves.png` | One-vs-rest ROC curves for all 4 classes |
| `example_predictions.png` | 2×2 grid of phase-folded LCs with predictions |
| `classification_report.csv` | Precision/recall/F1 per class |
| `results_summary.txt` | Full results summary for PPT |
| `xgb_model.json` | Saved trained model |
| `unknown_target_predictions.csv` | Predictions for 5 TESS unlabeled targets |
| `unknown_lc_plots/` | Phase-folded LC plots per unknown target |

---

## Data Sources

- **Training labels**: [Kepler Cumulative KOI Table](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative) (NASA Exoplanet Archive)
- **Training light curves**: Kepler photometry from [NASA MAST](https://mast.stsci.edu/) via `lightkurve`
- **Unknown targets**: TESS photometry (SPOC pipeline) from MAST

---

## Key Design Choices

1. **No BLS/TLS during training**: For labeled KOIs, the period is already known from the NASA catalog, so we fold directly — no redundant search.
2. **TLS for unknown targets**: Transit Least Squares models a realistic limb-darkened transit shape, giving better sensitivity than BLS. Falls back to `astropy.timeseries.BoxLeastSquares` if TLS is unavailable.
3. **Parallelized downloads**: 5 concurrent threads + local caching minimizes total runtime.
4. **Clean train/test separation**: Metrics are reported ONLY on the 25% held-out test set. Cross-validation is run only on the training partition.
5. **Explainability**: Feature importance plot directly ties the model's decisions to physically interpretable astrophysical diagnostics.
