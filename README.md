# Exoplanet Transit Signal Classification Pipeline
### ISRO Bharatiya Antariksh Hackathon — Problem Statement 7
#### AI-enabled Detection of Exoplanets from Noisy Astronomical Light Curves

---

## Overview

This repository contains a complete, end-to-end machine learning pipeline that:
1. Downloads **real** Kepler light curves from NASA's MAST archive
2. Detrends and phase-folds them on known orbital periods
3. Extracts 19 physically-motivated **shape features** from each light curve
4. Trains an **Ensemble Classifier (XGBoost, LightGBM, CatBoost)** to distinguish 4 categories of periodic signals
5. Evaluates the models on a **held-out test set** using soft-voting, producing all required plots/metrics
6. Applies the trained ensemble to **genuinely unlabeled TESS targets** (never seen during training)

---

## Signal Classes

| Class | Description | Key Diagnostic |
|-------|-------------|----------------|
| **Transit** | Genuine exoplanet transit | Flat-bottomed, symmetric, no secondary eclipse |
| **Eclipsing Binary** | Stellar eclipse (SS flag) | Odd/even depth difference, secondary eclipse present |
| **Blend** | Contamination from nearby source (CO/EC flags) | Centroid offset indicators, shallow/diluted |
| **Other/Noise** | Not transit-like (NT flag) | Irregular shape, low SNR, asymmetric |

---

## Scale-Up & Enhancement Results (N=20 vs N=100)

The pipeline was scaled from a small pilot (N=20 per class) to a larger dataset (N=100 per class), yielding significant performance enhancements and inter-model agreement. 

| Metric | N=20 Run (45 clean samples) | N=100 Run (229 clean samples) | Enhancement |
|---|---|---|---|
| **Ensemble Accuracy** | 0.4167 | **0.6034** | **+18.7 pp** (+45% relative) |
| **Ensemble Macro-F1** | 0.4937 | **0.5978** | **+10.4 pp** (+21% relative) |
| **Model Agreement** | 25.0% | **69.0%** | **+44.0 pp** |

**Key Feature Discovery:** At N=100, newly engineered physics-motivated features like `secondary_eclipse_significance` emerged into the **Top 3** most important features, demonstrating that these signals require sufficient data to overcome estimation noise. 

---

## Pipeline Scripts

| Script | Purpose |
|--------|---------|
| `00_setup.py` | Install dependencies, verify imports |
| `01_data_prep.py` | Download KOI table → sample 400 targets → download LCs → extract features → `training_features.csv` |
| `02_train_eval.py` | Train Ensemble (XGB, LGB, Cat) → evaluate on test set → generate plots + metrics |
| `03_predict_unknown.py` | Apply Ensemble to 5 unlabeled TESS targets → `unknown_target_predictions.csv` |

---

## How to Run

```bash
# Step 1: Install dependencies
python 00_setup.py

# Step 2: Download data and extract features (~5-10 min, mostly download-bound)
python 01_data_prep.py

# Step 3: Train and evaluate the models (~2-5 min)
python 02_train_eval.py

# Step 4: Predict on unknown TESS targets (~5-15 min)
python 03_predict_unknown.py
```

> **Resume support**: If `01_data_prep.py` is interrupted, simply re-run it — it uses Lightkurve's built-in MAST cache to instantly skip already-processed targets. Corrupt FITS files are automatically deleted and re-downloaded.

---

## Features Extracted (19 total)

| Feature | Description |
|---------|-------------|
| `transit_depth` | Depth of the flux dip (fractional) |
| `transit_duration_hrs` | Width of the transit in hours |
| `ingress_duration_hrs` | Time from baseline to minimum flux |
| `egress_duration_hrs` | Time from minimum back to baseline |
| `ingress_egress_ratio` | Ratio of ingress/egress (≈1.0 for true planets) |
| `depth_duration_ratio` | V-shape vs U-shape metric |
| `odd_even_depth_diff` | Depth difference between odd/even transits |
| `odd_even_depth_significance`| Statistical significance of odd/even diff |
| `secondary_eclipse_depth` | Depth at phase 0.5 (secondary eclipse) |
| `secondary_eclipse_significance`| Statistical significance of secondary eclipse |
| `snr` | Depth / baseline scatter |
| `flat_bottom_fraction` | Fraction of transit at minimum flux level |
| `transit_symmetry_score`| Pearson correlation between ingress and inverted egress |
| `out_of_transit_variability`| Standard deviation of out-of-transit flux |
| `transit_count`| Number of transits observed in the light curve |
| `stellar_density_proxy`| Derived metric correlating with stellar density |
| `depth_consistency_across_transits`| Stability of depth across individual epochs |
| `koi_period` | Orbital period (days) |
| `koi_prad` | Planet radius (Earth radii, from catalog) |

---

## Output Files

| File | Description |
|------|-------------|
| `training_features.csv` | Extracted features for all training targets |
| `download_failures.csv` | Log of targets that failed to download or extract |
| `confusion_matrix.png` | 4×4 confusion matrix on held-out test set |
| `feature_importance.png` | XGBoost feature importance (gain) |
| `roc_curves.png` | One-vs-rest ROC curves for all 4 classes |
| `scale_comparison.csv` | Comparison of N=20 vs N=100 run metrics |
| `model_comparison.csv` | Precision/recall/F1 for each model and the ensemble |
| `results_summary.txt` | Full results summary for PPT |
| `xgb_model.json`, `lgb_model.txt`, `cat_model.cbm` | Saved trained models |
| `unknown_target_predictions.csv` | Predictions for 5 TESS unlabeled targets |
| `unknown_lc_plots/` | Phase-folded LC plots per unknown target |
| `results_n20/` | Preserved outputs from the earlier N=20 pilot run |

---

## Data Sources

- **Training labels**: [Kepler Cumulative KOI Table](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative) (NASA Exoplanet Archive)
- **Training light curves**: Kepler photometry from [NASA MAST](https://mast.stsci.edu/) via `lightkurve`
- **Unknown targets**: TESS photometry (SPOC pipeline) from MAST

---

## Key Design Choices

1. **Ensemble Soft-Voting**: We combine XGBoost, LightGBM, and CatBoost. Soft-voting provides more robust probability estimates, especially for borderline "Blend" targets which share characteristics with both transits and EBs.
2. **No BLS/TLS during training**: For labeled KOIs, the period is already known from the NASA catalog, so we fold directly — no redundant search.
3. **TLS for unknown targets**: Transit Least Squares models a realistic limb-darkened transit shape, giving better sensitivity than BLS. Falls back to `astropy.timeseries.BoxLeastSquares` if TLS fails.
4. **Clean train/test separation**: Metrics are reported ONLY on the 25% held-out test set. 
5. **Explainability**: Feature importance directly ties the models' decisions to physically interpretable astrophysical diagnostics.
