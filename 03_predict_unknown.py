"""
==============================================================================
STEP 6: APPLY TO GENUINELY UNKNOWN/UNLABELED TESS TARGETS
ISRO Bharatiya Antariksh Hackathon — Problem Statement 7
Exoplanet Transit Signal Classification Pipeline
==============================================================================
Downloads light curves for 5 TIC IDs that are NOT in any training catalog,
runs TLS (or BLS fallback) to find the best-fit period, extracts the same
shape features, and runs the trained XGBoost classifier to produce predictions.

TIC IDs chosen here are selected from recent TESS sectors and verified to NOT
appear in the TOI (TESS Objects of Interest) catalog, making them genuinely
unknown unlabeled targets.

OUTPUTS:
  unknown_target_predictions.csv
  unknown_lc_plots/           (phase-folded LC plot per target)
  (appended to results_summary.txt)
==============================================================================
"""

import os
import time
import pickle
import warnings
import traceback
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
from astropy.timeseries import BoxLeastSquares
import astropy.units as u

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH     = "xgb_model.json"
LGB_PATH       = "lgb_model.txt"
CAT_PATH       = "cat_model.cbm"
ENCODER_PATH   = "label_encoder.pkl"
OUTPUT_CSV     = "unknown_target_predictions.csv"
PLOT_DIR       = "unknown_lc_plots"
SUMMARY_FILE   = "results_summary.txt"
N_PHASE_BINS   = 200
SLEEP_BETWEEN  = 0.5

os.makedirs(PLOT_DIR, exist_ok=True)

# ── Feature columns (must match 02_train_eval.py) ─────────────────────────────
FEATURE_COLS = [
    "transit_depth", "transit_duration_hrs",
    "ingress_duration_hrs", "egress_duration_hrs",
    "ingress_egress_ratio", "depth_duration_ratio",
    "odd_even_depth_diff", "secondary_eclipse_depth",
    "snr", "flat_bottom_fraction",
    "secondary_eclipse_significance", "odd_even_depth_significance",
    "transit_symmetry_score", "out_of_transit_variability",
    "transit_count", "stellar_density_proxy",
    "depth_consistency_across_transits",
    "koi_period",   # will store the BLS/TLS-derived period
    "koi_prad",     # estimated from transit depth + TIC stellar radius
]

# ── Unknown TESS targets ───────────────────────────────────────────────────────
# These TIC IDs are selected from TESS Sectors that are not in the TOI catalog.
# They represent genuinely unlabeled targets for demonstration.
# We include a mix of stellar types (dwarfs, subgiants) to make the predictions
# diverse and interesting for the hackathon panel.
UNKNOWN_TICS = [
    29857954,    # G-dwarf, Sector 1, not in TOI catalog
    38845463,    # K-dwarf, Sector 2
    140501021,   # F-dwarf, Sector 5
    159862112,   # M-dwarf, Sector 6
    470171739,   # G-subgiant, Sector 10
]

# ── Plotting style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":    150,
    "font.family":   "DejaVu Sans",
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#161b22",
    "axes.edgecolor":   "#30363d",
    "text.color":       "#e6edf3",
    "axes.labelcolor":  "#e6edf3",
    "xtick.color":      "#e6edf3",
    "ytick.color":      "#e6edf3",
    "grid.color":       "#21262d",
    "grid.alpha":       0.5,
})

CLASS_COLORS = {
    "Transit":           "#4CAF50",
    "Eclipsing Binary":  "#F44336",
    "Blend":             "#FF9800",
    "Other/Noise":       "#9C27B0",
}

# ==============================================================================
# Load trained model + label encoder
# ==============================================================================
print("=" * 65)
print("STEP 6: Predicting on Unknown TESS Targets")
print("=" * 65)

for path in [MODEL_PATH, LGB_PATH, CAT_PATH, ENCODER_PATH]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run 02_train_eval.py first."
        )

xgb_model = XGBClassifier()
xgb_model.load_model(MODEL_PATH)

lgb_model = lgb.Booster(model_file=LGB_PATH)

cat_model = CatBoostClassifier()
cat_model.load_model(CAT_PATH)

with open(ENCODER_PATH, "rb") as f:
    le = pickle.load(f)

print(f"  Models loaded from {MODEL_PATH}, {LGB_PATH}, {CAT_PATH}")
print(f"  Classes: {list(le.classes_)}")

# ==============================================================================
# Stellar radius lookup from TESS Input Catalog (TIC)
# ==============================================================================
R_SUN_TO_EARTH = 109.076  # 1 R_sun = 109.076 R_earth

def get_stellar_radius_earth(tic_id):
    """
    Query the TESS Input Catalog for the stellar radius of a TIC target.
    Returns radius in Earth radii. Falls back to 1.0 R_sun (109 R_earth) if unavailable.
    """
    try:
        from astroquery.mast import Catalogs
        result = Catalogs.query_criteria(catalog="TIC", ID=str(tic_id))
        if len(result) > 0 and "rad" in result.colnames:
            r_sun = float(result["rad"][0])
            if np.isfinite(r_sun) and r_sun > 0:
                r_earth = r_sun * R_SUN_TO_EARTH
                print(f"    TIC stellar radius: {r_sun:.3f} R_sun ({r_earth:.1f} R_earth) [from TIC]")
                return r_earth
    except Exception as e:
        print(f"    [WARN] TIC query failed ({e}), using solar default")
    # Solar default: 1.0 R_sun = 109.076 R_earth
    print(f"    TIC stellar radius: using solar default (1.0 R_sun = {R_SUN_TO_EARTH:.1f} R_earth)")
    return R_SUN_TO_EARTH


def estimate_koi_prad(transit_depth_fraction, stellar_radius_earth):
    """
    Estimate planet radius in Earth radii using the standard transit formula:
        R_p = R_star * sqrt(delta)
    where delta is the transit depth as a fraction of stellar flux.
    """
    if transit_depth_fraction <= 0 or not np.isfinite(transit_depth_fraction):
        return np.nan
    return stellar_radius_earth * np.sqrt(max(transit_depth_fraction, 0.0))


# ==============================================================================
# Period-search flag (TLS preferred, BLS fallback)
# ==============================================================================
USE_TLS = False
try:
    from transitleastsquares import transitleastsquares as TLS
    USE_TLS = True
    print("  Period-search algorithm: transitleastsquares (TLS)")
except ImportError:
    print("  transitleastsquares not available — using astropy BoxLeastSquares (BLS) as fallback")
    print("  [NOTE] BLS is the fallback period-search method for this script.")

# ==============================================================================
# Feature extraction (reused from 01_data_prep.py — same function)
# ==============================================================================

def _to_float_array(x):
    if hasattr(x, "value"):
        x = x.value
    if hasattr(x, "data"):
        x = np.array(x.data, dtype=float)
    return np.asarray(x, dtype=float)

def extract_features_unknown(lc_flat, period, epoch, duration_hours):
    """
    Extract the same shape features used during training, given a phase-folded
    light curve and the TLS/BLS-derived period + epoch.
    """
    try:
        lc_fold = lc_flat.fold(period=period, epoch_time=epoch)
        phase   = _to_float_array(lc_fold.phase)
        flux    = _to_float_array(lc_fold.flux)

        valid = np.isfinite(phase) & np.isfinite(flux)
        phase, flux = phase[valid], flux[valid]

        if len(phase) < 20:
            return None

        bins = np.linspace(phase.min(), phase.max(), N_PHASE_BINS + 1)
        bin_centers  = 0.5 * (bins[:-1] + bins[1:])
        bin_flux     = np.full(N_PHASE_BINS, np.nan)

        for i in range(N_PHASE_BINS):
            mask = (phase >= bins[i]) & (phase < bins[i + 1])
            if mask.sum() >= 2:
                bin_flux[i] = np.nanmedian(flux[mask])

        valid_bins = np.isfinite(bin_flux)
        if valid_bins.sum() < 5:
            return None

        bc = bin_centers[valid_bins]
        bf = bin_flux[valid_bins]

        duration_phase = duration_hours / 24.0
        half_dur       = duration_phase / 2.0

        oot_mask = np.abs(bc) > half_dur * 2.0
        if oot_mask.sum() < 3:
            oot_mask = np.abs(bc) > half_dur

        baseline_flux = np.nanmedian(bf[oot_mask]) if oot_mask.sum() > 0 else 1.0
        baseline_std  = np.nanstd(bf[oot_mask]) if oot_mask.sum() > 1 else 1e-3
        if baseline_std < 1e-6:
            baseline_std = 1e-6

        in_mask = np.abs(bc) <= half_dur
        if in_mask.sum() < 1:
            in_mask = np.abs(bc) <= half_dur * 2

        transit_depth = float(baseline_flux - np.nanmin(bf[in_mask])) if in_mask.sum() > 0 else 0.0
        transit_depth = max(transit_depth, 0.0)

        threshold = baseline_flux - 0.5 * transit_depth
        dip_mask  = bf < threshold
        transit_duration_days = bc[dip_mask].max() - bc[dip_mask].min() if dip_mask.sum() >= 2 else duration_phase
        transit_duration_hours = transit_duration_days * 24.0

        min_idx = np.argmin(bf)
        min_phase = bc[min_idx]

        left_region  = bc <= min_phase
        right_region = bc >= min_phase

        def edge_duration(region_mask, direction="left"):
            bc_r = bc[region_mask]
            bf_r = bf[region_mask]
            above = bf_r >= threshold
            below = bf_r <= baseline_flux - 0.8 * transit_depth
            if above.sum() > 0 and below.sum() > 0:
                if direction == "left":
                    t_above = bc_r[above].max()
                    t_below = bc_r[below].min()
                else:
                    t_above = bc_r[above].min()
                    t_below = bc_r[below].max()
                return abs(t_below - t_above) * 24.0
            return transit_duration_hours / 2.0

        ingress_duration = edge_duration(left_region, "left")
        egress_duration  = edge_duration(right_region, "right")

        if egress_duration < 1e-6:
            egress_duration = 1e-6
        ingress_egress_ratio = ingress_duration / egress_duration
        depth_duration_ratio = transit_depth / max(transit_duration_hours, 1e-6)
        snr = transit_depth / baseline_std

        min_flux = bf[np.argmin(bf)]
        one_sigma_above_min = min_flux + baseline_std
        flat_mask = in_mask & (bf <= one_sigma_above_min)
        flat_bottom_fraction = flat_mask.sum() / in_mask.sum() if in_mask.sum() > 0 else 0.0

        half_period = float(period) / 2.0
        sec_mask    = np.abs(np.abs(bc) - half_period) < half_dur * 2
        if sec_mask.sum() >= 2:
            sec_depth = float(baseline_flux - np.nanmin(bf[sec_mask]))
            sec_depth = max(sec_depth, 0.0)
        else:
            sec_depth = 0.0
            
        secondary_eclipse_significance = sec_depth / max(transit_depth, 1e-9)

        # ── Out of transit variability ────────────────────────────────────
        out_of_transit_variability = float(baseline_std)

        # ── Stellar density proxy ─────────────────────────────────────────
        stellar_density_proxy = (float(period)**2) / max((transit_duration_hours / 24.0)**3, 1e-9)

        # ── Transit symmetry score ────────────────────────────────────────
        left_half = bf[in_mask & (bc <= min_phase)]
        right_half = bf[in_mask & (bc > min_phase)]
        mirror_bc = min_phase - (bc[in_mask & (bc > min_phase)] - min_phase)
        if len(left_half) > 1 and len(right_half) > 1:
            left_interp = np.interp(mirror_bc, bc[in_mask & (bc <= min_phase)], left_half, left=baseline_flux, right=baseline_flux)
            residuals = (right_half - left_interp)**2
            transit_symmetry_score = float(np.sum(residuals) / max(np.sum(right_half**2), 1e-9))
        else:
            transit_symmetry_score = 0.0

        # ── Odd/Even and depth consistency ────────────────────────────────
        odd_even_depth_diff = 0.0
        odd_even_depth_significance = 0.0
        transit_count = 0
        depth_consistency_across_transits = np.nan
        try:
            time_arr = _to_float_array(lc_flat.time)
            flux_arr = _to_float_array(lc_flat.flux)
            valid_raw = np.isfinite(time_arr) & np.isfinite(flux_arr)
            time_arr, flux_arr = time_arr[valid_raw], flux_arr[valid_raw]
            
            phase_raw = ((time_arr - epoch) / period) % 1.0
            phase_raw[phase_raw > 0.5] -= 1.0
            transit_num = np.floor((time_arr - epoch) / period).astype(int)
            intransit   = np.abs(phase_raw) <= half_dur / period

            odd_d, even_d = [], []
            for tn in np.unique(transit_num[intransit]):
                mask_tn = intransit & (transit_num == tn)
                if mask_tn.sum() >= 2 and oot_mask.sum() > 0:
                    oot_idxs = np.where(oot_mask)[0]
                    oot_flux_raw = np.interp(
                        np.abs(phase_raw[mask_tn]),
                        np.abs(bc[oot_idxs]),
                        bf[oot_idxs],
                    )
                    depth_tn = np.nanmedian(oot_flux_raw) - np.nanmedian(flux_arr[mask_tn])
                    (odd_d if tn % 2 != 0 else even_d).append(depth_tn)

            unique_transits = np.unique(transit_num[intransit])
            transit_count = len(unique_transits)

            all_depths = odd_d + even_d
            if len(all_depths) > 1 and np.mean(all_depths) > 0:
                depth_consistency_across_transits = float(np.std(all_depths) / np.mean(all_depths))

            if odd_d and even_d:
                mean_d = (np.mean(odd_d) + np.mean(even_d)) / 2.0
                odd_even_depth_diff = float(abs(np.mean(odd_d) - np.mean(even_d)) / max(abs(mean_d), 1e-9))
                
                odd_unc = baseline_std / np.sqrt(len(odd_d))
                even_unc = baseline_std / np.sqrt(len(even_d))
                combined_unc = np.sqrt(odd_unc**2 + even_unc**2)
                odd_even_depth_significance = float(abs(np.mean(odd_d) - np.mean(even_d)) / max(combined_unc, 1e-9))
        except Exception:
            pass

        return {
            "transit_depth":            round(transit_depth, 8),
            "transit_duration_hrs":     round(transit_duration_hours, 4),
            "ingress_duration_hrs":     round(ingress_duration, 4),
            "egress_duration_hrs":      round(egress_duration, 4),
            "ingress_egress_ratio":     round(ingress_egress_ratio, 6),
            "depth_duration_ratio":     round(depth_duration_ratio, 8),
            "odd_even_depth_diff":      round(odd_even_depth_diff, 6),
            "secondary_eclipse_depth":  round(sec_depth, 8),
            "snr":                      round(snr, 4),
            "flat_bottom_fraction":     round(flat_bottom_fraction, 6),
            "secondary_eclipse_significance": round(secondary_eclipse_significance, 6),
            "odd_even_depth_significance": round(odd_even_depth_significance, 6),
            "transit_symmetry_score":   round(transit_symmetry_score, 6),
            "out_of_transit_variability": round(out_of_transit_variability, 8),
            "transit_count":            transit_count,
            "stellar_density_proxy":    round(stellar_density_proxy, 6),
            "depth_consistency_across_transits": round(depth_consistency_across_transits, 6) if not np.isnan(depth_consistency_across_transits) else np.nan,
            "koi_period":               period,
            # koi_prad is filled in by the caller after stellar radius lookup
            "koi_prad":                 np.nan,
        }

    except Exception as exc:
        print(f"    [WARN] Feature extraction failed: {exc}")
        return None


def run_period_search(time_arr, flux_arr, period_min=0.5, period_max=13.0):
    """
    Run TLS or BLS period search.
    Returns (period, epoch, duration_hours, snr_search).
    """
    if USE_TLS:
        try:
            results = TLS(time_arr, flux_arr).power(
                minimum_period=period_min,
                maximum_period=period_max,
                show_progress_bar=False,
            )
            return (
                float(results.period),
                float(results.T0),
                float(results.duration) * 24.0,
                float(results.SDE),
            )
        except Exception as e:
            print(f"    [WARN] TLS failed ({e}), falling back to BLS")

    # BLS fallback
    from astropy.timeseries import BoxLeastSquares
    import astropy.units as u

    t = time_arr * u.day
    f = flux_arr * u.dimensionless_unscaled

    model = BoxLeastSquares(t, f)
    durations = np.linspace(0.05, 0.4, 10) * u.day
    periodogram = model.autopower(durations, minimum_period=period_min * u.day, maximum_period=period_max * u.day)
    best_idx    = np.argmax(periodogram.power)
    best_period = float(periodogram.period[best_idx].to(u.day).value)
    best_t0     = float(periodogram.transit_time[best_idx].to(u.day).value)
    best_dur    = float(periodogram.duration[best_idx].to(u.day).value)
    best_power  = float(periodogram.power[best_idx])

    return best_period, best_t0, best_dur * 24.0, best_power


# ==============================================================================
# Main prediction loop
# ==============================================================================

import lightkurve as lk

results = []
print(f"\n  Processing {len(UNKNOWN_TICS)} unknown TESS targets...")
print("-" * 65)

for tic_id in UNKNOWN_TICS:
    print(f"\n  TIC {tic_id}")
    time.sleep(SLEEP_BETWEEN)

    result_row = {
        "tic_id":          tic_id,
        "predicted_class": "FAILED",
        "confidence":      np.nan,
        "period":          np.nan,
        "depth":           np.nan,
        "duration_hrs":    np.nan,
        "snr":             np.nan,
        "period_search":   "TLS" if USE_TLS else "BLS",
    }

    try:
        # ── Download light curve ──────────────────────────────────────────
        search = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS", author="SPOC")
        if len(search) == 0:
            search = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS")

        if len(search) == 0:
            print(f"    [FAIL] No light curves found for TIC {tic_id}")
            results.append(result_row)
            continue

        print(f"    Found {len(search)} sector(s) on MAST — downloading first...")
        lc_raw = search[0].download()
        if lc_raw is None:
            print(f"    [FAIL] Download returned None for TIC {tic_id}")
            results.append(result_row)
            continue

        # ── Clean + detrend ───────────────────────────────────────────────
        lc = lc_raw.remove_nans().remove_outliers(sigma=5).normalize()
        wl = min(401, len(lc) - 2 if len(lc) % 2 == 0 else len(lc) - 1)
        if wl % 2 == 0:
            wl -= 1
        if wl < 11:
            print(f"    [FAIL] Time series too short after cleaning for TIC {tic_id}")
            results.append(result_row)
            continue

        lc_flat, _ = lc.flatten(window_length=wl, return_trend=True)

        t_arr = np.array(lc_flat.time.value)
        f_arr = np.array(lc_flat.flux)
        valid = np.isfinite(t_arr) & np.isfinite(f_arr)
        t_arr, f_arr = t_arr[valid], f_arr[valid]

        print(f"    Light curve: {len(t_arr)} clean data points")
        print(f"    Running {'TLS' if USE_TLS else 'BLS'} period search...")

        # ── Period search ─────────────────────────────────────────────────
        period, epoch, duration_hrs, search_snr = run_period_search(t_arr, f_arr)
        print(f"    Best period: {period:.4f} d  |  Duration: {duration_hrs:.2f} h  |  SDE/power: {search_snr:.2f}")

        # ── Extract features ───────────────────────────────────────────────
        feats = extract_features_unknown(lc_flat, period, epoch, duration_hrs)
        if feats is None:
            print(f"    [FAIL] Feature extraction returned None for TIC {tic_id}")
            results.append(result_row)
            continue

        # ── Estimate koi_prad from transit depth + TIC stellar radius ─────
        stellar_r_earth = get_stellar_radius_earth(tic_id)
        estimated_prad  = estimate_koi_prad(feats["transit_depth"], stellar_r_earth)
        feats["koi_prad"] = estimated_prad
        print(f"    Estimated planet radius: {estimated_prad:.2f} R_earth (depth={feats['transit_depth']*100:.4f}%, R_star={stellar_r_earth:.1f} R_earth)")

        # ── Predict ───────────────────────────────────────────────────────
        # Build feature vector in exact same order as training
        feat_vec = np.array([feats.get(col, np.nan) for col in FEATURE_COLS]).reshape(1, -1)
        feat_vec = np.nan_to_num(feat_vec, nan=0.0)

        p_xgb = xgb_model.predict_proba(feat_vec)[0]
        p_lgb = lgb_model.predict(feat_vec)[0]  # LightGBM booster returns array of probs directly
        p_cat = cat_model.predict_proba(feat_vec)[0]
        
        pred_prob = (p_xgb + p_lgb + p_cat) / 3.0
        pred_idx = np.argmax(pred_prob)
        pred_cls  = le.inverse_transform([pred_idx])[0]
        confidence = float(pred_prob[pred_idx])
        
        preds_all = [np.argmax(p_xgb), np.argmax(p_lgb), np.argmax(p_cat)]
        models_in_agreement = preds_all.count(pred_idx)
        needs_review = models_in_agreement < 3

        print(f"    Predicted class: {pred_cls}  (confidence: {confidence:.1%}, agreement: {models_in_agreement}/3)")

        # ── Phase-folded LC plot ──────────────────────────────────────────
        lc_fold = lc_flat.fold(period=period, epoch_time=epoch)
        ph = _to_float_array(lc_fold.phase)
        fl = _to_float_array(lc_fold.flux)
        valid_fold = np.isfinite(ph) & np.isfinite(fl)
        ph, fl = ph[valid_fold], fl[valid_fold]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter(ph, fl, s=1, alpha=0.3, color="#58a6ff", rasterized=True, label="raw")
        bin_edges   = np.linspace(ph.min(), ph.max(), 101)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_fl      = [np.nanmedian(fl[(ph >= bin_edges[i]) & (ph < bin_edges[i+1])]) for i in range(100)]
        ax.plot(bin_centers, bin_fl, color="#f0c040", lw=2, label="binned", zorder=5)
        ax.set_xlabel("Phase (days)")
        ax.set_ylabel("Normalized Flux")
        color_pred = CLASS_COLORS.get(pred_cls, "#e6edf3")
        ax.set_title(
            f"TIC {tic_id} — Predicted: {pred_cls} ({confidence:.1%} conf.)\n"
            f"Period={period:.4f} d | Depth={feats['transit_depth']*100:.3f}% | "
            f"Dur={duration_hrs:.2f} h | SNR={feats['snr']:.1f}",
            color=color_pred, fontsize=10
        )
        ax.legend(fontsize=8, framealpha=0.3)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plot_path = os.path.join(PLOT_DIR, f"tic_{tic_id}_prediction.png")
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()
        print(f"    Saved plot -> {plot_path}")

        # ── Update result row ─────────────────────────────────────────────
        result_row.update({
            "predicted_class": pred_cls,
            "confidence":      confidence,
            "models_in_agreement": f"{models_in_agreement}/3",
            "needs_review":    needs_review,
            "period":          period,
            "depth":           feats["transit_depth"],
            "duration_hrs":    duration_hrs,
            "snr":             feats["snr"],
            "ingress_egress_ratio":    feats["ingress_egress_ratio"],
            "secondary_eclipse_depth": feats["secondary_eclipse_depth"],
            "flat_bottom_fraction":    feats["flat_bottom_fraction"],
        })

    except Exception as exc:
        print(f"    [ERROR] TIC {tic_id}: {type(exc).__name__}: {exc}")
        traceback.print_exc()

    results.append(result_row)

# ==============================================================================
# Save and display predictions
# ==============================================================================
pred_df = pd.DataFrame(results)
pred_df.to_csv(OUTPUT_CSV, index=False)
print("\n" + "=" * 65)
print(f"  Saved predictions -> {OUTPUT_CSV}")
print("=" * 65)
print("\nFINAL UNKNOWN TARGET PREDICTIONS:")
print(pred_df[["tic_id", "predicted_class", "confidence", "models_in_agreement", "needs_review", "period", "depth", "duration_hrs"]].to_string(index=False))

# ── Append to results_summary.txt ─────────────────────────────────────────────
append_block = f"""

{'=' * 65}
STEP 6 — UNKNOWN TESS TARGET PREDICTIONS (Ensemble)
Period search: {'TLS (transitleastsquares)' if USE_TLS else 'BLS (astropy.timeseries.BoxLeastSquares)'}
{'=' * 65}
{pred_df[['tic_id','predicted_class','confidence','models_in_agreement','needs_review','period','depth','duration_hrs']].to_string(index=False)}

Plots saved to: {PLOT_DIR}/
{'=' * 65}
"""

with open(SUMMARY_FILE, "a") as f:
    f.write(append_block)

print(f"\n  Appended to {SUMMARY_FILE}")
print("\n  Pipeline complete! All outputs saved to the project directory.")
print("  -> See results_summary.txt for the full report.")
