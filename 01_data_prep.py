"""
==============================================================================
STEP 2 & 3: DATA COLLECTION + FEATURE EXTRACTION  (KEPLER KOI VERSION)
ISRO Bharatiya Antariksh Hackathon -- Problem Statement 7
Exoplanet Transit Signal Classification Pipeline
==============================================================================
Downloads the Kepler Cumulative KOI table from NASA Exoplanet Archive,
maps koi_fpflag_* flags to 4 class labels, downloads real Kepler light
curves from MAST, detrends & phase-folds at the known catalog period,
and extracts 19 shape-based features per target.

SCALE-UP v2: N_PER_CLASS=100 (400 total targets).
  - Pre-run timing estimate with 45-minute guard.
  - Fully resume-safe (already-cached KOIs skipped instantly).
  - Corrupt FITS files auto-detected and deleted for re-download.
  - Progress printed every 10 targets.
  - All failures logged to download_failures.csv.

CLASS MAPPING (Kepler KOI flags -- priority order):
  koi_fpflag_ss == 1              -> "Eclipsing Binary"
  koi_fpflag_co == 1 OR ec == 1  -> "Blend"
  koi_fpflag_nt == 1              -> "Other/Noise"
  No flags set                    -> "Transit"

OUTPUT:  training_features.csv   (appended incrementally, resume-safe)
         lc_cache/               (cached FITS files)
         failed_downloads.csv
         data_prep.log           (full progress log)
==============================================================================
"""

import os
import sys
import time
import csv
import warnings
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
warnings.filterwarnings("ignore")

# ==============================================================================
# Redirect stdout/stderr to a log file so Windows pipe-close can't crash us
# ==============================================================================
class TeeWriter:
    """Write to multiple streams; silently swallow errors on any one of them."""
    def __init__(self, *streams):
        self._streams = streams
    def write(self, data):
        for s in self._streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass
    def flush(self):
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass
    def fileno(self):          # needed by some libraries
        return self._streams[0].fileno()

_log_file = open("data_prep.log", "w", encoding="utf-8", buffering=1)
sys.stdout = TeeWriter(sys.__stdout__, _log_file)
sys.stderr = TeeWriter(sys.__stderr__, _log_file)

def safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except Exception:
        pass

# ── Output paths ──────────────────────────────────────────────────────────────
# NOTE: We do NOT use a custom FITS cache.  lightkurve caches downloads
# automatically in ~/.cache/lightkurve/mastDownload/ -- no Windows file-
# locking issues, no corrupt-partial-file problems.
FEATURES_CSV  = "training_features.csv"
FAILURES_CSV  = "download_failures.csv"
KOI_CSV       = "koi_table.csv"

# ── Sampling parameters ───────────────────────────────────────────────────────
N_PER_CLASS   = 100
RANDOM_SEED   = 42
SLEEP_BETWEEN = 0.5
MAX_WORKERS   = 5

# ── Feature extraction settings ───────────────────────────────────────────────
N_PHASE_BINS  = 200

# Feature column names (order matters -- must match 02_train_eval.py)
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
    "kepid", "koi_period", "koi_prad", "true_class",
]

# ==============================================================================
# STEP 2: Load Kepler KOI Table
# ==============================================================================
safe_print("=" * 65)
safe_print("STEP 2: Loading Kepler Cumulative KOI Table")
safe_print("=" * 65)

KOI_URL = (
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query="
    "select+kepid,koi_disposition,koi_period,koi_time0bk,"
    "koi_duration,koi_depth,koi_prad,koi_model_snr,"
    "koi_fpflag_nt,koi_fpflag_ss,koi_fpflag_co,koi_fpflag_ec"
    "+from+cumulative&format=csv"
)

if os.path.exists(KOI_CSV):
    safe_print(f"  [CACHE] Loading existing {KOI_CSV}")
    koi_table = pd.read_csv(KOI_CSV, comment="#")
else:
    safe_print("  Fetching from NASA Exoplanet Archive ...")
    koi_table = pd.read_csv(KOI_URL, comment="#")
    koi_table.to_csv(KOI_CSV, index=False)
    safe_print(f"  [OK] Downloaded {len(koi_table)} KOIs -> {KOI_CSV}")

safe_print(f"  Total KOIs: {len(koi_table)}")

koi_table.columns = [c.strip() for c in koi_table.columns]
for fc in ["koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec"]:
    koi_table[fc] = koi_table.get(fc, pd.Series([0]*len(koi_table))).fillna(0).astype(int)

def assign_class(row):
    if row["koi_fpflag_ss"] == 1:
        return "Eclipsing Binary"
    if row["koi_fpflag_co"] == 1 or row["koi_fpflag_ec"] == 1:
        return "Blend"
    if row["koi_fpflag_nt"] == 1:
        return "Other/Noise"
    return "Transit"

koi_table["true_class"] = koi_table.apply(assign_class, axis=1)

essential = ["kepid", "koi_period", "koi_time0bk", "koi_duration"]
koi_table = koi_table.dropna(subset=essential)
koi_table = koi_table[(koi_table["koi_period"] > 0) & (koi_table["koi_duration"] > 0)]
koi_table["kepid"] = koi_table["kepid"].astype(int)

safe_print("\n  Class distribution in full table:")
safe_print(koi_table["true_class"].value_counts().to_string())

# Stratified balanced sampling
rng = np.random.RandomState(RANDOM_SEED)
sampled_parts = []
for cls in ["Transit", "Eclipsing Binary", "Blend", "Other/Noise"]:
    subset = koi_table[koi_table["true_class"] == cls]
    n = min(N_PER_CLASS, len(subset))
    if n == 0:
        continue
    sampled_parts.append(subset.sample(n=n, random_state=rng))
    safe_print(f"  Sampling {n:3d} / {len(subset):5d} from '{cls}'")

sampled = pd.concat(sampled_parts).reset_index(drop=True)
safe_print(f"\n  Total sampled KOIs: {len(sampled)}")

# ==============================================================================
# STEP 3A: Feature extraction function
# ==============================================================================

def _to_float_array(x):
    """
    Safely convert an astropy Quantity, Column, or plain array to a plain
    numpy float64 array.  In lightkurve >= 2.0 phase/flux are Quantity objects;
    calling .value strips the units.
    """
    if hasattr(x, "value"):          # astropy Quantity / Time
        x = x.value
    if hasattr(x, "data"):           # MaskedColumn
        x = np.array(x.data, dtype=float)
    return np.asarray(x, dtype=float)


def extract_features(lc_flat, period, epoch, duration_hours, kepid=None):
    """
    Extract 10 shape-based features from a phase-folded detrended light curve.
    Works with lightkurve >= 2.0 (Quantity-aware).
    """
    try:
        # ── Phase-fold ────────────────────────────────────────────────────
        lc_fold = lc_flat.fold(period=period, epoch_time=epoch)

        # CRITICAL FIX: extract plain float arrays from Quantity objects
        phase = _to_float_array(lc_fold.phase)
        flux  = _to_float_array(lc_fold.flux)

        valid = np.isfinite(phase) & np.isfinite(flux)
        phase, flux = phase[valid], flux[valid]

        if len(phase) < 20:
            return None

        # ── Bin the folded curve ──────────────────────────────────────────
        bins        = np.linspace(phase.min(), phase.max(), N_PHASE_BINS + 1)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        bin_flux    = np.full(N_PHASE_BINS, np.nan)

        for i in range(N_PHASE_BINS):
            mask = (phase >= bins[i]) & (phase < bins[i + 1])
            if mask.sum() >= 2:
                bin_flux[i] = np.nanmedian(flux[mask])

        valid_bins = np.isfinite(bin_flux)
        if valid_bins.sum() < 5:
            return None

        bc = bin_centers[valid_bins]
        bf = bin_flux[valid_bins]

        # ── Transit window ────────────────────────────────────────────────
        duration_phase = duration_hours / 24.0
        half_dur       = duration_phase / 2.0

        oot_mask = np.abs(bc) > half_dur * 2.0
        if oot_mask.sum() < 3:
            oot_mask = np.abs(bc) > half_dur

        baseline_flux = np.nanmedian(bf[oot_mask]) if oot_mask.sum() > 0 else 1.0
        baseline_std  = np.nanstd(bf[oot_mask])   if oot_mask.sum() > 1 else 1e-3
        baseline_std  = max(float(baseline_std), 1e-6)
        baseline_flux = float(baseline_flux)

        in_mask = np.abs(bc) <= half_dur
        if in_mask.sum() < 1:
            in_mask = np.abs(bc) <= half_dur * 2

        # ── Transit depth ─────────────────────────────────────────────────
        transit_depth = float(baseline_flux - np.nanmin(bf[in_mask])) if in_mask.sum() > 0 else 0.0
        transit_depth = max(transit_depth, 0.0)

        # ── Measured transit duration ─────────────────────────────────────
        threshold = baseline_flux - 0.5 * transit_depth
        dip_mask  = bf < threshold
        if dip_mask.sum() >= 2:
            transit_duration_days = float(bc[dip_mask].max() - bc[dip_mask].min())
        else:
            transit_duration_days = float(duration_phase)
        transit_duration_hours = transit_duration_days * 24.0

        # ── Ingress / Egress ──────────────────────────────────────────────
        min_idx   = int(np.argmin(bf))
        min_phase = float(bc[min_idx])

        left_region  = bc <= min_phase
        right_region = bc >= min_phase

        def edge_duration(region_mask, direction="left"):
            bc_r  = bc[region_mask]
            bf_r  = bf[region_mask]
            above = bf_r >= threshold
            below = bf_r <= baseline_flux - 0.8 * transit_depth
            if above.sum() > 0 and below.sum() > 0:
                if direction == "left":
                    return float(abs(bc_r[below].min() - bc_r[above].max()) * 24.0)
                else:
                    return float(abs(bc_r[above].min() - bc_r[below].max()) * 24.0)
            return float(transit_duration_hours / 2.0)

        ingress_duration     = edge_duration(left_region,  "left")
        egress_duration      = edge_duration(right_region, "right")
        egress_duration      = max(egress_duration, 1e-6)
        ingress_egress_ratio = ingress_duration / egress_duration
        depth_duration_ratio = transit_depth / max(transit_duration_hours, 1e-6)

        # ── SNR ───────────────────────────────────────────────────────────
        snr = transit_depth / baseline_std

        # ── Flat-bottom fraction ──────────────────────────────────────────
        min_flux_val = float(bf[min_idx])
        flat_mask    = in_mask & (bf <= min_flux_val + baseline_std)
        flat_bottom_fraction = float(flat_mask.sum()) / float(in_mask.sum()) if in_mask.sum() > 0 else 0.0

        # ── Secondary eclipse depth & significance ───────────────────────
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
        # split folded transit at minimum
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
            # CRITICAL FIX: use .value on time and flux
            time_arr = _to_float_array(lc_flat.time)
            flux_arr = _to_float_array(lc_flat.flux)
            valid_r  = np.isfinite(time_arr) & np.isfinite(flux_arr)
            time_arr, flux_arr = time_arr[valid_r], flux_arr[valid_r]

            phase_raw   = ((time_arr - float(epoch)) / float(period)) % 1.0
            phase_raw[phase_raw > 0.5] -= 1.0
            transit_num = np.floor((time_arr - float(epoch)) / float(period)).astype(int)
            intransit   = np.abs(phase_raw) <= (half_dur / float(period))

            odd_d, even_d = [], []
            oot_baseline  = float(np.nanmedian(flux_arr[~intransit])) if (~intransit).sum() > 0 else 1.0

            unique_transits = np.unique(transit_num[intransit])
            transit_count = len(unique_transits)

            for tn in unique_transits:
                mask_tn = intransit & (transit_num == tn)
                if mask_tn.sum() >= 2:
                    depth_tn = oot_baseline - float(np.nanmedian(flux_arr[mask_tn]))
                    (odd_d if tn % 2 != 0 else even_d).append(depth_tn)

            all_depths = odd_d + even_d
            if len(all_depths) > 1 and np.mean(all_depths) > 0:
                depth_consistency_across_transits = float(np.std(all_depths) / np.mean(all_depths))
            else:
                depth_consistency_across_transits = np.nan

            if odd_d and even_d:
                mean_d = (np.mean(odd_d) + np.mean(even_d)) / 2.0
                odd_even_depth_diff = float(abs(np.mean(odd_d) - np.mean(even_d)) / max(abs(mean_d), 1e-9))
                
                odd_unc = baseline_std / np.sqrt(len(odd_d))
                even_unc = baseline_std / np.sqrt(len(even_d))
                combined_unc = np.sqrt(odd_unc**2 + even_unc**2)
                odd_even_depth_significance = float(abs(np.mean(odd_d) - np.mean(even_d)) / max(combined_unc, 1e-9))
        except Exception:
            odd_even_depth_diff = 0.0
            odd_even_depth_significance = 0.0
            transit_count = 0
            depth_consistency_across_transits = np.nan

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
        }

    except Exception as exc:
        safe_print(f"    [WARN] extract_features failed KIC {kepid}: {type(exc).__name__}: {exc}")
        return None


# ==============================================================================
# STEP 3B: Download & process one KOI
# ==============================================================================

def process_one_koi(row):
    """
    Download (or load from cache) a Kepler light curve, clean/detrend/fold,
    extract features. Returns (feature_dict, None) or (None, error_string).
    """
    import lightkurve as lk

    kepid  = int(row["kepid"])
    period = float(row["koi_period"])
    epoch  = float(row["koi_time0bk"])
    dur_h  = float(row["koi_duration"])
    cls    = str(row["true_class"])
    prad   = float(row["koi_prad"]) if pd.notna(row.get("koi_prad")) else float("nan")

    try:
        # ── Download via lightkurve (uses its own ~/.cache/lightkurve/ cache) ──
        # We intentionally do NOT maintain a separate custom FITS cache to avoid
        # Windows file-locking (PermissionError WinError 32) issues.
        time.sleep(SLEEP_BETWEEN)
        search = lk.search_lightcurve(
            f"KIC {kepid}", mission="Kepler", author="Kepler"
        )
        if len(search) == 0:
            return None, f"KIC {kepid}: no light curves on MAST"
        # download(cache=True) uses lightkurve's built-in MAST cache
        try:
            lc_raw = search[0].download(cache=True)
        except Exception as dl_exc:
            # Auto-delete corrupt cache files so the next run can re-download
            exc_str = str(dl_exc)
            if "corrupt" in exc_str.lower() or "interrupted" in exc_str.lower():
                # Extract the bad file path from the error message
                import re as _re
                match = _re.search(r"product (.+?) of type", exc_str)
                if match:
                    bad_path = match.group(1).strip()
                    try:
                        if os.path.exists(bad_path):
                            os.remove(bad_path)
                            safe_print(f"    [AUTO-DELETE] Removed corrupt cache: {bad_path}")
                    except Exception:
                        pass
            return None, f"KIC {kepid}: download error: {type(dl_exc).__name__}: {dl_exc}"
        if lc_raw is None:
            return None, f"KIC {kepid}: download returned None"

        # ── Clean ─────────────────────────────────────────────────────────
        lc = lc_raw.remove_nans().remove_outliers(sigma=5).normalize()

        if len(lc) < 50:
            return None, f"KIC {kepid}: only {len(lc)} clean points"

        # ── Detrend ───────────────────────────────────────────────────────
        wl = min(401, len(lc) - 2)
        if wl % 2 == 0:
            wl -= 1
        wl = max(wl, 11)
        lc_flat, _ = lc.flatten(window_length=wl, return_trend=True)

        # ── Extract features ──────────────────────────────────────────────
        feats = extract_features(lc_flat, period, epoch, dur_h, kepid=kepid)
        if feats is None:
            return None, f"KIC {kepid}: feature extraction returned None"

        feats.update({
            "kepid":      kepid,
            "koi_period": round(period, 6),
            "koi_prad":   prad,
            "true_class": cls,
        })
        return feats, None

    except Exception as exc:
        return None, f"KIC {kepid}: {type(exc).__name__}: {exc}"


# ==============================================================================
# STEP 3C: Parallel download + incremental CSV save
# ==============================================================================
safe_print("\n" + "=" * 65)
safe_print("STEP 3: Downloading Kepler LCs & extracting features")
safe_print(f"  Targets: {len(sampled)} | Workers: {MAX_WORKERS} | Cache: lightkurve built-in (~/.cache/lightkurve/)")
safe_print("=" * 65)

# Resume support: skip kepids already in features CSV
already_done = set()
if os.path.exists(FEATURES_CSV):
    try:
        existing_df = pd.read_csv(FEATURES_CSV)
        already_done = set(existing_df["kepid"].astype(int).tolist())
        safe_print(f"  [RESUME] {len(already_done)} already processed -- skipping")
    except Exception:
        pass

todo = sampled[~sampled["kepid"].isin(already_done)]
safe_print(f"  Remaining to download: {len(todo)}")

# ── Pre-run timing estimate ───────────────────────────────────────────────────
# Based on observed ~3s/target from cached+new mix; uncached ~5s/target.
# Use 4s/target as a conservative estimate for new downloads.
_cached_kic_dir = os.path.join(os.path.expanduser("~"), ".cache", "lightkurve", "mastDownload", "Kepler")
_n_cached = sum(1 for dp, dn, fn in os.walk(_cached_kic_dir) for _ in fn) if os.path.exists(_cached_kic_dir) else 0
_estimate_per_target_s = 2.0 if len(already_done) > 10 else 4.5
_est_total_s = len(todo) * _estimate_per_target_s / MAX_WORKERS
_est_min = _est_total_s / 60.0
safe_print(f"\n  Estimated download time: ~{_est_min:.0f} min ({_estimate_per_target_s:.1f}s/target, {MAX_WORKERS} workers)")
safe_print(f"  (Lightkurve MAST cache has ~{_n_cached} FITS files; cached targets are near-instant)")

if _est_min > 45 and len(todo) > 0:
    safe_print(f"\n  [WARNING] Estimated runtime ({_est_min:.0f} min) exceeds 45-minute threshold!")
    safe_print(f"  [WARNING] Consider reducing N_PER_CLASS (e.g. to 60) for a faster intermediate run.")
    safe_print(f"  [WARNING] Proceeding anyway -- stop the script if you change your mind.\n")

# Prepare incremental CSV writer
csv_exists    = os.path.exists(FEATURES_CSV)
_csv_fh       = open(FEATURES_CSV, "a", newline="", encoding="utf-8")
_csv_writer   = None   # initialized on first successful feature row

failure_rows  = []
n_ok          = 0
n_fail        = 0
processed     = 0
_run_start    = time.time()

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(process_one_koi, row): row for _, row in todo.iterrows()}

    for future in as_completed(futures):
        processed += 1
        feats, err = None, None
        try:
            feats, err = future.result()
        except Exception as fe:
            err = str(fe)

        if feats is not None and err is None:
            n_ok += 1
            # ── Write this row immediately to CSV ─────────────────────────
            try:
                if _csv_writer is None:
                    _csv_writer = csv.DictWriter(
                        _csv_fh,
                        fieldnames=list(feats.keys()),
                        extrasaction="ignore",
                    )
                    if not csv_exists or os.path.getsize(FEATURES_CSV) == 0:
                        _csv_writer.writeheader()
                _csv_writer.writerow(feats)
                _csv_fh.flush()
            except Exception as we:
                safe_print(f"    [WARN] CSV write error: {we}")

            # Progress: every 10 targets or first 5
            if processed % 10 == 0 or processed <= 5:
                elapsed = time.time() - _run_start
                rate = processed / elapsed if elapsed > 0 else 0
                eta_s = (len(todo) - processed) / rate if rate > 0 else 0
                safe_print(
                    f"  [{processed:>3}/{len(todo)}] OK "
                    f"KIC {feats['kepid']} | {feats['true_class']} "
                    f"| depth={feats['transit_depth']:.5f} | snr={feats['snr']:.1f}"
                    f"  (ok={n_ok} fail={n_fail} eta~{eta_s/60:.1f}min)"
                )
        else:
            n_fail += 1
            try:
                row_data = futures[future]
                kepid_str = str(row_data["kepid"])
            except Exception:
                kepid_str = "unknown"
            failure_rows.append({"kepid": kepid_str, "error": str(err or "unknown")})
            if processed % 10 == 0 or processed <= 5:
                safe_print(f"  [{processed:>3}/{len(todo)}] FAIL  {str(err or '')[:85]}")

_csv_fh.close()

# Save failures
if failure_rows:
    try:
        fail_df = pd.DataFrame(failure_rows)
        if os.path.exists(FAILURES_CSV):
            old = pd.read_csv(FAILURES_CSV)
            fail_df = pd.concat([old, fail_df], ignore_index=True)
        fail_df.to_csv(FAILURES_CSV, index=False)
    except Exception as fe:
        safe_print(f"  [WARN] Could not save failures CSV: {fe}")

# ==============================================================================
# Final summary
# ==============================================================================
safe_print("\n" + "=" * 65)
safe_print("STEP 3 COMPLETE")
safe_print("=" * 65)

try:
    if os.path.exists(FEATURES_CSV):
        final_df = pd.read_csv(FEATURES_CSV)
        safe_print(f"  Total feature rows : {len(final_df)}")
        safe_print(f"  Class distribution :")
        safe_print(final_df["true_class"].value_counts().to_string())
        safe_print(f"\n  Feature columns: {list(final_df.columns)}")
        safe_print(f"\n  Sample (first 3 rows):")
        pd.set_option("display.max_columns", 20)
        pd.set_option("display.width", 120)
        safe_print(final_df.head(3).to_string())
    else:
        safe_print("  [ERROR] training_features.csv not found!")

    safe_print(f"\n  OK: {n_ok} | FAIL: {n_fail} | Total attempted: {len(todo)}")
    lk_cache = os.path.join(os.path.expanduser("~"), ".cache", "lightkurve", "mastDownload", "Kepler")
    lk_fits  = sum(1 for dp, dn, fn in os.walk(lk_cache) for f in fn if f.endswith(".fits")) if os.path.exists(lk_cache) else 0
    safe_print(f"  Lightkurve MAST cache FITS: {lk_fits}")
except Exception as se:
    safe_print(f"  [WARN] Summary failed: {se}")

safe_print("\n  -> Next step: python 02_train_eval.py")

try:
    _log_file.close()
except Exception:
    pass
