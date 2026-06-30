"""
==============================================================================
STEP 1: ENVIRONMENT SETUP & DEPENDENCY INSTALLATION
ISRO Bharatiya Antariksh Hackathon — Problem Statement 7
Exoplanet Transit Signal Classification Pipeline
==============================================================================
Run this script FIRST before any other script in the pipeline.
It installs all required packages and verifies they can be imported.
"""

import subprocess
import sys

def install_package(package, fallback=None):
    """Try to install a package, optionally falling back to another."""
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print(f"  [OK] Installed: {package}")
        return True
    except subprocess.CalledProcessError:
        if fallback:
            print(f"  [WARN] Failed to install {package}, trying fallback: {fallback}")
            return install_package(fallback)
        print(f"  [FAIL] Could not install: {package}")
        return False

print("=" * 60)
print("Installing dependencies for Exoplanet Pipeline...")
print("=" * 60)

packages = [
    "lightkurve",
    "astropy",
    "xgboost",
    "scikit-learn",
    "pandas",
    "numpy",
    "matplotlib",
    "seaborn",
    "scipy",
    "lightgbm",
    "catboost",
    "streamlit",
]

for pkg in packages:
    install_package(pkg)

# Try TLS; if it fails, we fall back to astropy BLS at runtime
print("\nAttempting to install transitleastsquares (TLS)...")
tls_ok = install_package("transitleastsquares")
if tls_ok:
    try:
        import transitleastsquares  # noqa: F401
        print("  [OK] TLS import verified — will use TLS for unknown targets.")
    except Exception as e:
        tls_ok = False
        print(f"  [WARN] TLS installed but import failed ({e}) — will use astropy BLS.")

if not tls_ok:
    print("  [INFO] Will use astropy.timeseries.BoxLeastSquares as the period-search fallback.")

print("\n" + "=" * 60)
print("Verifying imports...")
print("=" * 60)

import_ok = True
for mod in ["lightkurve", "astropy", "xgboost", "sklearn", "pandas", "numpy", "matplotlib", "seaborn", "lightgbm", "catboost"]:
    try:
        __import__(mod)
        print(f"  [OK] {mod}")
    except ImportError as e:
        print(f"  [FAIL] {mod}: {e}")
        import_ok = False

if import_ok:
    print("\nAll required packages are available. Proceed to Step 2: python 01_data_prep.py")
else:
    print("\nSome imports failed. Please resolve the above issues before proceeding.")
    sys.exit(1)
