import streamlit as st
import pandas as pd
import os
from PIL import Image

# ------------------------------------------------------------------------------
# Page Configuration & Theming
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Exoplanet Pipeline Dashboard",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for an extra "space" feel
st.markdown("""
<style>
    /* Neon glow for headers */
    h1, h2, h3 {
        color: #E0E6ED;
        text-shadow: 0 0 10px rgba(155, 93, 229, 0.5);
    }
    
    /* Highlight text styling */
    .highlight-yellow {
        color: #F9E076;
        font-weight: bold;
    }
    
    /* Table hover effect */
    .stDataFrame {
        border: 1px solid #2A3143;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------
@st.cache_data
def load_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def highlight_needs_review(row):
    """
    Highlights rows in the unknown target predictions table that need human review.
    Uses an astronomical amber/orange for attention in dark mode.
    """
    color = 'background-color: rgba(255, 153, 0, 0.25); color: #FFD166;' if row.get('needs_review', False) else ''
    return [color] * len(row)

# ------------------------------------------------------------------------------
# Sidebar Navigation
# ------------------------------------------------------------------------------
with st.sidebar:
    st.image("logo.jpg", use_container_width=True)
    st.title("VyomVoyage")
    st.markdown("---")
    
    nav = st.radio(
        "Navigation",
        ["🌌 Pipeline Overview", "📊 Model Performance", "🔭 Unknown TESS Targets"]
    )
    
    st.markdown("---")
    st.info("""
    **ISRO Bharatiya Antariksh Hackathon**
    Problem Statement 7: AI-enabled Detection of Exoplanets from Noisy Astronomical Light Curves.
    """)

# ------------------------------------------------------------------------------
# View 1: Pipeline Overview
# ------------------------------------------------------------------------------
if nav == "🌌 Pipeline Overview":
    st.title("🌌 Exoplanet Pipeline Overview")
    
    st.markdown("""
    This dashboard visualizes the results of an end-to-end Machine Learning pipeline that identifies exoplanet transits from noisy astronomical light curves. 
    
    The pipeline downloads Kepler KOI data, extracts 19 physics-motivated features, and trains an Ensemble Classifier (XGBoost, LightGBM, CatBoost) using Soft-Voting.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Signal Classes")
        st.markdown("""
        * **Transit**: Genuine exoplanet transit (flat-bottomed, symmetric)
        * **Eclipsing Binary**: Stellar eclipse (odd/even depth differences)
        * **Blend**: Contamination from a nearby source (shallow, V-shaped)
        * **Other/Noise**: Irregular shape, asymmetric
        """)
        
    with col2:
        st.subheader("Scale-Up Results (N=20 vs N=100)")
        scale_df = load_csv("scale_comparison.csv")
        if scale_df is not None:
            st.dataframe(scale_df, use_container_width=True, hide_index=True)
            st.caption("Training on N=100 targets per class yielded significant boosts to Macro-F1 and Inter-Model Agreement.")
        else:
            st.warning("`scale_comparison.csv` not found.")

# ------------------------------------------------------------------------------
# View 2: Model Performance
# ------------------------------------------------------------------------------
elif nav == "📊 Model Performance":
    st.title("📊 Model Performance (N=100 Ensemble)")
    
    model_df = load_csv("model_comparison.csv")
    if model_df is not None:
        st.subheader("Performance Metrics (Held-out Test Set)")
        st.dataframe(model_df.style.highlight_max(subset=['Accuracy', 'Macro-F1'], color='rgba(155, 93, 229, 0.4)'), use_container_width=True)
    
    st.markdown("---")
    st.subheader("Visual Diagnostics")
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("**Confusion Matrix**")
        if os.path.exists("confusion_matrix.png"):
            st.image("confusion_matrix.png", use_container_width=True)
            
    with c2:
        st.markdown("**ROC Curves**")
        if os.path.exists("roc_curves.png"):
            st.image("roc_curves.png", use_container_width=True)
            
    with c3:
        st.markdown("**Feature Importance (XGBoost)**")
        if os.path.exists("feature_importance.png"):
            st.image("feature_importance.png", use_container_width=True)

# ------------------------------------------------------------------------------
# View 3: Unknown TESS Targets
# ------------------------------------------------------------------------------
elif nav == "🔭 Unknown TESS Targets":
    st.title("🔭 Predictions on Unknown TESS Targets")
    st.markdown("""
    The trained ensemble model is applied to unseen Light Curves from the TESS mission. 
    Periods are detected automatically using **Transit Least Squares (TLS)**.
    """)
    
    pred_df = load_csv("unknown_target_predictions.csv")
    
    if pred_df is not None:
        st.subheader("Predictions Table")
        st.markdown("Rows marked <span style='color:#FFD166;'>**Orange**</span> indicate that models disagreed or confidence is low, and the target **Needs Human Review** by astronomers.", unsafe_allow_html=True)
        
        # Apply the styling function to highlight rows needing review
        styled_df = pred_df.style.apply(highlight_needs_review, axis=1).format({
            "confidence": "{:.1%}",
            "period": "{:.4f} d",
            "duration_hrs": "{:.2f} h",
            "snr": "{:.2f}"
        }, na_rep="—")
        
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.subheader("Target Inspection")
        
        # Filter out FAILED targets for the selectbox
        valid_targets = pred_df[pred_df['predicted_class'] != 'FAILED']['tic_id'].astype(str).tolist()
        
        if valid_targets:
            selected_tic = st.selectbox("Select a TIC ID to view its phase-folded light curve:", valid_targets)
            
            plot_path = os.path.join("unknown_lc_plots", f"tic_{selected_tic}_prediction.png")
            if os.path.exists(plot_path):
                st.image(plot_path, caption=f"Phase-folded Light Curve for TIC {selected_tic}")
            else:
                st.warning(f"Plot not found for TIC {selected_tic}")
        else:
            st.info("No successful predictions to inspect.")
    else:
        st.warning("`unknown_target_predictions.csv` not found. Please run the inference script first.")
