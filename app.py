import streamlit as st
import joblib  # Built-in module, no extra installation required!
import numpy as np

# ---------------------------------------------------------
# 1. Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# 2. Custom CSS Styling
# ---------------------------------------------------------
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #f8f9fa;
        color: #212529;
    }
    
    /* Hero Header Container */
    .header-container {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    
    .header-container h1 {
        color: white !important;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    /* Input Field Styling */
    div[data-baseweb="input"] {
        border-radius: 8px !important;
    }
    
    /* Result Box Card */
    .metric-card {
        background-color: #ffffff;
        border-left: 6px solid #2563eb;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-top: 1.5rem;
    }
    
    /* Custom Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e5e7eb;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 3. Model Loading (Using Built-in Pickle)
# ---------------------------------------------------------
MODEL_PATH = "best_regression_model.pkl"

@st.cache_resource
def load_model():
    """Load model using joblib to match the file's binary format."""
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except FileNotFoundError:
        st.error(f"⚠️ Could not find model file `{MODEL_PATH}` in your repository root directory.")
        return None
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        return None

model = load_model()

# ---------------------------------------------------------
# 4. Hero Header Section
# ---------------------------------------------------------
st.markdown("""
    <div class="header-container">
        <h1>🏠 House Price Analytics & Prediction</h1>
        <p style="font-size: 1.1rem; opacity: 0.95; margin: 0;">
            Adjust input feature metrics below to compute model predictions in real-time.
        </p>
    </div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 5. Sidebar Model Info Dashboard
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/home.png", width=64)
    st.title("Model Dashboard")
    
    if model is not None:
        st.success("🟢 **Model Loaded Successfully**")
        st.divider()
        
        # Read exact number of features expected by the trained Ridge model
        n_features = getattr(model, "n_features_in_", 79)
        alpha = getattr(model, "alpha", "N/A")
        solver = getattr(model, "solver", "Auto")
        
        st.markdown("### Model Details")
        st.write(f"• **Algorithm:** Ridge Regression")
        st.write(f"• **Features Input:** `{n_features}`")
        st.write(f"• **Alpha Regularization:** `{alpha}`")
        st.write(f"• **Solver Method:** `{solver}`")
        
        st.divider()
        
        # Reset Button
        if st.button("🔄 Reset Inputs to Default", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key.startswith("feature_"):
                    del st.session_state[key]
            st.rerun()
    else:
        st.error("🔴 **Model File Missing**")

# ---------------------------------------------------------
# 6. Tabbed Input System & Prediction
# ---------------------------------------------------------
if model is not None:
    n_features = getattr(model, "n_features_in_", 79)
    
    st.subheader("📊 Input Features")
    st.caption("Inputs are grouped into organized tabs to make navigating high-dimensional feature sets easy.")
    
    # Divide features across structured tabs (20 features per tab)
    features_per_tab = 20
    num_tabs = int(np.ceil(n_features / features_per_tab))
    tab_names = [f"Features {i*features_per_tab + 1}–{min((i+1)*features_per_tab, n_features)}" for i in range(num_tabs)]
    
    tabs = st.tabs(tab_names)
    user_inputs = [0.0] * n_features

    for t_idx, tab in enumerate(tabs):
        with tab:
            start_feat = t_idx * features_per_tab
            end_feat = min((t_idx + 1) * features_per_tab, n_features)
            
            col1, col2 = st.columns(2)
            for i in range(start_feat, end_feat):
                target_col = col1 if (i - start_feat) % 2 == 0 else col2
                with target_col:
                    val = st.number_input(
                        label=f"Feature {i+1}",
                        value=0.0,
                        step=0.01,
                        format="%.4f",
                        key=f"feature_{i}"
                    )
                    user_inputs[i] = val

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Calculate Button
    if st.button("🚀 Compute Prediction", type="primary", use_container_width=True):
        input_array = np.array(user_inputs).reshape(1, -1)
        
        try:
            prediction = model.predict(input_array)[0]
            
            # Display Prediction Metrics Card
            st.markdown(f"""
                <div class="metric-card">
                    <p style="color: #6b7280; font-size: 0.9rem; margin-bottom: 0.25rem; font-weight: 600;">ESTIMATED TARGET VALUE</p>
                    <h2 style="color: #111827; font-size: 2.25rem; margin: 0; font-weight: 700;">{prediction:,.4f}</h2>
                </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Execution Error: Could not compute prediction. Details: {e}")
