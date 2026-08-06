import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# 1. Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Real Estate Valuator & Forecast Engine",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# 2. Dynamic Dark/Light UI Styling
# ---------------------------------------------------------
st.markdown("""
    <style>
    /* Card Container Styling */
    .hero-card {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        padding: 2rem;
        border-radius: 16px;
        color: #ffffff !important;
        margin-bottom: 2rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2);
    }
    .hero-card h1, .hero-card p {
        color: #ffffff !important;
    }
    
    /* Result Display Card */
    .metric-display {
        background-color: rgba(37, 99, 235, 0.08);
        border-left: 6px solid #2563eb;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Input Label Formatting */
    .stNumberInput label {
        font-weight: 600 !important;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 3. Model Loading
# ---------------------------------------------------------
MODEL_PATH = "best_regression_model.pkl"

@st.cache_resource
def load_model():
    """Load model using joblib to match binary format."""
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except FileNotFoundError:
        st.error(f"⚠️ Could not find model file `{MODEL_PATH}` in your root directory.")
        return None
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        return None

model = load_model()

# ---------------------------------------------------------
# 4. Hero Banner
# ---------------------------------------------------------
st.markdown("""
    <div class="hero-card">
        <h1 style="margin: 0; font-size: 2.2rem;">🏠 Smart Real Estate Valuation & Future Forecast</h1>
        <p style="margin-top: 0.5rem; font-size: 1.05rem; opacity: 0.9;">
            Enter property metrics to estimate instant market value and project future appreciation potential.
        </p>
    </div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 5. Sidebar Dashboard
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/home.png", width=60)
    st.title("Model Dashboard")
    
    if model is not None:
        st.success("🟢 **Model Active**")
        st.divider()
        
        # Read exact number of features expected by the trained model
        n_features = getattr(model, "n_features_in_", 79)
        alpha = getattr(model, "alpha", 1.0)
        solver = getattr(model, "solver", "auto")
        
        st.markdown("### System Specs")
        st.markdown(f"• **Model:** Ridge Regression")
        st.markdown(f"• **Features Handled:** `{n_features}`")
        st.markdown(f"• **Alpha Regularization:** `{alpha}`")
        st.markdown(f"• **Solver:** `{solver}`")
        
        st.divider()
        
        if st.button("🔄 Reset Inputs", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key.startswith("feature_"):
                    del st.session_state[key]
            st.rerun()
    else:
        st.error("🔴 **Model Disconnected**")

# ---------------------------------------------------------
# 6. Main Content - Inputs & Future Forecast Features
# ---------------------------------------------------------
if model is not None:
    n_features = getattr(model, "n_features_in_", 79)
    
    # --- SECTION A: Categorical & Future Market Factors ---
    st.subheader("🌐 Location & Market Projections")
    st.caption("Select high-level categorical features used to calculate multi-year price growth trends.")
    
    cat_col1, cat_col2, cat_col3 = st.columns(3)
    
    with cat_col1:
        location_type = st.selectbox(
            "Neighborhood Zone Type",
            ["Urban / Metro Center", "Suburban Growth Corridor", "Rural / Outskirts"],
            help="Determines baseline annual growth rates."
        )
    with cat_col2:
        condition_grade = st.selectbox(
            "Property Condition Grade",
            ["Excellent (Recently Renovated)", "Good (Move-in Ready)", "Fair (Needs Work)"],
            index=1
        )
    with cat_col3:
        forecast_years = st.slider(
            "Future Projection Horizon (Years)",
            min_value=1,
            max_value=10,
            value=5
        )

    st.divider()

    # --- SECTION B: Numerical Model Feature Input ---
    st.subheader("📊 Primary Numerical Features")
    st.caption("Adjust numerical parameters required by the Ridge model.")

    # Group 79 features into organized expanders
    features_per_group = 20
    num_groups = int(np.ceil(n_features / features_per_group))
    user_inputs = [0.0] * n_features

    for g_idx in range(num_groups):
        start_feat = g_idx * features_per_group
        end_feat = min((g_idx + 1) * features_per_group, n_features)
        
        with st.expander(f"📁 Feature Group {g_idx + 1} (Features {start_feat + 1} to {end_feat})", expanded=(g_idx == 0)):
            col1, col2 = st.columns(2)
            
            for i in range(start_feat, end_feat):
                target_col = col1 if (i - start_feat) % 2 == 0 else col2
                with target_col:
                    val = st.number_input(
                        label=f"Metric Feature #{i+1}",
                        value=0.0,
                        step=0.1,
                        format="%.4f",
                        key=f"feature_{i}"
                    )
                    user_inputs[i] = val

    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- SECTION C: Prediction Execution & Future Growth ---
    if st.button("🚀 Calculate Valuation & Future Forecast", type="primary", use_container_width=True):
        input_array = np.array(user_inputs).reshape(1, -1)
        
        try:
            # 1. Base Valuation Prediction
            base_prediction = float(model.predict(input_array)[0])
            
            st.divider()
            st.subheader("🎯 Valuation & Future Forecast Results")
            
            m_col1, m_col2 = st.columns([1, 1])
            
            with m_col1:
                st.markdown(f"""
                    <div class="metric-display">
                        <span style="font-size: 0.95rem; font-weight: 600; opacity: 0.8;">CURRENT ESTIMATED VALUE</span>
                        <h1 style="margin: 0; font-size: 2.5rem; color: #2563eb;">${base_prediction:,.2f}</h1>
                    </div>
                """, unsafe_allow_html=True)
            
            # 2. Future Growth Logic Calculation
            # Determine growth percentage based on categorical selection
            if "Urban" in location_type:
                annual_growth = 0.055  # 5.5% yearly
            elif "Suburban" in location_type:
                annual_growth = 0.042  # 4.2% yearly
            else:
                annual_growth = 0.028  # 2.8% yearly
                
            if "Excellent" in condition_grade:
                annual_growth += 0.01  # +1% bonus growth
            elif "Fair" in condition_grade:
                annual_growth -= 0.008

            # Calculate future projected values
            future_data = []
            current_val = base_prediction
            
            for yr in range(1, forecast_years + 1):
                current_val *= (1 + annual_growth)
                future_data.append({"Year": f"Year {yr}", "Projected Value ($)": round(current_val, 2)})
            
            df_forecast = pd.DataFrame(future_data).set_index("Year")
            future_final_val = future_data[-1]["Projected Value ($)"]

            with m_col2:
                st.markdown(f"""
                    <div class="metric-display">
                        <span style="font-size: 0.95rem; font-weight: 600; opacity: 0.8;">FORECASTED VALUE ({forecast_years} YRS)</span>
                        <h1 style="margin: 0; font-size: 2.5rem; color: #16a34a;">${future_final_val:,.2f}</h1>
                    </div>
                """, unsafe_allow_html=True)

            # 3. Future Price Trend Visualization Chart
            st.markdown(f"### 📈 Projected Price Trajectory over Next {forecast_years} Years")
            st.line_chart(df_forecast["Projected Value ($)"])
            
            # Summary Table
            with st.expander("📄 View Detailed Year-by-Year Breakdown"):
                st.dataframe(df_forecast, use_container_width=True)
            
        except Exception as e:
            st.error(f"Prediction Error: {e}")
