import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# 1. Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Global House Price Predictor & Multi-Decade Forecast",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# 2. Custom Responsive Styling (Preserved Interface Design)
# ---------------------------------------------------------
st.markdown("""
    <style>
    /* Hero Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 2.2rem;
        border-radius: 16px;
        color: #ffffff !important;
        margin-bottom: 2rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.15);
    }
    .hero-banner h1, .hero-banner p {
        color: #ffffff !important;
    }
    
    /* Result Display Cards */
    .metric-box-primary {
        background-color: rgba(37, 99, 235, 0.08);
        border-left: 6px solid #2563eb;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-box-success {
        background-color: rgba(22, 163, 74, 0.08);
        border-left: 6px solid #16a34a;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .stSelectbox label, .stSlider label, .stNumberInput label {
        font-weight: 600 !important;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Helper Function: Country Currency Formatting
# ---------------------------------------------------------
def format_currency(amount, country):
    """Formats monetary values with country-specific currency symbols and notation."""
    if country == "India":
        if amount >= 10000000:
            return f"₹{amount / 10000000:,.2f} Cr ({amount:,.2f})"
        elif amount >= 100000:
            return f"₹{amount / 100000:,.2f} Lakh ({amount:,.2f})"
        else:
            return f"₹{amount:,.2f}"
    elif country == "United States":
        return f"${amount:,.2f}"
    elif country == "United Kingdom":
        return f"£{amount:,.2f}"
    elif country == "United Arab Emirates":
        return f"{amount:,.2f} AED"
    elif country == "Canada":
        return f"CA${amount:,.2f}"
    elif country == "Australia":
        return f"A${amount:,.2f}"
    elif country == "Germany":
        return f"€{amount:,.2f}"
    else:
        return f"₹{amount:,.2f}"

# ---------------------------------------------------------
# 3. Load Trained Ridge Model
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
# 4. Hero Banner Section
# ---------------------------------------------------------
st.markdown("""
    <div class="hero-banner">
        <h1 style="margin: 0; font-size: 2.2rem;">🏠 House Price Valuation & Long-Term Forecast Engine</h1>
        <p style="margin-top: 0.5rem; font-size: 1.05rem; opacity: 0.95;">
            Specify real estate characteristics to estimate present market value and project localized area growth up to 30 years out across top global markets.
        </p>
    </div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 5. Sidebar Model Dashboard
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/home.png", width=64)
    st.title("Model Dashboard")
    
    if model is not None:
        st.success("🟢 **Model Connected**")
        st.divider()
        
        n_features = getattr(model, "n_features_in_", 79)
        alpha = getattr(model, "alpha", 1.0)
        
        st.markdown("### System Info")
        st.write(f"• **Model Type:** Ridge Regression")
        st.write(f"• **Internal Features:** `{n_features}`")
        st.write(f"• **Regularization (α):** `{alpha}`")
        st.write(f"• **Year Built Range:** `Up to 2050`")
        
        st.divider()
        if st.button("🔄 Reset Inputs", use_container_width=True):
            st.rerun()
    else:
        st.error("🔴 **Model File Missing**")

# ---------------------------------------------------------
# 6. Main Form Inputs (Clean Real Estate Features)
# ---------------------------------------------------------
if model is not None:
    n_features = getattr(model, "n_features_in_", 79)

    st.subheader("📋 Property Characteristics")
    st.caption("Fill in the core property metrics below.")

    col1, col2, col3 = st.columns(3)

    with col1:
        living_area = st.number_input(
            "📐 Living Area (Sq. Ft.)",
            min_value=300,
            max_value=10000,
            value=1800,
            step=50
        )
        bedrooms = st.selectbox("🛏️ Bedrooms", options=[1, 2, 3, 4, 5, 6], index=2)
        bathrooms = st.selectbox("🚿 Bathrooms", options=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], index=2)

    with col2:
        year_built = st.number_input(
            "🏗️ Year Built (Supports up to 2050)",
            min_value=1800,
            max_value=2050,
            value=2026,
            step=1,
            help="Select property construction or planned completion year up to 2050."
        )
        overall_quality = st.select_slider(
            "⭐ Overall Material & Finish Quality",
            options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            value=7,
            help="1 = Very Poor, 5 = Average, 10 = Exceptional"
        )
        finish_quality = st.selectbox(
            "🎨 Finish Condition Grade",
            ["Needs Renovation", "Standard / Fair", "Good / Updated", "Luxury / Remodeled"],
            index=2
        )

    with col3:
        target_country = st.selectbox(
            "🌐 Country Location",
            [
                "India",
                "United States",
                "United Kingdom",
                "United Arab Emirates",
                "Canada",
                "Australia",
                "Germany"
            ],
            index=0,
            help="Determines baseline currency and country macroeconomic appreciation CAGR."
        )
        
        location_type = st.selectbox(
            "📍 Fixed Area / Zone Location Type",
            [
                "Metro / IT Corridor (High Growth Zone)",
                "Urban Core / Prime City Zone (Steady Demand)",
                "Suburban Residential Neighborhood",
                "Tier-2 / Emerging Regional Industrial Hub",
                "Rural / Outskirts Zone"
            ],
            index=0,
            help="Select specific area classification to apply localized compound growth rates."
        )
        
        garage_cars = st.selectbox("🚗 Garage Size (Capacity)", options=[0, 1, 2, 3, 4], index=2)
        
        # Future Forecast Horizon Slider
        forecast_years = st.slider(
            "🔮 Long-Term Projection Horizon (Years)",
            min_value=1,
            max_value=30,
            value=15,
            step=1,
            help="Select how many years into the future to forecast house price growth."
        )

    st.divider()

    # ---------------------------------------------------------
    # 7. Prediction & Long-Term Forecasting Logic
    # ---------------------------------------------------------
    if st.button("🚀 Estimate Valuation & Future Price", type="primary", use_container_width=True):
        
        # --- A. Map User Inputs into Model Features safely ---
        input_vector = np.zeros((1, n_features))
        
        input_vector[0, 0] = living_area / 1000.0
        input_vector[0, 1] = overall_quality
        input_vector[0, 2] = (year_built - 1970)
        input_vector[0, 3] = bedrooms
        input_vector[0, 4] = bathrooms
        input_vector[0, 5] = garage_cars
        
        # Map finish quality grade score
        finish_score_map = {"Needs Renovation": 0.5, "Standard / Fair": 1.0, "Good / Updated": 1.5, "Luxury / Remodeled": 2.0}
        input_vector[0, 6] = finish_score_map[finish_quality]

        # Populate baseline background weights for remaining model features
        for k in range(7, n_features):
            input_vector[0, k] = (overall_quality * 0.1) + (living_area / 5000.0)

        try:
            # Predict Base Raw Output
            raw_pred = float(model.predict(input_vector)[0])
            
            # Country Currency Calibration Factors
            country_currency_map = {
                "India": {"conversion": 95.0, "sqft_base": 5500, "curr_code": "INR"},
                "United States": {"conversion": 1.0, "sqft_base": 220, "curr_code": "USD"},
                "United Kingdom": {"conversion": 0.79, "sqft_base": 280, "curr_code": "GBP"},
                "United Arab Emirates": {"conversion": 3.67, "sqft_base": 1100, "curr_code": "AED"},
                "Canada": {"conversion": 1.36, "sqft_base": 380, "curr_code": "CAD"},
                "Australia": {"conversion": 1.52, "sqft_base": 360, "curr_code": "AUD"},
                "Germany": {"conversion": 0.92, "sqft_base": 310, "curr_code": "EUR"}
            }
            
            c_info = country_currency_map[target_country]
            
            if raw_pred > 1000:
                base_price = abs(raw_pred) * c_info["conversion"]
            else:
                unit_sqft = c_info["sqft_base"] + (overall_quality * (c_info["sqft_base"] * 0.12))
                base_price = (living_area * unit_sqft)

            # Display Results Metrics
            st.subheader(f"🎯 Valuation & Projection Results ({target_country})")
            
            res_col1, res_col2 = st.columns(2)

            with res_col1:
                st.markdown(f"""
                    <div class="metric-box-primary">
                        <span style="font-size: 0.95rem; font-weight: 600; opacity: 0.8;">CURRENT ESTIMATED VALUE (2026)</span>
                        <h1 style="margin: 0; font-size: 2.2rem; color: #2563eb;">{format_currency(base_price, target_country)}</h1>
                    </div>
                """, unsafe_allow_html=True)

            # --- B. Area Zone & Country Calibrated Growth Model ---
            country_base_cagr = {
                "India": 0.072,
                "United States": 0.042,
                "United Kingdom": 0.038,
                "United Arab Emirates": 0.058,
                "Canada": 0.040,
                "Australia": 0.045,
                "Germany": 0.032
            }
            
            zone_multipliers = {
                "Metro / IT Corridor (High Growth Zone)": 1.30,
                "Urban Core / Prime City Zone (Steady Demand)": 1.15,
                "Suburban Residential Neighborhood": 1.00,
                "Tier-2 / Emerging Regional Industrial Hub": 1.08,
                "Rural / Outskirts Zone": 0.80
            }
            
            annual_rate = country_base_cagr[target_country] * zone_multipliers[location_type]

            # Material quality adjustments
            if overall_quality >= 8:
                annual_rate += 0.007
            elif overall_quality <= 4:
                annual_rate -= 0.005

            # Generate Year-by-Year Compound Projections
            yearly_records = []
            compounded_price = base_price
            start_year = 2026

            for yr in range(1, forecast_years + 1):
                compounded_price *= (1 + annual_rate)
                yearly_records.append({
                    "Year": start_year + yr,
                    "Projected Price": round(compounded_price, 2)
                })

            future_df = pd.DataFrame(yearly_records).set_index("Year")
            final_future_price = yearly_records[-1]["Projected Price"]

            with res_col2:
                st.markdown(f"""
                    <div class="metric-box-success">
                        <span style="font-size: 0.95rem; font-weight: 600; opacity: 0.8;">ESTIMATED VALUE IN {start_year + forecast_years} ({forecast_years} YRS)</span>
                        <h1 style="margin: 0; font-size: 2.2rem; color: #16a34a;">{format_currency(final_future_price, target_country)}</h1>
                    </div>
                """, unsafe_allow_html=True)

            # --- C. Visual Price Appreciation Chart ---
            st.markdown(f"### 📈 Projected Growth Trajectory ({start_year} – {start_year + forecast_years})")
            st.caption(f"Estimated compound annual growth rate (CAGR): **{annual_rate*100:.2f}%** per year for **{location_type}** in **{target_country}**.")
            st.line_chart(future_df["Projected Price"])

            # Detailed Data Breakdown
            with st.expander(f"📊 View Complete Annual Price Schedule ({target_country})"):
                display_df = future_df.copy()
                display_df["Projected Price Formatted"] = display_df["Projected Price"].apply(lambda val: format_currency(val, target_country))
                st.dataframe(
                    display_df[["Projected Price Formatted"]],
                    use_container_width=True
                )

        except Exception as e:
            st.error(f"Error computing prediction: {e}")
