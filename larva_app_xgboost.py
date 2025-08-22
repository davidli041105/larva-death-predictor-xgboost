import streamlit as st
import xgboost as xgb
import math

def load_model():
    model = xgb.XGBRegressor()
    model.load_model("xgb_model.json")
    return model

model = load_model()

st.title("🐛 Larva Death Time Predictor (XGBoost based, Mealworms only)")

# Icon projections
feature_icons = {
    "Wt. (fresh)/g": "🐛",
    "Weight Loss/g": "⚖️",
    "Len/cm": "📏",
    "Initial M.C./%": "💧",
    "M.C. Loss/%": "💧",
    "Area-to-Volume Ratio": "🔶",
    "Death Temp (T1)/°C": "🌡️",
    "Death Temp (T2)/°C": "🌡️",
}

# Define features for user input
features_info = [
    ("Wt. (fresh)/g", 0.0001, "%.4f"),
    ("Weight Loss/g", 0.0001, "%.4f"),
    ("Len/cm", 0.01, "%.2f"),
    ("Initial M.C./%", 0.01, "%.2f"),
    ("M.C. Loss/%", 0.01, "%.2f"),
    ("Area-to-Volume Ratio", 0.001, "%.3f"),
    ("Death Temp (T1)/°C", 0.1, "%.1f"),
    ("Death Temp (T2)/°C", 0.1, "%.1f"),
]

inputs = []

st.subheader("Please enter your larva's features (if there's missing values, select NaN):")

for label, step, fmt in features_info:
    # Adjust ratios between elements
    cols = st.columns([0.15, 1, 0.3])
    with cols[0]:
        # 放大 emoji
        st.markdown(f"<div style='font-size:30px'>{feature_icons.get(label, '🔹')}</div>", unsafe_allow_html=True)
    with cols[1]:
        value = st.number_input(label, step=step, format=fmt)
    with cols[2]:
        use_nan = st.checkbox("NaN", key=f"nan_{label}")
        if use_nan:
            value = float("nan")
    inputs.append(value)

if st.button("Predict!"):
    if all(math.isnan(x) for x in inputs):
        # All feature missing → Error
        st.error("❌ All features are missing! Please enter at least one value before predicting.")
    else:
        try:
            pred = model.predict([inputs])
            st.success(f"Time to death: {pred[0]:.4f} seconds")

            # Some feature missing → pop up additional warning
            if any(math.isnan(x) for x in inputs):
                st.warning("⚠️ Some feature values are missing. The prediction might be less accurate.")

        except Exception as e:
            st.error(f"Prediction Failed! Please check your input again!: {e}")
