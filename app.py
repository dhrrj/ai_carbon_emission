"""Interactive Streamlit dashboard for urban carbon-emission risk."""

import streamlit as st

from carbon_urban import FEATURE_COLUMNS, predict_risk, train_model

st.set_page_config(
    page_title="Urban Carbon Intelligence",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
    .stApp { background: #f5f7f4; color: #17241e; }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1440px; }
    h1, h2, h3 { font-family: 'Manrope', sans-serif !important; letter-spacing: -0.035em; }
    .eyebrow { color: #528364; text-transform: uppercase; letter-spacing: .14em;
      font-size: .72rem; font-weight: 700; }
    .hero-title { font-family: 'Manrope', sans-serif; font-size: 2.45rem;
      font-weight: 800; letter-spacing: -.055em; line-height: 1.15; margin: .3rem 0; }
    .hero-copy { color: #68766e; margin: .3rem 0 1.2rem; }
    .metric-card { background: white; border: 1px solid #e6ebe6; border-radius: 16px;
      padding: 18px 20px; min-height: 102px; }
    .metric-label { color: #77847b; font-size: .78rem; font-weight: 600; }
    .metric-value { color: #17241e; font-family: 'Manrope', sans-serif;
      font-size: 1.65rem; font-weight: 800; margin-top: 7px; }
    .result-card { background: white; border: 1px solid #e6ebe6; border-radius: 18px;
      padding: 24px; }
    .risk-pill { display: inline-block; border-radius: 999px; padding: 7px 13px;
      font-weight: 700; font-size: .82rem; margin: 10px 0 2px; }
    .stButton > button { border-radius: 10px; min-height: 46px; font-weight: 700;
      border: 0; background: #20764e; color: white; }
    .stButton > button:hover { background: #185e3e; color: white; }
    [data-testid="stSidebar"] { background: #fff; border-right: 1px solid #e9ede9; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Training the city emissions model...")
def load_model():
    return train_model()


trained = load_model()

with st.sidebar:
    st.markdown("### 🌿 Urban Carbon")
    st.caption("SMART CITY · EMISSIONS MONITOR")
    st.markdown("---")
    st.markdown("#### Live sensor reading")
    district = st.selectbox("District", [1, 2, 3, 4, 5], index=2)
    temperature = st.slider("Temperature (°C)", 15.0, 42.0, 32.0, 0.1)
    humidity = st.slider("Humidity (%)", 30.0, 90.0, 58.0, 1.0)
    traffic = st.slider("Traffic congestion", 0.0, 100.0, 62.0, 1.0)
    energy = st.slider("Grid energy demand (MW)", 200.0, 800.0, 520.0, 1.0)
    transit = st.slider("Transit delay (minutes)", 0.0, 45.0, 12.0, 1.0)
    industry = st.slider("Industrial activity", 10.0, 100.0, 48.0, 1.0)
    st.markdown("---")
    st.caption("Predictions update automatically using the XGBoost classifier.")

st.markdown('<div class="eyebrow">City operations · Carbon intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Urban emissions dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-copy">Monitor environmental signals and assess carbon-emission risk across city districts.</div>',
    unsafe_allow_html=True,
)

summary_a, summary_b, summary_c, summary_d = st.columns(4)
for column, label, value in [
    (summary_a, "MODEL ACCURACY", f"{trained['accuracy']:.1%}"),
    (summary_b, "TRAINING READINGS", f"{trained['sample_count']:,}"),
    (summary_c, "DISTRICTS MONITORED", "5"),
    (summary_d, "RISK CATEGORIES", str(len(trained["class_names"]))),
]:
    with column:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div></div>',
            unsafe_allow_html=True,
        )

st.markdown("")
left, right = st.columns([1.15, 0.85], gap="large")

sensor_reading = {
    "district_id": district,
    "temperature_c": temperature,
    "humidity_pct": humidity,
    "traffic_congestion_index": traffic,
    "grid_energy_demand_mw": energy,
    "public_transit_delay_mins": transit,
    "industrial_zone_activity": industry,
}
prediction = predict_risk(trained, sensor_reading)

with left:
    st.markdown("### District risk assessment")
    st.caption(
        f"Current sensor profile · District {district}"
        + " · live prediction"
    )
    risk_styles = {
        "High Risk": ("#fff0ed", "#ba4534", "Immediate attention recommended"),
        "Medium Risk": ("#fff5df", "#9a6b0b", "Monitor conditions and plan mitigation"),
        "Low Risk": ("#e8f5ec", "#267447", "Conditions are within a lower-risk range"),
    }
    background, foreground, description = risk_styles[prediction["risk"]]
    st.markdown(
        f'<div class="result-card"><div class="metric-label">PREDICTED EMISSION RISK</div>'
        f'<span class="risk-pill" style="background:{background};color:{foreground}">'
        f'{prediction["risk"]}</span><div style="color:#68766e;margin-top:8px">'
        f'{description}</div><div style="margin-top:20px;font-weight:700">'
        f'Model confidence&nbsp; {prediction["confidence"]:.1%}</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("#### Probability by risk category")
    for risk in trained["class_names"]:
        st.progress(
            float(prediction["probabilities"][risk]),
            text=f"{risk} · {prediction['probabilities'][risk]:.1%}",
        )

with right:
    st.markdown("### Sensor overview")
    st.caption("Values supplied to the risk model")
    sensor_labels = {
        "district_id": "District",
        "temperature_c": "Temperature",
        "humidity_pct": "Humidity",
        "traffic_congestion_index": "Traffic congestion",
        "grid_energy_demand_mw": "Grid energy demand",
        "public_transit_delay_mins": "Transit delay",
        "industrial_zone_activity": "Industrial activity",
    }
    display_values = {
        "district_id": f"District {district}",
        "temperature_c": f"{temperature:.1f} °C",
        "humidity_pct": f"{humidity:.0f}%",
        "traffic_congestion_index": f"{traffic:.0f} / 100",
        "grid_energy_demand_mw": f"{energy:.0f} MW",
        "public_transit_delay_mins": f"{transit:.0f} min",
        "industrial_zone_activity": f"{industry:.0f} / 100",
    }
    for feature in FEATURE_COLUMNS:
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;padding:13px 2px;'
            f'border-bottom:1px solid #e6ebe6"><span style="color:#68766e">'
            f'{sensor_labels[feature]}</span><strong>{display_values[feature]}</strong></div>',
            unsafe_allow_html=True,
        )

st.markdown("")
with st.expander("Model evaluation details"):
    st.markdown(f"**Held-out test accuracy:** {trained['accuracy']:.2%}")
    report_rows = []
    for risk in trained["class_names"]:
        scores = trained["classification_report"][risk]
        report_rows.append(
            {
                "Risk category": risk,
                "Precision": scores["precision"],
                "Recall": scores["recall"],
                "F1 score": scores["f1-score"],
                "Samples": int(scores["support"]),
            }
        )
    st.dataframe(report_rows, hide_index=True, use_container_width=True)
