import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from utils import (BLUE, PINK, GOLD, apply_css, get_target, load_data,
                   style_fig, train_model)

apply_css()
st.title("🩺 Risk Predictor")
st.caption("Enter the patient's details below and click Predict.")

df = load_data()
t = get_target(df)
model, cols, acc = train_model()

labels = {
    "age": "Age", "sex": "Sex (1 = Male, 0 = Female)", "cp": "Chest pain type (0-3)",
    "trestbps": "Resting blood pressure", "chol": "Cholesterol (mg/dl)",
    "fbs": "Fasting sugar > 120 (1 = Yes)", "restecg": "Resting ECG (0-2)",
    "thalach": "Max heart rate", "exang": "Exercise-induced angina (1 = Yes)",
    "oldpeak": "ST depression", "slope": "ST slope (0-2)",
    "ca": "Major vessels (0-4)", "thal": "Thalassemia (0-3)",
}

inputs = {}
grid = st.columns(3)
for i, c in enumerate(cols):
    col = df[c]
    if col.nunique() <= 5:
        opts = sorted(col.unique().tolist())
        inputs[c] = grid[i % 3].selectbox(labels.get(c, c), opts, index=opts.index(col.mode()[0]))
    else:
        inputs[c] = grid[i % 3].slider(labels.get(c, c), float(col.min()), float(col.max()), float(col.median()))

if st.button("🔍 Predict Risk", type="primary"):
    row = pd.DataFrame([inputs])[cols]
    prob = float(model.predict_proba(row)[0][1])
    score = round(100 - prob * 100)

    r1, r2 = st.columns(2)
    with r1:
        if prob >= 0.5:
            st.markdown('<div class="result bad">⚠️ High risk of heart disease</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result good">✅ Low risk of heart disease</div>', unsafe_allow_html=True)
        st.metric("Heart Health Score", f"{score} / 100")
        st.progress(score / 100)

        st.markdown("#### Things to watch")
        flags = []
        if inputs.get("chol", 0) > 240: flags.append("Cholesterol is above 240 mg/dl.")
        if inputs.get("trestbps", 0) > 140: flags.append("Resting blood pressure is above 140.")
        if inputs.get("exang") == 1: flags.append("Chest pain during exercise was reported.")
        if inputs.get("oldpeak", 0) > 2: flags.append("ST depression is high (above 2).")
        if inputs.get("fbs") == 1: flags.append("Fasting blood sugar is above 120.")
        if inputs.get("age", 0) > 60: flags.append("Age is above 60.")
        if flags:
            for f in flags:
                st.write("• " + f)
        else:
            st.write("No major risk indicators found in these inputs.")

    with r2:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=prob * 100, number={"suffix": "%"},
            title={"text": "Disease probability"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": PINK},
                   "steps": [{"range": [0, 40], "color": "#2a9d8f"},
                             {"range": [40, 70], "color": "#e9c46a"},
                             {"range": [70, 100], "color": "#9d0208"}]}))
        st.plotly_chart(style_fig(gauge, 300))

    # Radar: patient vs averages
    feats = [f for f in ["age", "trestbps", "chol", "thalach", "oldpeak"] if f in cols]
    names = {"age": "Age", "trestbps": "Blood Pressure", "chol": "Cholesterol",
             "thalach": "Max Heart Rate", "oldpeak": "ST Depression"}

    def norm(f, v):
        lo, hi = df[f].min(), df[f].max()
        return (v - lo) / (hi - lo) if hi > lo else 0

    radar = go.Figure()
    series = [("This patient", [norm(f, inputs[f]) for f in feats], GOLD),
              ("Healthy average", [norm(f, df[df[t] == 0][f].mean()) for f in feats], BLUE),
              ("Disease average", [norm(f, df[df[t] == 1][f].mean()) for f in feats], PINK)]
    for name, vals, color in series:
        radar.add_trace(go.Scatterpolar(r=vals + vals[:1], theta=[names[f] for f in feats] + [names[feats[0]]],
                                        name=name, line=dict(color=color), fill="toself", opacity=.55))
    radar.update_layout(title="Patient vs Average Profiles", polar=dict(radialaxis=dict(visible=False)))
    st.plotly_chart(style_fig(radar, 450))


