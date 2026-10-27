import pandas as pd
import plotly.express as px
import streamlit as st
from utils import (PALETTE, PINK, apply_css, get_target, kpi, load_data,
                   style_fig, train_model, EXTRA)

apply_css()
st.title("📊 Dashboard")

df = load_data()
t = get_target(df)

# ---------- Filters ----------
st.sidebar.header("🔎 Filters")
if "age" in df.columns:
    lo, hi = int(df.age.min()), int(df.age.max())
    a = st.sidebar.slider("Age range", lo, hi, (lo, hi))
    df = df[df.age.between(*a)]
if "Sex" in df.columns:
    sx = st.sidebar.multiselect("Sex", ["Male", "Female"], default=["Male", "Female"])
    df = df[df.Sex.isin(sx)]
if df.empty:
    st.warning("No patients match these filters.")
    st.stop()

# ---------- KPIs ----------
k = st.columns(5)
k[0].markdown(kpi(len(df), "Patients"), unsafe_allow_html=True)
k[1].markdown(kpi(f"{(df[t] == 1).mean() * 100:.1f}%", "Disease rate"), unsafe_allow_html=True)
k[2].markdown(kpi(f"{df.age.mean():.0f}", "Average age"), unsafe_allow_html=True)
k[3].markdown(kpi(f"{df.chol.mean():.0f}", "Avg cholesterol"), unsafe_allow_html=True)
k[4].markdown(kpi(f"{df.thalach.mean():.0f}", "Avg max heart rate"), unsafe_allow_html=True)
st.write("")

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Deep Dive", "Model Insights", "Data"])

with tab1:
    c1, c2 = st.columns(2)
    c1.plotly_chart(style_fig(px.pie(df, names="Status", hole=.55, color="Status",
                                     color_discrete_map=PALETTE, title="Patient Status")))
    c2.plotly_chart(style_fig(px.histogram(df, x="age", color="Status", barmode="overlay", nbins=25,
                                           color_discrete_map=PALETTE, title="Age Distribution")))
    c3, c4 = st.columns(2)
    if "Chest Pain" in df.columns:
        c3.plotly_chart(style_fig(px.histogram(df, x="Chest Pain", color="Status", barmode="group",
                                               color_discrete_map=PALETTE, title="Chest Pain Type")))
    if "Sex" in df.columns:
        rate = df.groupby("Sex")[t].mean().mul(100).reset_index()
        c4.plotly_chart(style_fig(px.bar(rate, x="Sex", y=t, text_auto=".1f", color_discrete_sequence=[PINK],
                                         title="Disease Rate by Sex (%)", labels={t: "Disease rate %"})))

with tab2:
    order = ["<40", "40-49", "50-59", "60-69", "70+"]
    g = df.groupby("Age Group")[t].mean().mul(100).reindex(order).dropna().reset_index()
    st.plotly_chart(style_fig(px.line(g, x="Age Group", y=t, markers=True, title="Disease Rate by Age Group (%)",
                                      labels={t: "Disease rate %"}, color_discrete_sequence=[PINK])))
    d1, d2 = st.columns(2)
    d1.plotly_chart(style_fig(px.scatter(df, x="age", y="thalach", color="Status", opacity=.8,
                                         color_discrete_map=PALETTE,
                                         title="Age vs Max Heart Rate")))
    d2.plotly_chart(style_fig(px.box(df, x="Status", y="chol", color="Status", color_discrete_map=PALETTE,
                                     title="Cholesterol by Status")))
    nums = df.drop(columns=[c for c in EXTRA if c in df.columns]).corr()
    st.plotly_chart(style_fig(px.imshow(nums, color_continuous_scale="RdPu", aspect="auto",
                                        title="Correlation Heatmap"), height=520))

with tab3:
    model, cols, acc = train_model()
    st.metric("Model accuracy (test set)", f"{acc * 100:.1f}%")
    imp = pd.DataFrame({"Feature": cols, "Importance": model.feature_importances_}).sort_values("Importance")
    st.plotly_chart(style_fig(px.bar(imp, x="Importance", y="Feature", orientation="h",
                                     color="Importance", color_continuous_scale="RdPu",
                                     title="What drives the prediction?"), height=480))

with tab4:
    st.dataframe(df.drop(columns=[c for c in EXTRA if c in df.columns]))
    st.download_button("Download filtered data (CSV)",
                       df.drop(columns=[c for c in EXTRA if c in df.columns]).to_csv(index=False),
                       "filtered_heart_data.csv")
