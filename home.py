import streamlit as st
from utils import apply_css, card, kpi, load_data, get_target, train_model

apply_css()
df = load_data()
t = get_target(df)
_, _, acc = train_model()

st.markdown(
    """
    <div class="hero">
        <h1>❤️ CardioSense AI</h1>
        <p>Explore heart health data, understand the risk factors, and predict heart disease risk in seconds.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
c1.markdown(kpi(f"{len(df):,}", "Patient records"), unsafe_allow_html=True)
c2.markdown(kpi(f"{df.shape[1] - 4}", "Health indicators"), unsafe_allow_html=True)
c3.markdown(kpi(f"{(df[t] == 1).mean() * 100:.0f}%", "Disease prevalence"), unsafe_allow_html=True)
c4.markdown(kpi(f"{acc * 100:.1f}%", "Model accuracy"), unsafe_allow_html=True)


