import os
import streamlit as st

st.set_page_config(page_title="CardioSense AI", page_icon="❤️", layout="wide")

if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "heart.csv")):
    st.error("heart.csv not found. Place it in the same folder as app.py and reload.")
    st.stop()

pages = st.navigation([
    st.Page("views/home.py", title="Home", icon="🏠", default=True),
    st.Page("views/dashboard.py", title="Dashboard", icon="📊"),
    st.Page("views/predict.py", title="Risk Predictor", icon="🩺"),
])
st.sidebar.markdown("### ❤️ CardioSense AI")
st.sidebar.caption("Smart heart disease analytics and prediction")
pages.run()
