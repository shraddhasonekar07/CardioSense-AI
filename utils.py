import os
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

BASE = os.path.dirname(os.path.abspath(__file__))
PINK, BLUE, GOLD = "#ff4d6d", "#4cc9f0", "#ffd166"
PALETTE = {"Heart Disease": PINK, "Healthy": BLUE}


def apply_css():
    st.markdown(
        """
        <style>
        .stApp { background: radial-gradient(circle at top left,#3a0f3f 0%,#1a0b2e 45%,#0d0618 100%); }
        h1,h2,h3,h4,p,li,label,span,div { color:#f5f0ff; }
        section[data-testid="stSidebar"] { background:#12071f; border-right:1px solid #2d1745; }
        .hero { padding:56px 40px; border-radius:26px; margin-bottom:26px; text-align:center;
                background:linear-gradient(120deg,#ff4d6d 0%,#c9184a 45%,#7b2cbf 100%);
                box-shadow:0 18px 50px rgba(255,77,109,.35); }
        .hero h1 { font-size:3.2rem; margin:0; letter-spacing:1px; }
        .hero p { font-size:1.2rem; opacity:.92; margin-top:10px; }
        .card { background:rgba(255,255,255,.07); border:1px solid rgba(255,255,255,.14);
                padding:22px; border-radius:18px; text-align:center; height:100%;
                transition:transform .2s; }
        .card:hover { transform:translateY(-5px); border-color:#ff4d6d; }
        .card .big { font-size:2.1rem; font-weight:800; color:#ff8fa3; margin:0; }
        .card .icon { font-size:2.2rem; }
        .card h3 { margin:6px 0; }
        .card p { opacity:.8; margin:0; font-size:.95rem; }
        .result { padding:22px; border-radius:16px; text-align:center; font-size:1.4rem; font-weight:700; }
        .bad { background:linear-gradient(90deg,#c9184a,#9d0208); }
        .good { background:linear-gradient(90deg,#2a9d8f,#1b7a6f); }
        .step { background:rgba(255,255,255,.05); border-left:4px solid #ff4d6d;
                padding:14px 18px; border-radius:10px; margin-bottom:10px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def card(icon, title, text):
    return f'<div class="card"><div class="icon">{icon}</div><h3>{title}</h3><p>{text}</p></div>'


def kpi(value, label):
    return f'<div class="card"><p class="big">{value}</p><p>{label}</p></div>'


def style_fig(fig, height=380):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=55, b=20, l=10, r=10),
                      height=height)
    return fig


def get_target(df):
    return "target" if "target" in df.columns else df.columns[-1]


@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(BASE, "heart.csv"))
    t = get_target(df)
    df["Status"] = df[t].map({1: "Heart Disease", 0: "Healthy"})
    if "sex" in df.columns:
        df["Sex"] = df["sex"].map({1: "Male", 0: "Female"})
    if "cp" in df.columns:
        df["Chest Pain"] = df["cp"].map({0: "Typical Angina", 1: "Atypical Angina",
                                         2: "Non-anginal", 3: "Asymptomatic"})
    if "age" in df.columns:
        df["Age Group"] = pd.cut(df["age"], [0, 40, 50, 60, 70, 120],
                                 labels=["<40", "40-49", "50-59", "60-69", "70+"]).astype(str)
    return df


EXTRA = ["Status", "Sex", "Chest Pain", "Age Group"]


@st.cache_resource
def train_model():
    df = load_data()
    t = get_target(df)
    X = df.drop(columns=[t] + [c for c in EXTRA if c in df.columns])
    y = df[t]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=300, random_state=42).fit(Xtr, ytr)
    return model, list(X.columns), accuracy_score(yte, model.predict(Xte))
