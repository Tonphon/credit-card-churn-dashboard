"""
Credit Card Churn Intelligence Dashboard
Relationship Manager Portal
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import yfinance as yf

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Card Churn Intelligence",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
SEGMENT_NAMES = {
    0: "Revolvers",
    1: "Stable Mid-Tier",
    2: "Disengaged At-Risk",
    3: "Dormant High-Value",
    4: "High-Value Actives",
}
SEGMENT_COLORS = {
    0: "#4C72B0",
    1: "#DD8452",
    2: "#55A868",
    3: "#8172B2",
    4: "#C44E52",
}
SEGMENT_CHURN_KNOWN = {0: 5.1, 1: 7.8, 2: 37.9, 3: 18.1, 4: 2.1}
BOT_BASELINE = 1.50  # %

FEATURE_COLS = [
    "Customer_Age", "Dependent_count", "Months_on_book", "Total_Relationship_Count",
    "Months_Inactive_12_mon", "Contacts_Count_12_mon", "Credit_Limit", "Avg_Open_To_Buy",
    "Total_Trans_Ct", "Avg_Utilization_Ratio", "Spend_Per_Transaction", "Engagement_Score",
    "Credit_Headroom", "Segment", "Gender_enc", "Education_Level_enc", "Marital_Status_enc",
    "Income_Category_enc", "Card_Category_enc",
]

SEG_FEATURES = [
    "Total_Trans_Ct", "Total_Trans_Amt", "Avg_Utilization_Ratio", "Engagement_Score",
    "Months_Inactive_12_mon", "Contacts_Count_12_mon", "Total_Relationship_Count", "Credit_Limit",
]

GENDER_MAP    = {0: "Female", 1: "Male"}
EDUCATION_MAP = {
    0: "College", 1: "Doctorate", 2: "Graduate", 3: "High School",
    4: "Post-Graduate", 5: "Uneducated", 6: "Unknown",
}
MARITAL_MAP   = {0: "Divorced", 1: "Married", 2: "Single", 3: "Unknown"}
INCOME_MAP    = {
    0: "$120K +", 1: "$40K - $60K", 2: "$60K - $80K",
    3: "$80K - $120K", 4: "Less than $40K", 5: "Unknown",
}
CARD_MAP      = {0: "Blue", 1: "Gold", 2: "Platinum", 3: "Silver"}

RM_ACTIONS = {
    0: (
        "<b>⚠️ Monitor &amp; Relieve Utilization Pressure</b><br><br>"
        "This customer is revolving a high balance relative to their credit limit — "
        "the strongest predictor of future stress. Recommended actions:<br>"
        "<ul style='margin:0.4rem 0 0 1rem;line-height:1.8;'>"
        "<li>Offer a <b>credit limit review</b> to ease utilization pressure.</li>"
        "<li>Propose a <b>balance consolidation</b> or instalment plan at a preferential rate.</li>"
        "<li>Schedule a proactive outreach call to understand financial goals and reinforce loyalty.</li>"
        "</ul>"
    ),
    1: (
        "<b>📈 Deepen the Relationship</b><br><br>"
        "A reliable, moderate-engagement customer with room to grow. Focus on value-add:<br>"
        "<ul style='margin:0.4rem 0 0 1rem;line-height:1.8;'>"
        "<li>Introduce a <b>rewards upgrade</b> or co-branded benefit.</li>"
        "<li>Cross-sell a complementary product (savings, insurance, or a second card tier).</li>"
        "<li>Use the next billing cycle as a natural touchpoint.</li>"
        "</ul>"
    ),
    2: (
        "<b>🚨 Immediate Retention Action Required</b><br><br>"
        "Elevated churn risk — low transaction activity and elevated support contacts. "
        "Act within <b>48 hours</b>:<br>"
        "<ul style='margin:0.4rem 0 0 1rem;line-height:1.8;'>"
        "<li><b>Call personally</b> to understand pain points before pitching anything.</li>"
        "<li>Lead with a tangible retention offer: annual fee waiver, bonus rewards, or rate reduction.</li>"
        "<li>Escalate to a Senior RM if initial outreach is unsuccessful after two attempts.</li>"
        "</ul>"
    ),
    3: (
        "<b>💎 Re-Activate a Premium Relationship</b><br><br>"
        "High credit limit but low utilisation signals this customer has drifted to other providers:<br>"
        "<ul style='margin:0.4rem 0 0 1rem;line-height:1.8;'>"
        "<li>Send a <b>premium experience reminder</b> — concierge, travel lounge, or cashback.</li>"
        "<li>Offer an exclusive spend incentive (bonus miles or statement credit) for next quarter.</li>"
        "<li>Assign a dedicated RM contact and communicate this upgrade in your outreach.</li>"
        "</ul>"
    ),
    4: (
        "<b>🏆 Protect &amp; Reward Your Best Customer</b><br><br>"
        "Top-tier engagement — focus on retention through delight, not reactive intervention:<br>"
        "<ul style='margin:0.4rem 0 0 1rem;line-height:1.8;'>"
        "<li>Proactively extend <b>VIP status acknowledgement</b> or tier upgrade.</li>"
        "<li>Offer early access to new product features, exclusive events, or enhanced credit terms.</li>"
        "<li>Monitor for any behaviour shifts (spend dip &gt;20% MoM) as an early-warning trigger.</li>"
        "</ul>"
    ),
}

# ── File paths (relative to app/ directory) ───────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "..", "models")
DATA_DIR  = os.path.join(BASE, "..", "data")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1923 0%, #1a2636 100%);
    }
    section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

    .page-header {
        background: linear-gradient(135deg, #0f1923 0%, #1e3a5f 100%);
        padding: 1.4rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.4rem;
        border-left: 4px solid #4C72B0;
    }
    .page-header h1 { color: #f1f5f9; font-size: 1.5rem; font-weight: 600; margin: 0; }
    .page-header p  { color: #94a3b8; font-size: 0.88rem; margin: 0.25rem 0 0 0; }

    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.07);
        height: 100%;
    }
    .metric-card .label { font-size: 0.72rem; color: #64748b; text-transform: uppercase;
                          letter-spacing: 0.06em; font-weight: 600; }
    .metric-card .value { font-size: 1.6rem; font-weight: 700; color: #0f172a; line-height: 1.2;
                          margin-top: 0.2rem; }
    .metric-card .delta { font-size: 0.76rem; margin-top: 0.25rem; }
    .delta-up  { color: #ef4444; }
    .delta-neu { color: #64748b; }

    .bot-badge {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 0.75rem 0.9rem;
        margin-bottom: 0.6rem;
    }
    .bot-label { font-size: 0.68rem; color: #94a3b8; text-transform: uppercase;
                 letter-spacing: 0.07em; }
    .bot-value { font-size: 1.35rem; font-weight: 700; }
    .bot-live  { color: #4ade80; }
    .bot-warn  { color: #fb923c; }

    .seg-pill {
        display: inline-block;
        padding: 0.3rem 0.85rem;
        border-radius: 20px;
        font-size: 0.84rem;
        font-weight: 600;
        color: #fff;
    }

    .action-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #4C72B0;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        font-size: 0.86rem;
        line-height: 1.75;
        color: #1e293b;
    }

    .macro-box {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 0.8rem 0.9rem;
        font-size: 0.77rem;
        color: #cbd5e1;
        line-height: 1.6;
    }


    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }
    header     { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Resource loaders ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def load_models():
    model  = joblib.load(os.path.join(MODEL_DIR, "churn_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    kmeans = joblib.load(os.path.join(MODEL_DIR, "kmeans.pkl"))
    return model, scaler, kmeans


@st.cache_data(show_spinner="Loading portfolio data…")
def load_data():
    df        = pd.read_csv(os.path.join(DATA_DIR, "test_predictions.csv"))
    shap_df   = pd.read_csv(os.path.join(DATA_DIR, "shap_sample.csv"))
    shap_vals = np.load(os.path.join(DATA_DIR, "shap_values_sample.npy"))
    df["Segment_Name"] = df["Segment"].map(SEGMENT_NAMES)
    return df, shap_df, shap_vals


@st.cache_data(ttl=3600, show_spinner=False)
def get_bot_data():
    """Fetch USD/THB live rate as a macro health indicator."""
    try:
        hist = yf.Ticker("THB=X").history(period="5d")
        if not hist.empty:
            usdthb = float(hist["Close"].iloc[-1])
            prev   = float(hist["Close"].iloc[-2]) if len(hist) > 1 else usdthb
            chg    = (usdthb - prev) / prev * 100
            return usdthb, chg, True
    except Exception:
        pass
    return 35.20, 0.0, False


# ── Chart helpers ─────────────────────────────────────────────────────────────
def apply_layout(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=45, b=20),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fafc",
        font=dict(family="Inter", size=12, color="#374151"),
        title=dict(font=dict(size=13, color="#1e293b")),
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)",
            borderwidth=1,
            bordercolor="#e2e8f0",
            font=dict(color="#374151"),
        ),
        xaxis=dict(
            gridcolor="#e8ecf1",
            showline=False,
            zeroline=False,
            tickfont=dict(color="#64748b", size=11),
            title_font=dict(color="#475569"),
        ),
        yaxis=dict(
            gridcolor="#e8ecf1",
            showline=False,
            zeroline=False,
            tickfont=dict(color="#64748b", size=11),
            title_font=dict(color="#475569"),
        ),
    )
    return fig


def gauge_chart(prob: float) -> go.Figure:
    pct   = prob * 100
    color = "#22c55e" if pct < 20 else "#f59e0b" if pct < 50 else "#ef4444"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 42, "color": "#0f172a"}},
        title={"text": "Churn Probability", "font": {"size": 13, "color": "#64748b"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8",
                     "tickfont": {"size": 10}},
            "bar":  {"color": color, "thickness": 0.26},
            "bgcolor": "#f8fafc",
            "borderwidth": 1,
            "bordercolor": "#e2e8f0",
            "steps": [
                {"range": [0, 20],   "color": "#dcfce7"},
                {"range": [20, 50],  "color": "#fef9c3"},
                {"range": [50, 100], "color": "#fee2e2"},
            ],
            "threshold": {
                "line": {"color": "#1e293b", "width": 2},
                "thickness": 0.8,
                "value": 15,
            },
        },
    ))
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=35, b=5),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(family="Inter", color="#1e293b"),
    )
    return fig


# ── Initialise ────────────────────────────────────────────────────────────────
model, scaler, kmeans = load_models()
df, shap_df, shap_vals = load_data()
usdthb, thb_chg, live_ok = get_bot_data()

BOT_RATE       = 2.50   # Current BOT policy rate (%)
above_baseline = BOT_RATE > BOT_BASELINE
MACRO_ADJ      = 0.03   # +3 pp applied to Revolver churn probs when rate elevated

if above_baseline:
    df = df.copy()
    df.loc[df["Segment"] == 0, "Churn_Probability"] = (
        df.loc[df["Segment"] == 0, "Churn_Probability"] + MACRO_ADJ
    ).clip(upper=1.0)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💳 Churn Intelligence")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["Portfolio Overview", "Customer Explorer", "Segment Deep Dive"],
        label_visibility="collapsed",
    )
    st.markdown("---")

    sign      = "+" if thb_chg >= 0 else ""
    thb_cls   = "bot-warn" if abs(thb_chg) > 0.5 else "bot-live"
    rate_cls  = "bot-warn" if above_baseline else "bot-live"

    st.markdown(f"""
    <div class="bot-badge">
        <div class="bot-label">BOT Policy Rate</div>
        <div class="bot-value {rate_cls}">{BOT_RATE:.2f}%</div>
        <div class="bot-label" style="margin-top:4px;">
            Baseline {BOT_BASELINE:.2f}% — {'▲ elevated' if above_baseline else '▼ normal'}
        </div>
    </div>
    <div class="bot-badge">
        <div class="bot-label">Live USD/THB {'✓' if live_ok else '⚠ fallback'}</div>
        <div class="bot-value {thb_cls}">{usdthb:.2f}</div>
        <div class="bot-label" style="margin-top:4px;">Daily chg: {sign}{thb_chg:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

    if above_baseline:
        st.warning(
            "Rate above baseline. Revolver churn probabilities adjusted **+3 pp** for macro pressure.",
            icon="⚠️",
        )

    st.markdown("---")
    st.markdown("""
    <div class="macro-box">
    <b style="color:#e2e8f0">Macro Commentary</b><br><br>
    The Bank of Thailand holds at <b>2.50%</b>, 100 bp above the post-COVID baseline.
    Household debt remains elevated and THB volatility signals imported cost pressure.
    High-utilisation revolvers are most exposed — proactive restructuring and outreach
    are the priority action for this quarter.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption(f"Data: test split · {pd.Timestamp.now().strftime('%d %b %Y')}")
    st.markdown("""
    <div style="background:#1e293b; border:1px solid #334155; border-left:3px solid #4C72B0;
                border-radius:6px; padding:0.7rem 0.8rem; font-size:0.7rem;
                color:#94a3b8; line-height:1.6;">
        <b style="color:#cbd5e1;">Dataset Note</b><br>
        Sourced from a US credit card portfolio (Kaggle BankChurners).
        The BOT macro adjustment layer demonstrates the analytical framework
        that would apply to Thai customer data and is included for illustrative purposes.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1  ·  Portfolio Overview
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Portfolio Overview":
    st.markdown("""
    <div class="page-header">
        <h1>Portfolio Overview</h1>
        <p>Segment distribution, churn risk summary, and portfolio health metrics</p>
    </div>
    """, unsafe_allow_html=True)

    # KPI row ─────────────────────────────────────────────────────────────────
    total    = len(df)
    churners = int(df["Churned_Actual"].sum())
    avg_prob = df["Churn_Probability"].mean()
    high_risk = int((df["Churn_Probability"] > 0.5).sum())

    c1, c2, c3, c4 = st.columns(4)
    for col, label, value, delta, delta_cls in [
        (c1, "Total Customers",       f"{total:,}",         "Test set",           "delta-neu"),
        (c2, "Observed Churners",      f"{churners:,}",      f"{churners/total*100:.1f}% attrition", "delta-up"),
        (c3, "Avg Churn Probability",  f"{avg_prob*100:.1f}%", "Macro-adjusted",   "delta-neu"),
        (c4, "High-Risk Customers",   f"{high_risk:,}",     f"{high_risk/total*100:.1f}% of portfolio", "delta-up"),
    ]:
        col.markdown(f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            <div class="delta {delta_cls}">{delta}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 1: Distribution + Churn rate ────────────────────────────────────────
    col_l, col_r = st.columns(2, gap="medium")

    with col_l:
        seg_counts = (
            df.groupby("Segment").size()
            .reset_index(name="Count")
            .assign(
                Name=lambda x: x["Segment"].map(SEGMENT_NAMES),
                Color=lambda x: x["Segment"].map(SEGMENT_COLORS),
                Pct=lambda x: x["Count"] / x["Count"].sum() * 100,
            )
        )
        fig1 = go.Figure(go.Bar(
            x=seg_counts["Name"],
            y=seg_counts["Count"],
            marker_color=seg_counts["Color"],
            marker_line_width=0,
            text=seg_counts["Pct"].map(lambda x: f"{x:.1f}%"),
            textposition="outside",
            textfont=dict(color="#1e293b", size=12, family="Inter"),
            hovertemplate="<b>%{x}</b><br>Customers: %{y:,}<br>Share: %{text}<extra></extra>",
        ))
        fig1.update_layout(
            title=dict(text="Segment Distribution", font=dict(size=13, color="#1e293b")),
            yaxis_title="Customers", showlegend=False,
            yaxis_range=[0, seg_counts["Count"].max() * 1.22],
        )
        apply_layout(fig1)
        st.plotly_chart(fig1, use_container_width=True)

    with col_r:
        seg_churn = (
            df.groupby("Segment")["Churned_Actual"]
            .mean()
            .reset_index()
            .rename(columns={"Churned_Actual": "Rate"})
            .assign(
                Name=lambda x: x["Segment"].map(SEGMENT_NAMES),
                Color=lambda x: x["Segment"].map(SEGMENT_COLORS),
                Pct=lambda x: x["Rate"] * 100,
            )
        )
        fig2 = go.Figure(go.Bar(
            x=seg_churn["Name"],
            y=seg_churn["Pct"],
            marker_color=seg_churn["Color"],
            marker_line_width=0,
            text=seg_churn["Pct"].map(lambda x: f"{x:.1f}%"),
            textposition="outside",
            textfont=dict(color="#1e293b", size=12, family="Inter"),
            hovertemplate="<b>%{x}</b><br>Churn Rate: %{y:.1f}%<extra></extra>",
        ))
        fig2.update_layout(
            title=dict(text="Observed Churn Rate by Segment", font=dict(size=13, color="#1e293b")),
            yaxis_title="Churn Rate (%)", showlegend=False,
            yaxis_range=[0, seg_churn["Pct"].max() * 1.28],
        )
        apply_layout(fig2)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<hr style='border-color:#e2e8f0'>", unsafe_allow_html=True)

    # Row 2: Boxplot ──────────────────────────────────────────────────────────
    st.markdown("**Predicted Churn Probability Distribution by Segment**")
    def hex_to_rgba(hex_color: str, alpha: float) -> str:
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    fig3 = go.Figure()
    for sid, sname in SEGMENT_NAMES.items():
        probs = df[df["Segment"] == sid]["Churn_Probability"] * 100
        hex_c = SEGMENT_COLORS[sid]
        fig3.add_trace(go.Box(
            y=probs, name=sname,
            marker=dict(color=hex_c, size=4, opacity=0.5),
            line=dict(color=hex_c, width=2),
            fillcolor=hex_to_rgba(hex_c, 0.18),
            boxmean=True,
            whiskerwidth=0.5,
            hovertemplate=f"<b>{sname}</b><br>Prob: %{{y:.1f}}%<extra></extra>",
        ))
    title_txt = (
        "Predicted Churn Probability (%) — macro-adjusted for elevated BOT rate"
        if above_baseline else "Predicted Churn Probability (%)"
    )
    fig3.update_layout(
        title=dict(text=title_txt, font=dict(size=13, color="#1e293b")),
        yaxis_title="Churn Probability (%)",
        yaxis_ticksuffix="%",
        showlegend=False,
    )
    apply_layout(fig3, height=400)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("<hr style='border-color:#e2e8f0'>", unsafe_allow_html=True)

    # Row 3: Summary table ────────────────────────────────────────────────────
    st.markdown("**Segment Summary Table**")
    rows = []
    for sid, sname in SEGMENT_NAMES.items():
        sub = df[df["Segment"] == sid]
        rows.append({
            "Segment": sname,
            "Customers": len(sub),
            "Portfolio Share": f"{len(sub)/len(df)*100:.1f}%",
            "Actual Churn Rate": f"{sub['Churned_Actual'].mean()*100:.1f}%",
            "Avg Churn Prob": f"{sub['Churn_Probability'].mean()*100:.1f}%",
            "High-Risk (>50%)": int((sub["Churn_Probability"] > 0.5).sum()),
            "Avg Utilization": f"{sub['Avg_Utilization_Ratio'].mean():.1%}",
            "Avg Trans Ct": f"{sub['Total_Trans_Ct'].mean():.0f}",
            "Avg Inactive Mo": f"{sub['Months_Inactive_12_mon'].mean():.1f}",
        })
    st.dataframe(pd.DataFrame(rows).set_index("Segment"), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2  ·  Customer Explorer
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Customer Explorer":
    st.markdown("""
    <div class="page-header">
        <h1>Customer Explorer</h1>
        <p>Enter a customer profile to predict segment, churn probability, and receive a tailored RM action plan</p>
    </div>
    """, unsafe_allow_html=True)

    col_form, col_results = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown("#### Customer Profile")
        with st.form("customer_form"):
            st.markdown("**Demographics**")
            fa   = st.number_input("Age",               min_value=18, max_value=100, value=42)
            fd   = st.number_input("Dependents",        min_value=0,  max_value=10,  value=2)
            fm   = st.number_input("Months on Book",    min_value=1,  max_value=72,  value=36)
            fg   = st.selectbox("Gender",           list(GENDER_MAP.values()),    index=0)
            fe   = st.selectbox("Education Level",  list(EDUCATION_MAP.values()), index=2)
            fmar = st.selectbox("Marital Status",   list(MARITAL_MAP.values()),   index=1)
            finc = st.selectbox("Income Category",  list(INCOME_MAP.values()),    index=1)
            fcard= st.selectbox("Card Category",    list(CARD_MAP.values()),      index=0)

            st.markdown("**Account Behaviour**")
            frel = st.number_input("Total Relationship Count", min_value=1, max_value=6, value=3)
            fin  = st.number_input("Months Inactive (12m)",   min_value=0, max_value=6, value=2)
            fcon = st.number_input("Service Contacts (12m)",  min_value=0, max_value=6, value=2)

            st.markdown("**Credit & Spending**")
            flim  = st.number_input("Credit Limit ($)",         min_value=1000,  max_value=35000, value=8000,  step=500)
            fotb  = st.number_input("Avg Open-to-Buy ($)",      min_value=0,     max_value=35000, value=5000,  step=500)
            ftct  = st.number_input("Total Transactions (12m)", min_value=1,     max_value=200,   value=60)
            ftamt = st.number_input("Total Trans Amount ($)",   min_value=100,   max_value=50000, value=4500,  step=100)
            futil = st.slider("Avg Utilization Ratio", 0.0, 1.0, 0.30, step=0.01, format="%.2f")

            submitted = st.form_submit_button(
                "Predict Churn Risk", use_container_width=True, type="primary"
            )

    with col_results:
        if submitted:
            # Reverse-encode categoricals
            g_enc  = {v: k for k, v in GENDER_MAP.items()}[fg]
            e_enc  = {v: k for k, v in EDUCATION_MAP.items()}[fe]
            m_enc  = {v: k for k, v in MARITAL_MAP.items()}[fmar]
            i_enc  = {v: k for k, v in INCOME_MAP.items()}[finc]
            c_enc  = {v: k for k, v in CARD_MAP.items()}[fcard]

            # Engineered features
            spend_tx    = ftamt / ftct         if ftct > 0 else 0.0
            eng_score   = ftct  / fm           if fm   > 0 else 0.0
            credit_hr   = flim  - fotb

            # KMeans segmentation (scale 8 seg features)
            seg_input  = np.array([[ftct, ftamt, futil, eng_score, fin, fcon, frel, flim]])
            seg_scaled = scaler.transform(seg_input)
            seg_label  = int(kmeans.predict(seg_scaled)[0])
            seg_name   = SEGMENT_NAMES[seg_label]
            seg_color  = SEGMENT_COLORS[seg_label]

            # Churn model (19 features)
            model_input = np.array([[
                fa, fd, fm, frel, fin, fcon,
                flim, fotb, ftct, futil, spend_tx,
                eng_score, credit_hr, seg_label,
                g_enc, e_enc, m_enc, i_enc, c_enc,
            ]])
            churn_prob = float(model.predict_proba(model_input)[0, 1])

            if seg_label == 0 and above_baseline:
                churn_prob = min(churn_prob + MACRO_ADJ, 1.0)

            pct = churn_prob * 100
            risk_label = "Low Risk" if pct < 20 else "Medium Risk" if pct < 50 else "High Risk"
            risk_color = "#22c55e" if pct < 20 else "#f59e0b"     if pct < 50 else "#ef4444"

            st.markdown("#### Prediction Results")

            # Segment card
            st.markdown(f"""
            <div class="metric-card" style="margin-bottom:1rem;">
                <div class="label">Predicted Segment</div>
                <div style="margin-top:0.5rem;">
                    <span class="seg-pill" style="background:{seg_color};">{seg_name}</span>
                </div>
                <div class="delta delta-neu" style="margin-top:0.4rem;">
                    Segment avg churn: {SEGMENT_CHURN_KNOWN[seg_label]:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Gauge
            st.plotly_chart(gauge_chart(churn_prob), use_container_width=True)

            # Risk badge
            macro_note = "&nbsp;<span style='font-size:0.76rem;color:#94a3b8;'>(+3pp macro adj.)</span>" \
                         if seg_label == 0 and above_baseline else ""
            st.markdown(f"""
            <div style="text-align:center; margin:-0.4rem 0 1rem 0;">
                <span style="background:{risk_color}22; color:{risk_color};
                             border:1px solid {risk_color}66;
                             padding:0.3rem 1rem; border-radius:20px;
                             font-weight:600; font-size:0.88rem;">
                    {risk_label}
                </span>{macro_note}
            </div>
            """, unsafe_allow_html=True)

            # Computed feature expander
            with st.expander("Computed Engineered Features", expanded=False):
                cc1, cc2, cc3 = st.columns(3)
                cc1.metric("Spend / Tx",      f"${spend_tx:,.0f}")
                cc2.metric("Engagement Score", f"{eng_score:.2f}")
                cc3.metric("Credit Headroom",  f"${credit_hr:,.0f}")

            # RM recommendation
            st.markdown("#### RM Action Recommendation")
            st.markdown(f"""
            <div class="action-box" style="border-left-color:{seg_color};">
            {RM_ACTIONS[seg_label]}
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="text-align:center; padding:5rem 2rem; color:#94a3b8;">
                <div style="font-size:3rem; margin-bottom:1rem;">💳</div>
                <div style="font-size:0.95rem; font-weight:500; line-height:1.8;">
                    Fill in the customer profile on the left<br>
                    and click <b style="color:#4C72B0;">Predict Churn Risk</b>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3  ·  Segment Deep Dive
# ═══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown("""
    <div class="page-header">
        <h1>Segment Deep Dive</h1>
        <p>Behavioural profile, churn distribution, and SHAP feature importance by segment</p>
    </div>
    """, unsafe_allow_html=True)

    seg_options = {f"{SEGMENT_NAMES[i]}  ·  Seg {i}": i for i in range(5)}
    sel_label   = st.selectbox("Select Segment", list(seg_options.keys()), index=2,
                               label_visibility="collapsed")
    sel_seg     = seg_options[sel_label]
    sel_name    = SEGMENT_NAMES[sel_seg]
    sel_color   = SEGMENT_COLORS[sel_seg]

    sub_df   = df[df["Segment"] == sel_seg]
    shap_mask      = shap_df["Segment"] == sel_seg
    sub_shap_df    = shap_df[shap_mask]
    sub_shap_vals  = shap_vals[shap_mask.values]

    # KPI strip ───────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    for col, label, value in [
        (c1, "Customers in Segment", f"{len(sub_df):,}"),
        (c2, "Actual Churn Rate",    f"{sub_df['Churned_Actual'].mean()*100:.1f}%"),
        (c3, "Avg Churn Probability",f"{sub_df['Churn_Probability'].mean()*100:.1f}%"),
        (c4, "High-Risk (>50%)",     f"{int((sub_df['Churn_Probability']>0.5).sum()):,}"),
    ]:
        col.markdown(f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value" style="color:{sel_color};">{value}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 1: Profile + Distribution ───────────────────────────────────────────
    col_l, col_r = st.columns(2, gap="medium")

    with col_l:
        st.markdown("**Behavioural Profile vs Portfolio Average**")
        profile_map = {
            "Trans Count":       "Total_Trans_Ct",
            "Utilization":       "Avg_Utilization_Ratio",
            "Months Inactive":   "Months_Inactive_12_mon",
            "Contacts (12m)":    "Contacts_Count_12_mon",
            "Engagement Score":  "Engagement_Score",
            "Relationship Ct":   "Total_Relationship_Count",
        }
        feats    = list(profile_map.values())
        labels   = list(profile_map.keys())
        seg_means  = sub_df[feats].mean().values
        port_means = df[feats].mean().values
        ratio      = np.where(port_means != 0, seg_means / port_means, 1.0)

        def hex_to_rgba(hex_color: str, alpha: float) -> str:
            h = hex_color.lstrip("#")
            r2, g2, b2 = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            return f"rgba({r2},{g2},{b2},{alpha})"

        bar_colors = [
            sel_color if r >= 1 else hex_to_rgba(sel_color, 0.5)
            for r in ratio
        ]
        fig_prof = go.Figure()
        fig_prof.add_trace(go.Bar(
            x=labels, y=ratio,
            marker_color=bar_colors,
            marker_line_width=0,
            text=[f"{r:.2f}x" for r in ratio],
            textposition="outside",
            textfont=dict(color="#1e293b", size=11, family="Inter"),
            name=sel_name,
            hovertemplate="<b>%{x}</b><br>Ratio: %{y:.2f}x vs portfolio avg<extra></extra>",
        ))
        fig_prof.add_shape(
            type="line", x0=-0.5, x1=len(labels)-0.5, y0=1, y1=1,
            line=dict(color="#94a3b8", width=1.5, dash="dash"),
        )
        fig_prof.update_layout(
            title=dict(text="Ratio to Portfolio Average  (1.0 = avg)", font=dict(size=13, color="#1e293b")),
            yaxis_title="Ratio", showlegend=False,
            yaxis_range=[0, max(ratio) * 1.3 + 0.15],
            annotations=[dict(
                x=len(labels) - 0.5, y=1.06, text="Portfolio avg",
                showarrow=False, font=dict(size=10, color="#94a3b8"), xanchor="right",
            )],
        )
        apply_layout(fig_prof)
        st.plotly_chart(fig_prof, use_container_width=True)

    with col_r:
        st.markdown("**Churn Probability Distribution**")
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=df["Churn_Probability"] * 100,
            name="Portfolio",
            marker_color="#94a3b8",
            marker_line_width=0,
            opacity=0.35,
            nbinsx=30,
            hovertemplate="Range: %{x:.0f}%<br>Count: %{y}<extra></extra>",
        ))
        fig_hist.add_trace(go.Histogram(
            x=sub_df["Churn_Probability"] * 100,
            name=sel_name,
            marker_color=sel_color,
            marker_line_width=0,
            opacity=0.85,
            nbinsx=30,
            hovertemplate="Range: %{x:.0f}%<br>Count: %{y}<extra></extra>",
        ))
        fig_hist.add_vline(
            x=50, line_dash="dot", line_color="#ef4444", line_width=2,
            annotation_text="50% threshold",
            annotation_position="top right",
            annotation_font=dict(size=10, color="#ef4444"),
        )
        fig_hist.update_layout(
            title=dict(text="Churn Probability (%) Distribution", font=dict(size=13, color="#1e293b")),
            xaxis_title="Churn Probability (%)",
            yaxis_title="Customers",
            barmode="overlay",
            legend=dict(orientation="h", y=-0.22, font=dict(color="#374151")),
        )
        apply_layout(fig_hist)
        st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("<hr style='border-color:#e2e8f0'>", unsafe_allow_html=True)

    # Row 2: SHAP ─────────────────────────────────────────────────────────────
    st.markdown("**Top SHAP Feature Drivers**")

    if len(sub_shap_vals) == 0:
        st.info(f"No SHAP samples for {sel_name} in the 500-row sample set.")
    else:
        mean_abs   = np.abs(sub_shap_vals).mean(axis=0)
        mean_signed = sub_shap_vals.mean(axis=0)
        shap_ser   = pd.Series(mean_abs, index=FEATURE_COLS)
        top12      = shap_ser.nlargest(12).sort_values()

        bar_cols = [
            "#ef4444" if mean_signed[FEATURE_COLS.index(f)] > 0 else "#22c55e"
            for f in top12.index
        ]
        fig_shap = go.Figure(go.Bar(
            x=top12.values,
            y=top12.index,
            orientation="h",
            marker_color=bar_cols,
            marker_line_width=0,
            text=[f"{v:.4f}" for v in top12.values],
            textposition="outside",
            textfont=dict(color="#1e293b", size=11, family="Inter"),
            hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.5f}<extra></extra>",
        ))
        fig_shap.update_layout(
            title=dict(
                text=f"Top 12 SHAP Drivers — {sel_name}  (n={len(sub_shap_vals)} samples)",
                font=dict(size=13, color="#1e293b"),
            ),
            xaxis_title="Mean |SHAP value|",
            yaxis_title=None,
            xaxis_range=[0, top12.values.max() * 1.22],
        )
        apply_layout(fig_shap, height=430)
        st.plotly_chart(fig_shap, use_container_width=True)

        st.markdown("""
        <div style="font-size:0.76rem; color:#64748b; margin-top:-0.5rem; padding-left:0.2rem;">
            <span style="color:#ef4444;">■</span> Increases churn risk &nbsp;|&nbsp;
            <span style="color:#22c55e;">■</span> Decreases churn risk &nbsp;
            (direction = mean signed SHAP for this segment)
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#e2e8f0'>", unsafe_allow_html=True)

    # Row 3: Feature comparison table ────────────────────────────────────────
    st.markdown("**Feature Averages — Segment vs Portfolio**")
    cmp_feats = [
        "Total_Trans_Ct", "Avg_Utilization_Ratio", "Months_Inactive_12_mon",
        "Contacts_Count_12_mon", "Engagement_Score", "Total_Relationship_Count",
        "Credit_Limit", "Avg_Open_To_Buy", "Credit_Headroom",
    ]
    cmp_rows = []
    for f in cmp_feats:
        seg_val  = sub_df[f].mean()
        port_val = df[f].mean()
        delta_pct = (seg_val - port_val) / port_val * 100 if port_val != 0 else 0
        cmp_rows.append({
            "Feature":       f,
            sel_name:        round(seg_val, 3),
            "Portfolio Avg": round(port_val, 3),
            "Δ vs Avg":      round(seg_val - port_val, 3),
            "Δ%":            f"{delta_pct:+.1f}%",
        })
    cmp_df = pd.DataFrame(cmp_rows).set_index("Feature")
    st.dataframe(cmp_df, use_container_width=True)

