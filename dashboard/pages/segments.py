# =========================================================
# CUSTOMER KNOWLEDGE DASHBOARD (ACADEMIC – FINAL VERSION)
# =========================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import os

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Customer Subscription Analytics",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================
# GLOBAL STYLE (ACADEMIC)
# =========================
st.markdown("""
<style>
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 0rem;
    padding-left: 2.5rem;
    padding-right: 2.5rem;
}

.chart-card {
    background-color: white;
    border-radius: 10px;
    padding: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}
</style>
""", unsafe_allow_html=True)

# =========================
# PLOTLY DEFAULT
# =========================
px.defaults.template = "plotly_white"

# =========================
# TITLE (ACADEMIC HIGHLIGHT)
# =========================
st.markdown("""
<h1 style="font-size:28px;font-weight:600;margin-bottom:0.15rem;">
    <span style="color:#2563eb;">Customer</span> Subscription Analytics
</h1>
<p style="color:#6b7280;font-size:14px;margin-top:0;">
    One-screen academic dashboard for customer knowledge discovery
</p>
""", unsafe_allow_html=True)

# =========================
# LOAD DATA
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "..","..", "data", "customer.csv"))
df = df[['age', 'balance', 'housing', 'loan', 'target']].copy()

# =========================
# FEATURE ENGINEERING
# =========================
df['age_group'] = pd.cut(
    df['age'],
    bins=[0, 29.5, 37.5, 58.5, 120],
    labels=['< 29', '29–37', '37–58', '> 58'],
    right=True
)

# enforce logical order (ACADEMIC DETAIL)
df['age_group'] = pd.Categorical(
    df['age_group'],
    categories=['< 29', '29–37', '37–58', '> 58'],
    ordered=True
)

df['balance_group'] = pd.cut(
    df['balance'],
    bins=[-9999, -46.5, 60.5, 798.5, 1578.5, 99999999],
    labels=[
        '< -46.5',
        '-46.5 – 60.5',
        '60.5 – 798.5',
        '798.5 – 1578.5',
        '> 1578.5'
    ],
    right=False
)

# =========================
# KPI ROW (ACADEMIC EMPHASIS)
# =========================
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(f"""
    <div style="text-align:center;">
        <div style="font-size:26px;font-weight:500;color:#374151;">
            {len(df):,}
        </div>
        <div style="font-size:14px;color:#6b7280;">Customers</div>
    </div>
    """, unsafe_allow_html=True)

k2.metric("Subscribers", f"{int(df['target'].sum()):,}")

k3.metric(
    "Subscription Rate",
    f"{df['target'].mean()*100:.1f}%"
)

k4.metric("Avg Balance", f"{df['balance'].mean():,.0f}")

# =========================
# COLOR PALETTE
# =========================
BLUE = "#2563eb"
TEAL = "#0d9488"
GRAY = "#4b5563"
LIGHT = "#c7d2fe"

# =========================
# ROW 1 – UNIVARIATE
# =========================
c1, c2, c3 = st.columns(3)

age_rate = df.groupby('age_group', observed=True)['target'].mean().reset_index()
max_idx = age_rate['target'].idxmax()

fig_age = px.bar(
    age_rate,
    x='age_group',
    y='target',
    title="Subscription Rate by Age Group",
    text=age_rate['target'].apply(lambda x: f"{x:.1%}")
)

fig_age.update_traces(
    marker_color=[BLUE if i == max_idx else LIGHT for i in range(len(age_rate))],
    opacity=0.85,
    hovertemplate="Age group: %{x}<br>Rate: %{y:.1%}<extra></extra>"
)

fig_age.add_annotation(
    text="This age segment exhibits the highest responsiveness",
    x=age_rate.loc[max_idx, 'age_group'],
    y=age_rate.loc[max_idx, 'target'],
    showarrow=True,
    arrowhead=2,
    ax=0, ay=-30
)

housing_rate = df.groupby('housing')['target'].mean().reset_index()
fig_housing = px.bar(
    housing_rate,
    x='housing',
    y='target',
    title="Subscription Rate by Housing Loan",
    text=housing_rate['target'].apply(lambda x: f"{x:.1%}"),
    color_discrete_sequence=[GRAY]
)

balance_rate = df.groupby('balance_group', observed=True)['target'].mean().reset_index()
fig_balance = px.bar(
    balance_rate,
    x='balance_group',
    y='target',
    title="Subscription Rate by Balance Group",
    text=balance_rate['target'].apply(lambda x: f"{x:.1%}"),
    color_discrete_sequence=[TEAL]
)

fig_balance.update_yaxes(range=[0, 0.18])

for fig in [fig_age, fig_housing, fig_balance]:
    fig.update_layout(
        height=260,
        margin=dict(l=20, r=20, t=50, b=20),
        yaxis_title=None,
        xaxis_title=None,
        transition=dict(duration=500, easing="cubic-in-out")
    )

with c1:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.plotly_chart(fig_age, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.plotly_chart(fig_housing, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

with c3:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.plotly_chart(fig_balance, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# ROW 2 – BIVARIATE
# =========================
c4, c5 = st.columns(2)

hb = df.groupby(['housing', 'balance_group'], observed=True)['target'].mean().reset_index()
fig_hb = px.bar(
    hb,
    x='target',
    y='balance_group',
    color='housing',
    barmode='group',
    title="Housing Loan × Balance Group",
    color_discrete_sequence=[BLUE, "#9ca3af"]
)

la = df.groupby(['loan', 'age_group'], observed=True)['target'].mean().reset_index()
fig_la = px.bar(
    la,
    x='target',
    y='age_group',
    color='loan',
    barmode='group',
    title="Personal Loan × Age Group",
    color_discrete_sequence=[GRAY, "#93c5fd"]
)

for fig in [fig_hb, fig_la]:
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=50, b=20),
        yaxis_title=None,
        xaxis_title=None,
        transition=dict(duration=500, easing="cubic-in-out")
    )
    fig.update_traces(opacity=0.85)

with c4:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.plotly_chart(fig_hb, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

with c5:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.plotly_chart(fig_la, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# FOOTNOTE
# =========================
st.caption(
    "Note: All values represent mean subscription rate within each customer segment."
)

## run py -m streamlit run dashboard/app.py