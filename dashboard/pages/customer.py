import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(
    page_title="Phân tích hành vi đăng ký tiền gửi",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================
# STYLE CHUNG
# =========================
st.markdown("""
<style>
.block-container {
    padding-top: 1.2rem;
    padding-left: 2.5rem;
    padding-right: 2.5rem;
}
.chart-card {
    background-color: white;
    border-radius: 10px;
    padding: 10px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}
</style>
""", unsafe_allow_html=True)

px.defaults.template = "plotly_white"

DATA_PATH = "data/processed/bank_marketing_raw.csv"
st.markdown("""
<h1 style="font-size:28px;font-weight:600;margin-bottom:0.2rem;">
    Phân tích đặc điểm khách hàng đăng ký tiền gửi
</h1>
<p style="color:#6b7280;font-size:14px;">
    Dashboard học thuật nhằm khám phá đặc điểm và xu hướng đăng ký của khách hàng
</p>
""", unsafe_allow_html=True)

# =========================
# TẢI DỮ LIỆU
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(DATA_PATH)
df = df[['age', 'balance', 'housing', 'loan', 'target', 'job', 'marital', 'default']].copy()

# =========================
# XỬ LÝ ĐẶC TRƯNG
# =========================
df['age_group'] = pd.cut(
    df['age'],
    bins=[0, 29.5, 37.5, 58.5, 120],
    labels=['< 29', '29–37', '37–58', '> 58']
)

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

BALANCE_COLORS = [
    "#e0f2f1",
    "#b2dfdb",
    "#80cbc4",
    "#4db6ac",
    "#00796b"
]

# =========================================================
# KPI – TÓM TẮT TRI THỨC
# =========================================================
housing_rate = df.groupby('housing')['target'].mean()
housing_impact = housing_rate['no'] - housing_rate['yes']

age_rate = df.groupby('age_group')['target'].mean()
age_polarization = (
    (age_rate['< 29'] + age_rate['> 58']) / 2
    - (age_rate['29–37'] + age_rate['37–58']) / 2
)

balance_rate = df.groupby('balance_group')['target'].mean()
balance_threshold = balance_rate[balance_rate > 0.15].index[0]

job_age = df.groupby(['job', 'age_group'], observed=True)['target'].mean().reset_index()
job_variance = job_age.groupby('job')['target'].std().mean()

k1, k2, k3, k4 = st.columns(4)

def kpi_card(title, value, subtitle):
    st.markdown(f"""
    <div style="
        background-color:white;
        border-radius:10px;
        padding:14px;
        box-shadow:0 1px 3px rgba(0,0,0,0.08);
        text-align:center;">
        <div style="font-size:13px;color:#6b7280;">{title}</div>
        <div style="font-size:26px;font-weight:600;color:#111827;">{value}</div>
        <div style="font-size:12px;color:#9ca3af;">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)

with k1:
    kpi_card("Tác động của vay mua nhà", f"{housing_impact:.1%}", "Rào cản mạnh nhất")

with k2:
    kpi_card("Phân cực theo độ tuổi", f"{age_polarization:.1%}", "Hiệu ứng vòng đời tài chính")

with k3:
    kpi_card("Ngưỡng số dư tối thiểu", balance_threshold, "Năng lực tài chính tối thiểu")

with k4:
    kpi_card("Vai trò nghề nghiệp", "Phụ thuộc ngữ cảnh", "Không phải biến độc lập")

# =========================================================
# BIỂU ĐỒ
# =========================================================
c1, c2 = st.columns(2)

AGE_COLORS = {
    '< 29': '#bfdbfe',
    '29–37': '#93c5fd',
    '37–58': '#3b82f6',
    '> 58': '#1e40af'
}

fig_job_age = px.bar(
    job_age,
    x='job',
    y='target',
    color='age_group',
    barmode='group',
    title="Tỷ lệ đăng ký theo nghề nghiệp và nhóm tuổi",
    color_discrete_map=AGE_COLORS
)
COMMON_MARGIN = dict(l=40, r=20, t=60, b=90)
fig_job_age.update_layout(
    height=360,
    margin=COMMON_MARGIN,
    yaxis_title="Tỷ lệ đăng ký",
    xaxis_title=None,
    xaxis_tickangle=-25,
    legend_title_text="Nhóm tuổi"
)


with c1:
    st.plotly_chart(fig_job_age, use_container_width=True)

age_balance = df.groupby(['age_group', 'balance_group'], observed=True)['target'].mean().reset_index()

fig_age_balance = px.bar(
    age_balance,
    x='age_group',
    y='target',
    color='balance_group',
    barmode='group',
    title="Tỷ lệ đăng ký theo nhóm tuổi và mức số dư",
    color_discrete_sequence=BALANCE_COLORS
)
fig_age_balance.update_layout(
    height=360,
    margin=COMMON_MARGIN,
    xaxis_title="Nhóm tuổi",
    yaxis_title= None,
    legend_title_text="Nhóm số dư"
)

with c2:
    st.plotly_chart(fig_age_balance, use_container_width=True)
# =========================================================
# ROW 2 – PHÂN TÍCH TƯƠNG TÁC HỖ TRỢ
# =========================================================
c3, c4 = st.columns([1, 1.2])

# ---------- Housing × Job (Slope Chart) ----------
hj = df.groupby(['housing', 'job'])['target'].mean().reset_index()
pivot_hj = hj.pivot(index='job', columns='housing', values='target').reset_index()

fig_slope = go.Figure()

for _, row in pivot_hj.iterrows():
    fig_slope.add_trace(
        go.Scatter(
            x=['Có vay mua nhà', 'Không vay mua nhà'],
            y=[row['yes'], row['no']],
            mode='lines+markers',
            name=row['job'],
            line=dict(width=2),
            marker=dict(size=6),
            hovertemplate=(
                f"Nghề nghiệp: {row['job']}<br>"
                "Tình trạng nhà ở: %{x}<br>"
                "Tỷ lệ đăng ký: %{y:.1%}<extra></extra>"
            )
        )
    )
COMMON_HEIGHT = 380
COMMON_MARGIN = dict(l=40, r=40, t=60, b=60)

fig_slope.update_layout(
    title="Ảnh hưởng của vay mua nhà theo từng nhóm nghề nghiệp",
    height=COMMON_HEIGHT,
    margin=COMMON_MARGIN,
    yaxis_title="Tỷ lệ đăng ký",
    xaxis_title=None,
    legend_title="Nghề nghiệp"
)


# ---------- Job × Balance Group (Heatmap) ----------
jb = df.groupby(['job', 'balance_group'], observed=True)['target'].mean().reset_index()
pivot_jb = jb.pivot(index='job', columns='balance_group', values='target')

fig_heatmap = px.imshow(
    pivot_jb,
    color_continuous_scale="Blues",
    aspect="auto",
    labels=dict(
        x="Nhóm số dư",
        y="Nghề nghiệp",
        color="Tỷ lệ đăng ký"
    ),
    title="Tỷ lệ đăng ký theo nghề nghiệp và mức số dư"
)
fig_heatmap.update_layout(
    height=COMMON_HEIGHT,
    margin=COMMON_MARGIN,
    coloraxis_colorbar=dict(
        thickness=14,
        len=0.75,
        y=0.5
    )
)

with c3:
    st.plotly_chart(fig_slope, use_container_width=True)

with c4:
    st.plotly_chart(fig_heatmap, use_container_width=True)

# =========================
# GHI CHÚ
# =========================
st.caption(
    "Lưu ý: Các giá trị thể hiện tỷ lệ đăng ký trung bình trong từng phân khúc khách hàng."
)
## run py -m streamlit run dashboard/app.py