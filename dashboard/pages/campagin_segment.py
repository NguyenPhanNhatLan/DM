import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from services.segmentation_service import SegConfig, run_segmentation_pipeline

DATA_PATH = "data/processed/bank_marketing_raw.csv"

# ---------- CSS ----------
st.markdown(
    """
<style>
/* widen container a bit */
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1200px; }

/* card style */
.kpi-card {
  padding: 16px 18px;
  border-radius: 16px;
  border: 1px solid rgba(49,51,63,0.12);
  box-shadow: 0 6px 22px rgba(0,0,0,0.06);
  background: rgba(255,255,255,0.02);
}
.kpi-title { font-size: 12px; opacity: 0.75; margin-bottom: 8px; }
.kpi-value { font-size: 34px; font-weight: 700; line-height: 1.1; }
.kpi-sub { font-size: 12px; opacity: 0.7; margin-top: 6px; }
.badge {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 12px;
  border: 1px solid rgba(49,51,63,0.15);
  opacity: 0.85;
}
</style>
""",
    unsafe_allow_html=True,
)


def kpi_card(title, value, sub=""):
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-title">{title}</div>
          <div class="kpi-value">{value}</div>
          <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render():
    st.title("Tổng quan chiến dịch (Overview)")

    # Sidebar controls
    with st.sidebar:
        st.header("Thiết lập")
        budget = st.slider("Ngân sách gọi (tỷ lệ khách hàng)", 0.05, 0.50, 0.10, 0.05)
        k = st.slider("Số cụm K (K-Means)", 3, 12, 8, 1)
        show_details = st.checkbox("Hiện chi tiết mô hình", value=False)

    df = pd.read_csv(DATA_PATH)
    cfg = SegConfig(call_budget=budget, k=k)
    out = run_segmentation_pipeline(df, cfg)

    overall = out["overall_conv"]
    cand_conv = out["cand_conv"]
    lift = out["cand_lift"]
    n_called = len(out["candidates"])
    n_total = len(out["df_scored"])

    # Headline insight
    st.markdown(
        f"""
        <span class="badge">Ngân sách gọi: {int(budget*100)}%</span>
        <span class="badge">K = {k}</span>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    st.markdown(
        f"**Kết luận nhanh:** Nếu chỉ gọi **Top {int(budget*100)}%** khách hàng có xác suất cao nhất, "
        f"tỷ lệ đồng ý tăng từ **{overall:.3f}** lên **{cand_conv:.3f}** (tương đương **{lift:.2f}×**)."
    )

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Tỷ lệ đồng ý (toàn bộ)", f"{overall:.3f}", "Baseline gọi ngẫu nhiên")
    with c2:
        kpi_card(
            "Quy mô gọi",
            f"{n_called:,}/{n_total:,}",
            f"≈ {int(budget*100)}% khách hàng",
        )
    with c3:
        kpi_card(
            "Tỷ lệ đồng ý (nhóm gọi)",
            f"{cand_conv:.3f}",
            "Precision trong nhóm được gọi",
        )
    with c4:
        kpi_card("Lift so với baseline", f"{lift:.2f}×", "Hiệu quả tăng bao nhiêu lần")

    st.divider()

    # Mini chart: Precision/Lift at budgets (using metrics already computed)
    st.subheader("Hiệu quả theo các ngưỡng gọi phổ biến")
    m = out["metrics"]
    budgets = ["Top 5%", "Top 10%", "Top 20%", "Top 30%"]
    precs = [m["top5_prec"], m["top10_prec"], m["top20_prec"], m["top30_prec"]]
    lifts = [m["top5_lift"], m["top10_lift"], m["top20_lift"], m["top30_lift"]]

    colA, colB = st.columns(2)
    with colA:
        fig = plt.figure()
        plt.bar(budgets, precs)
        plt.title("Precision@K (tỷ lệ YES trong nhóm được gọi)")
        plt.ylabel("Precision")
        plt.tight_layout()
        st.pyplot(fig)
    with colB:
        fig = plt.figure()
        plt.bar(budgets, lifts)
        plt.title("Lift@K (so với gọi ngẫu nhiên)")
        plt.ylabel("Lift (×)")
        plt.tight_layout()
        st.pyplot(fig)

    if show_details:
        st.subheader("Chất lượng mô hình (hold-out)")
        st.info(
            f"ROC-AUC = {m['roc_auc']:.3f} | PR-AUC = {m['pr_auc']:.3f}\n\n"
            f"- ROC-AUC đo khả năng phân biệt 0/1\n"
            f"- PR-AUC phù hợp khi dữ liệu mất cân bằng (YES ít)"
        )


render()
