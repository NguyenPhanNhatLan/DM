import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from services.segmentation_service import SegConfig, run_segmentation_pipeline

DATA_PATH = "data/processed/bank_marketing_raw.csv"

st.set_page_config(page_title="Overview Campaign", layout="wide")

st.markdown(
    """
<style>
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1200px; }

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
  margin-right: 6px;
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


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


# ===== helpers giống notebook =====
def precision_at_k_local(y_true, p, k_frac=0.2):
    y_true = pd.Series(y_true).reset_index(drop=True)
    p = np.asarray(p)
    n = len(p)
    k = max(1, int(n * k_frac))
    idx = np.argsort(-p)[:k]
    return float(y_true.iloc[idx].mean())


def lift_at_k_local(y_true, p, k_frac=0.2):
    base = float(pd.Series(y_true).mean())
    prec = precision_at_k_local(y_true, p, k_frac)
    return prec / base if base > 0 else np.nan


def render():
    st.title("Tổng quan hiệu quả chiến dịch")

    with st.sidebar:
        st.header("Thiết lập")
        budget = st.slider("Ngân sách gọi (tỷ lệ khách hàng)", 0.05, 0.50, 0.10, 0.05)
        show_details = st.checkbox("Hiện chi tiết mô hình", value=False)

    df = load_data(DATA_PATH)

    cfg = SegConfig(call_budget=budget)
    out = run_segmentation_pipeline(df, cfg)

    m = out["metrics"]
    overall = float(m.get("base_conv", 0.0))
    cand_conv = float(out.get("cand_conv", 0.0))
    lift = float(out.get("cand_lift", 0.0))

    n_called = len(out.get("candidates", []))
    n_total = len(out.get("df_scored", []))

    # Badge
    st.markdown(
        f"""
        <span class="badge">Ngân sách gọi: {int(budget*100)}%</span>
        <span class="badge">Baseline: {overall:.3f}</span>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    # Headline insight
    if overall > 0:
        st.markdown(
            f"**Kết luận nhanh:** Nếu chỉ gọi **Top {int(budget*100)}%** khách hàng có xác suất cao nhất, "
            f"tỷ lệ đồng ý tăng từ **{overall:.3f}** lên **{cand_conv:.3f}** (tương đương **{lift:.2f}×**)."
        )
    else:
        st.markdown(
            f"**Kết luận nhanh:** Nếu chỉ gọi **Top {int(budget*100)}%**, "
            f"tỷ lệ đồng ý trong nhóm gọi là **{cand_conv:.3f}**."
        )

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Tỷ lệ đồng ý (baseline)", f"{overall:.3f}", "BASE_CONV trên toàn bộ dữ liệu")
    with c2:
        kpi_card("Quy mô gọi", f"{n_called:,}/{n_total:,}", f"≈ {int(budget*100)}% khách hàng")
    with c3:
        kpi_card("Tỷ lệ đồng ý (nhóm gọi)", f"{cand_conv:.3f}", "Precision trong nhóm được gọi")
    with c4:
        kpi_card("Lift so với baseline", f"{lift:.2f}×", "cand_conv / base_conv")

    st.divider()

    # =========================
    # Bar charts (hold-out)
    # =========================
    st.subheader("Hiệu quả theo các ngưỡng gọi phổ biến (hold-out)")

    budgets_lbl = ["Top 5%", "Top 10%", "Top 20%", "Top 30%"]
    precs = [
        m.get("top5_prec", 0.0),
        m.get("top10_prec", 0.0),
        m.get("top20_prec", 0.0),
        m.get("top30_prec", 0.0),
    ]
    lifts = [
        m.get("top5_lift", 0.0),
        m.get("top10_lift", 0.0),
        m.get("top20_lift", 0.0),
        m.get("top30_lift", 0.0),
    ]

    colA, colB = st.columns(2)
    with colA:
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        ax.bar(budgets_lbl, precs)
        ax.set_title("Precision@K (tỷ lệ YES trong Top X%)")
        ax.set_ylabel("Precision")
        ax.tick_params(axis="x", labelrotation=15)
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

    with colB:
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        ax.bar(budgets_lbl, lifts)
        ax.set_title("Lift@K (so với baseline)")
        ax.set_ylabel("Lift (×)")
        ax.tick_params(axis="x", labelrotation=15)
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

    # =========================
    # Line charts (toàn bộ dữ liệu) + highlight slider budget
    # =========================
    st.divider()
    st.subheader("Campaign performance by call budget (toàn bộ dữ liệu)")

    df_scored = out.get("df_scored")
    if df_scored is None or "p_yes" not in df_scored.columns or cfg.target_col not in df_scored.columns:
        st.warning("Không đủ dữ liệu để vẽ line chart (cần df_scored có cột p_yes và target).")
    else:
        budgets = np.linspace(0.05, 0.50, 10)
        prec_list, lift_list = [], []

        p_all = df_scored["p_yes"].values
        y_all = df_scored[cfg.target_col].values

        for b in budgets:
            prec_list.append(precision_at_k_local(y_all, p_all, b))
            lift_list.append(lift_at_k_local(y_all, p_all, b))

        x = (budgets * 100).astype(int)

        # current budget highlight
        b_cur = float(budget)
        x_cur = int(round(b_cur * 100))
        prec_cur = precision_at_k_local(y_all, p_all, b_cur)
        lift_cur = lift_at_k_local(y_all, p_all, b_cur)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Precision@Budget
        axes[0].plot(x, prec_list, marker="o")
        axes[0].axvline(x_cur, linestyle="--", linewidth=1)
        axes[0].scatter([x_cur], [prec_cur], s=60, zorder=5)
        axes[0].annotate(
            f"{x_cur}%\n{prec_cur:.3f}",
            (x_cur, prec_cur),
            textcoords="offset points",
            xytext=(8, 8),
        )
        axes[0].set_title("Precision@Budget (Top X% called)")
        axes[0].set_xlabel("Call budget (%)")
        axes[0].set_ylabel("Precision")

        # Lift@Budget
        axes[1].plot(x, lift_list, marker="o")
        axes[1].axvline(x_cur, linestyle="--", linewidth=1)
        axes[1].scatter([x_cur], [lift_cur], s=60, zorder=5)
        axes[1].annotate(
            f"{x_cur}%\n{lift_cur:.2f}x",
            (x_cur, lift_cur),
            textcoords="offset points",
            xytext=(8, 8),
        )
        axes[1].set_title("Lift@Budget (Top X% called)")
        axes[1].set_xlabel("Call budget (%)")
        axes[1].set_ylabel("Lift vs overall")

        fig.suptitle("Campaign performance by call budget", fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

    # Details
    if show_details:
        st.subheader("Chất lượng mô hình (hold-out)")
        st.info(
            f"ROC-AUC = {m.get('roc_auc', 0.0):.3f} | PR-AUC = {m.get('pr_auc', 0.0):.3f}\n\n"
            f"- ROC-AUC đo khả năng phân biệt 0/1\n"
            f"- PR-AUC phù hợp khi dữ liệu mất cân bằng (YES ít)"
        )


render()
