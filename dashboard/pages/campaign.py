# campaign.py
# Streamlit dashboard for "Phân tích chiến dịch" – Bank Marketing dataset
# Run: streamlit run campaign.py

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from services.utils_campaign import has, bin_duration_seconds, bin_pdays, conversion_rate, agg_rate_and_n, load_data, DATA_PATH


# Sidebar: data + filters
st.sidebar.header("📁 Dữ liệu & Bộ lọc")

try:
    df_raw = load_data(DATA_PATH)
except Exception as e:
    st.error(f"Không đọc được file \n\nChi tiết lỗi: {e}")
    st.stop()

# Basic column availability hints
required_campaign_cols = ['date','contact', 'duration', 'campaign', 'pdays','previous','poutcome']
available_cols = [c for c in required_campaign_cols if has(df_raw, c)]
with st.sidebar.expander("Các biến có trong dashboard"):
    st.write(available_cols if available_cols else "Không tìm thấy các cột chiến dịch phổ biến.")

include_duration = st.sidebar.toggle(
    "After-call analysis (Include duration)",
    value=False,
    help="Bật để thêm biến duration (chỉ biết sau cuộc gọi). Tắt để làm chiến lược pre-call (actionable).",
)

# Build filter widgets based on existing columns
df = df_raw.copy()

years = sorted(df["year"].unique())

sel_years = st.sidebar.multiselect(
    "Năm triển khai",
    years,
    default=years
)

# đảm bảo df có date
if "date" in df_raw.columns:
    # min/max date
    min_date = df_raw["date"].min().date()
    max_date = df_raw["date"].max().date()

    # date range picker
    start_date, end_date = st.sidebar.date_input(
        "Chọn khoảng ngày",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    # month multiselect (dựa trên month_name hoặc month)
    # nếu bạn đã tạo month_name trong load_data()
    if "month_name" in df_raw.columns and "year" in df_raw.columns:
        # hiển thị dạng "2025 - January" để không lẫn giữa các năm
        df_raw["year_month_label"] = df_raw["date"].dt.to_period("M").astype(str)  # "YYYY-MM"
        months = sorted(df_raw["year_month_label"].unique().tolist())

        sel_months = st.sidebar.multiselect(
            "Chọn tháng (YYYY-MM)",
            months,
            default=months
        )
    else:
        sel_months = None

else:
    st.sidebar.warning("Dataset chưa có cột 'date' nên không thể lọc theo ngày/tháng.")
    start_date, end_date, sel_months = None, None, None


# Contact filter
if has(df, "contact"):
    contacts = sorted(df["contact"].dropna().astype(str).unique().tolist())
    sel_contacts = st.sidebar.multiselect("Contact", contacts, default=contacts)
else:
    sel_contacts = None

# Poutcome filter
if has(df, "poutcome"):
    poutcomes = sorted(df["poutcome"].dropna().astype(str).unique().tolist())
    sel_poutcomes = st.sidebar.multiselect("Previous outcome (poutcome)", poutcomes, default=poutcomes)
else:
    sel_poutcomes = None

# Campaign range filter
if has(df, "campaign"):
    cmin, cmax = int(np.nanmin(df["campaign"])), int(np.nanmax(df["campaign"]))
    sel_campaign = st.sidebar.slider("campaign (số lần liên hệ trong chiến dịch)", cmin, cmax, (cmin, min(cmax, 6)))
else:
    sel_campaign = None

# Previous range filter
if has(df, "previous"):
    pmin, pmax = int(np.nanmin(df["previous"])), int(np.nanmax(df["previous"]))
    sel_previous = st.sidebar.slider("previous (số lần liên hệ trước đó)", pmin, pmax, (pmin, min(pmax, 3)))
else:
    sel_previous = None


def apply_filters(d: pd.DataFrame) -> pd.DataFrame:
    out = d.copy()
    if "year" in out.columns and sel_years is not None:
        out = out[out["year"].isin(sel_years)]
    # --- date range filter ---
    if "date" in out.columns and start_date is not None and end_date is not None:
        out = out[
            (out["date"].dt.date >= start_date) &
            (out["date"].dt.date <= end_date)
        ]

    # --- month filter (YYYY-MM label) ---
    if "date" in out.columns and sel_months is not None:
        out = out[out["date"].dt.to_period("M").astype(str).isin(sel_months)]
    if sel_contacts is not None:
        out = out[out["contact"].astype(str).isin(sel_contacts)]
    if sel_poutcomes is not None:
        out = out[out["poutcome"].astype(str).isin(sel_poutcomes)]
    if sel_campaign is not None:
        out = out[(out["campaign"] >= sel_campaign[0]) & (out["campaign"] <= sel_campaign[1])]
    if sel_previous is not None:
        out = out[(out["previous"] >= sel_previous[0]) & (out["previous"] <= sel_previous[1])]

    # Duration toggle: if not include_duration, drop it from analysis pages but keep rows
    # (We still keep rows; we just won't show duration charts.)
    return out


df_f = apply_filters(df)

# Header + KPI
st.title("CAMPAIGN DASHBOARD")

# kiểm tra target y (0/1)
if "y" not in df_f.columns or df_f["y"].isna().all():
    st.warning("Không tìm thấy cột nhãn 'y' (0/1). Dashboard vẫn hiển thị số lượng nhưng conversion rate không tính được.")
    y_available = False
else:
    y_available = True


kpi1, kpi2, kpi3, kpi4 = st.columns(4)

total_contacts = len(df_f)

if y_available:
    conversions = int((df_f["y"] == 1).sum())
    conv_rate = float(df_f["y"].mean())
else:
    conversions = 0
    conv_rate = float("nan")

kpi1.metric("Tổng contacts", f"{total_contacts:,}")
kpi2.metric("Số chuyển đổi (y=1)", f"{conversions:,}")
kpi3.metric("Conversion rate", f"{conv_rate*100:.2f}%" if not np.isnan(conv_rate) else "N/A")
if "duration" in df_f.columns:
    kpi4.metric(f'Thời lượng trung bình (giây)', df_f['duration'].mean().round(2) if 'duration' in df_f.columns else 'N/A')

# KPI 4: ưu tiên theo cột có sẵn (poutcome > pdays > campaign)
elif "poutcome" in df_f.columns:
    # chuẩn hóa để tránh case/space
    pout = df_f["poutcome"].astype(str).str.strip().str.lower()
    success_share = (pout == "success").mean()
    kpi4.metric("Tỷ trọng poutcome=success", f"{success_share*100:.2f}%")

elif "pdays" in df_f.columns:
    never_share = (df_f["pdays"] == -1).mean()
    kpi4.metric("Tỷ trọng pdays=-1", f"{never_share*100:.2f}%")

elif "campaign" in df_f.columns:
    med_campaign = float(df_f["campaign"].median()) if df_f["campaign"].notna().any() else float("nan")
    kpi4.metric("Median campaign", f"{med_campaign:.0f}" if not np.isnan(med_campaign) else "N/A")

else:
    kpi4.metric("KPI chiến dịch", "N/A")

st.divider()


# Tabs (pages)
tab_overview, tab_time, tab_pressure, tab_channel, tab_duration, tab_rules = st.tabs(
    ["Tổng quan", "Theo thời gian", "Tần suất & lịch sử", "Kênh liên hệ", "Duration", "Tri thức & Luật"]
)

# -----------------------------
# Overview
# -----------------------------
with tab_overview:
    st.subheader("Tổng quan hiệu quả chiến dịch")
    c1, c2 = st.columns(2)

    # Ưu tiên dùng nhãn tháng chuẩn (YYYY-MM) nếu có date
    month_col = None
    if "date" in df_f.columns:
        # tạo label YYYY-MM ngay tại đây để chắc chắn luôn có
        df_f = df_f.copy()
        df_f["year_month"] = df_f["date"].dt.to_period("M").astype(str)
        month_col = "year_month"
    elif "month" in df_f.columns:
        month_col = "month"

    with c1:
        if month_col is not None and "y" in df_f.columns and df_f["y"].notna().any():
            g = (
                df_f.groupby(month_col)
                .agg(n=("y", "size"), conversion_rate=("y", "mean"))
                .reset_index()
                .sort_values(month_col)
            )

            fig = px.bar(
                g,
                x=month_col,
                y="conversion_rate",
                hover_data=["n"],
                title="Conversion rate theo tháng",
            )
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)
        elif month_col is None:
            st.info("Không có cột 'date' hoặc 'month' để phân tích theo tháng.")
        else:
            st.info("Không có cột 'y' (0/1) hợp lệ để tính conversion rate.")

    with c2:
        if month_col is not None:
            vol = (
                df_f.groupby(month_col)
                .size()
                .reset_index(name="calls")
                .sort_values(month_col)
            )
            fig = px.bar(vol, x=month_col, y="calls", title="Số cuộc gọi theo tháng")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Không có cột 'date' hoặc 'month' để thống kê số cuộc gọi theo tháng.")

    # Tri thức tự động: best/worst month
    if month_col is not None and "y" in df_f.columns and df_f["y"].notna().any():
        g = (
            df_f.groupby(month_col)
            .agg(n=("y", "size"), conversion_rate=("y", "mean"))
            .reset_index()
            .sort_values("conversion_rate", ascending=False)
        )

        if len(g) >= 2:
            best = g.iloc[0]
            worst = g.iloc[-1]
            st.write(
                f"- Tháng hiệu quả nhất: **{best[month_col]}** (CR ~ **{best['conversion_rate']*100:.2f}%**, n={int(best['n']):,})"
            )
            st.write(
                f"- Tháng kém nhất: **{worst[month_col]}** (CR ~ **{worst['conversion_rate']*100:.2f}%**, n={int(worst['n']):,})"
            )
        else:
            st.info("Không đủ tháng khác nhau để so sánh best/worst.")

    # ====== NEW: Donut + Daily dual-axis ======
    st.divider()
    d1, d2 = st.columns(2)

    # 1️⃣ Pie / Donut: Subscription Result (Yes vs No)
    with d1:
        if "y" in df_f.columns and df_f["y"].notna().any():
            donut = (
                df_f["y"]
                .value_counts(dropna=True)
                .rename_axis("y")
                .reset_index(name="count")
            )
            donut["label"] = donut["y"].map({1: "Yes", 0: "No"})

            fig = px.pie(
                donut,
                names="label",
                values="count",
                hole=0.55,
                title="Subscription Result (Yes vs No)",
            )
            fig.update_traces(textinfo="percent+label")

            st.plotly_chart(fig, use_container_width=True)

            total = int(donut["count"].sum())
            yes_n = int(donut.loc[donut["y"] == 1, "count"].sum())
            st.caption(f"Tổng: {total:,} | Yes: {yes_n:,} | CR: {(yes_n/total):.2%}")
        else:
            st.info("Không có cột 'y' (0/1) hợp lệ để vẽ tỷ lệ Yes/No.")

    # 3️⃣ Line chart: Daily Contact Volume vs Conversion (day 1-31)
    with d2:
        if "date" in df_f.columns and "y" in df_f.columns and df_f["y"].notna().any():

            tmp = df_f.copy()
            tmp["day"] = tmp["date"].dt.day  # lấy day từ datetime

            daily = (
                tmp.groupby("day")
                .agg(
                    calls=("day", "size"),
                    conversion_rate=("y", "mean"),
                )
                .reset_index()
                .sort_values("day")
            )

            fig = px.line(
                daily,
                x="day",
                y="calls",
                markers=True,
                title="Daily Contact Volume vs Conversion",
            )

            # conversion rate lên trục phụ (y2)
            fig.add_scatter(
                x=daily["day"],
                y=daily["conversion_rate"],
                mode="lines+markers",
                name="conversion_rate",
                yaxis="y2",
            )

            fig.update_layout(
                yaxis=dict(title="calls"),
                yaxis2=dict(
                    title="conversion_rate",
                    overlaying="y",
                    side="right",
                    tickformat=".0%",
                ),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )

            st.plotly_chart(fig, use_container_width=True)

            peak_calls_day = int(daily.loc[daily["calls"].idxmax(), "day"])
            best_cr_row = daily.loc[daily["conversion_rate"].idxmax()]

            st.caption(f"Ngày gọi nhiều nhất: day={peak_calls_day} (calls={int(daily['calls'].max()):,})")
            st.caption(
                f"Ngày hiệu quả nhất: day={int(best_cr_row['day'])} "
                f"(CR={best_cr_row['conversion_rate']:.2%}, calls={int(best_cr_row['calls']):,})"
            )
        elif "day" not in df_f.columns:
            st.info("Không có cột 'day' để vẽ Daily Contact Volume vs Conversion.")
        else:
            st.info("Không có cột 'y' (0/1) hợp lệ để tính conversion theo ngày.")

    # Tri thức tự động: best/worst month
    if month_col is not None and "y" in df_f.columns and df_f["y"].notna().any():
        g = (
            df_f.groupby(month_col)
            .agg(n=("y", "size"), conversion_rate=("y", "mean"))
            .reset_index()
            .sort_values("conversion_rate", ascending=False)
        )

        if len(g) >= 2:
            best = g.iloc[0]
            worst = g.iloc[-1]
            st.write(
                f"- Tháng hiệu quả nhất: **{best[month_col]}** (CR ~ **{best['conversion_rate']*100:.2f}%**, n={int(best['n']):,})"
            )
            st.write(
                f"- Tháng kém nhất: **{worst[month_col]}** (CR ~ **{worst['conversion_rate']*100:.2f}%**, n={int(worst['n']):,})"
            )
        else:
            st.info("Không đủ tháng khác nhau để so sánh best/worst.")

# -----------------------------
# Time analysis
# -----------------------------
with tab_time:
    st.subheader("Phân tích theo thời gian triển khai")

    # Chuẩn hoá trục thời gian
    if "date" in df_f.columns:
        tmp = df_f.copy()
        tmp["year_month"] = tmp["date"].dt.to_period("M").astype(str)
        tmp["day_of_week"] = tmp["date"].dt.day_name()
        month_col = "year_month"
        dow_col = "day_of_week"
    elif "month" in df_f.columns:
        tmp = df_f.copy()
        month_col = "month"
        dow_col = "day_of_week" if "day_of_week" in tmp.columns else None
    else:
        st.info("Không có cột 'date' hoặc 'month' nên không thể phân tích theo thời gian.")
        st.stop()

    # --- Heatmap: day_of_week × month ---
    if dow_col is not None and "y" in tmp.columns and tmp["y"].notna().any():
        pivot = (
            tmp.pivot_table(
                index=dow_col,
                columns=month_col,
                values="y",
                aggfunc="mean",
            )
            .fillna(0)
        )

        fig = px.imshow(
            pivot,
            aspect="auto",
            title=f"Heatmap conversion rate: {dow_col} × {month_col}",
        )
        fig.update_coloraxes(colorbar_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("Không đủ 'day_of_week' hoặc 'y' để vẽ heatmap conversion rate.")

    # --- Line charts: CR theo tháng & Volume theo tháng ---

    if "y" in tmp.columns and tmp["y"].notna().any():
        g = (
            tmp.groupby(month_col)
            .agg(
                calls=("y", "size"),
                conversion_rate=("y", "mean"),
            )
            .reset_index()
            .sort_values(month_col)
        )

        fig = go.Figure()

        fig.add_bar(
            x=g[month_col],
            y=g["calls"],
            name="Số cuộc gọi",
        )

        fig.add_scatter(
            x=g[month_col],
            y=g["conversion_rate"],
            name="Conversion rate",
            mode="lines+markers",
            yaxis="y2",
        )

        fig.update_layout(
            autosize=True,
            height=500,
            title="Số cuộc gọi & Conversion rate theo tháng",
            xaxis_title="Tháng",
            yaxis=dict(title="Số cuộc gọi"),
            yaxis2=dict(
                title="Conversion rate",
                overlaying="y",
                side="right",
                tickformat=".0%",
            ),
            margin=dict(l=40, r=40, t=60, b=40),
        )

        st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Campaign pressure & history
# -----------------------------
with tab_pressure:
    st.subheader("Tần suất liên hệ & lịch sử liên hệ")

    c1, c2 = st.columns(2)

    # --- Conversion rate theo campaign ---
    with c1:
        if has(df_f, "campaign") and has(df_f, "y") and df_f["y"].notna().any():
            g = (
                df_f.groupby("campaign")
                .agg(n=("y", "size"), conversion_rate=("y", "mean"))
                .reset_index()
                .sort_values("campaign")
            )

            # giới hạn trục x cho dễ đọc
            if len(g) > 30:
                g = g[g["campaign"] <= 20]

            fig = px.bar(
                g,
                x="campaign",
                y="conversion_rate",
                hover_data=["n"],
                title="Conversion rate theo campaign",
            )
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Thiếu cột 'campaign' hoặc 'y' để phân tích.")

    # --- Conversion rate theo pdays (binned) ---
    with c2:
        if has(df_f, "pdays") and has(df_f, "y") and df_f["y"].notna().any():
            tmp = df_f.copy()

            # dùng pdays_bin nếu đã có, nếu chưa thì tạo
            if "pdays_bin" not in tmp.columns:
                tmp["pdays_bin"] = bin_pdays(tmp["pdays"])

            g = (
                tmp.groupby("pdays_bin")
                .agg(n=("y", "size"), conversion_rate=("y", "mean"))
                .reset_index()
            )

            fig = px.bar(
                g,
                x="pdays_bin",
                y="conversion_rate",
                hover_data=["n"],
                title="Conversion rate theo pdays (binned)",
            )
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Thiếu cột 'pdays' hoặc 'y' để phân tích.")

    # --- Conversion rate theo poutcome ---
    if has(df_f, "poutcome") and has(df_f, "y") and df_f["y"].notna().any():
        g = (
            df_f.groupby("poutcome")
            .agg(n=("y", "size"), conversion_rate=("y", "mean"))
            .reset_index()
            .sort_values("conversion_rate", ascending=False)
        )

        fig = px.bar(
            g,
            x="poutcome",
            y="conversion_rate",
            hover_data=["n"],
            title="Conversion rate theo poutcome",
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# Channel analysis
# -----------------------------
with tab_channel:
    st.subheader("Kênh liên hệ & chất lượng danh sách")

    if not has(df_f, "contact"):
        st.info("Dataset không có cột 'contact'.")
    else:
        # chuẩn hoá contact để tránh lệch do hoa/thường/khoảng trắng
        tmp = df_f.copy()
        tmp["contact_clean"] = tmp["contact"].astype(str).str.strip().str.lower()

        # --- Conversion rate theo contact ---
        if has(tmp, "y") and tmp["y"].notna().any():
            g = (
                tmp.groupby("contact_clean")
                .agg(
                    n=("y", "size"),
                    conversion_rate=("y", "mean")
                )
                .reset_index()
                .sort_values("conversion_rate", ascending=False)
            )

            fig = px.bar(
                g,
                x="contact_clean",
                y="conversion_rate",
                hover_data=["n"],
                title="Conversion rate theo contact",
            )
            fig.update_yaxes(tickformat=".0%")
            fig.update_xaxes(title="contact")
            st.plotly_chart(fig, use_container_width=True)

        # --- Nếu không có y hợp lệ thì chỉ hiển thị số lượng ---
        else:
            g = (
                tmp.groupby("contact_clean")
                .size()
                .reset_index(name="n")
            )
            fig = px.bar(
                g,
                x="contact_clean",
                y="n",
                title="Số cuộc gọi theo contact",
            )
            fig.update_xaxes(title="contact")
            st.plotly_chart(fig, use_container_width=True)

        # --- KPI chất lượng danh sách: contact = unknown ---
        unk_mask = tmp["contact_clean"] == "unknown"
        unk_rate = unk_mask.mean()
        unk_n = int(unk_mask.sum())

        st.markdown("**Ghi chú chất lượng danh sách:**")
        st.write(
            f"- Tỷ trọng `contact = unknown`: **{unk_rate*100:.2f}%** "
            f"(n = {unk_n:,})"
        )

        if unk_rate > 0.2:
            st.warning(
                "Tỷ lệ contact=unknown khá cao → chất lượng danh sách/kênh liên hệ cần được cải thiện."
            )

# -----------------------------
# Duration (after-call only)
# -----------------------------
with tab_duration:
    st.subheader("Duration (After-call insight)")

    if not include_duration:
        st.info("Đang tắt 'Include duration'. Bật ở sidebar để xem phân tích duration (chỉ biết sau cuộc gọi).")
    elif not has(df_f, "duration"):
        st.info("Dataset không có cột 'duration'.")
    elif not has(df_f, "y") or df_f["y"].isna().all():
        st.info("Không có nhãn 'y' (0/1) để so sánh duration theo kết quả.")
    else:
        d = df_f.copy()
        d["duration"] = pd.to_numeric(d["duration"], errors="coerce")
        d = d.dropna(subset=["duration", "y"])

        if d.empty:
            st.info("Không có dữ liệu duration/y hợp lệ sau khi lọc.")
        else:
            # (tuỳ chọn) gắn nhãn để đọc dễ hơn
            d["y_label"] = d["y"].map({0: "Not subscribed (0)", 1: "Subscribed (1)"}).astype(str)

            c1, c2 = st.columns(2)

            # --- Distribution by outcome ---
            with c1:
                fig = px.histogram(
                    d,
                    x="duration",
                    color="y_label",
                    barmode="overlay",
                    title="Phân phối duration (seconds) theo kết quả",
                )
                st.plotly_chart(fig, use_container_width=True)

            # --- Conversion rate by duration bins ---
            with c2:
                d["duration_bin"] = bin_duration_seconds(d["duration"])  # dùng hàm bin của bạn
                g = (
                    d.groupby("duration_bin")
                    .agg(n=("y", "size"), conversion_rate=("y", "mean"))
                    .reset_index()
                )

                fig = px.bar(
                    g,
                    x="duration_bin",
                    y="conversion_rate",
                    hover_data=["n"],
                    title="Conversion rate theo nhóm duration",
                )
                fig.update_yaxes(tickformat=".0%")
                st.plotly_chart(fig, use_container_width=True)

            # --- Threshold insight (~510s) ---
            thr = 510.5
            left = d[d["duration"] <= thr]
            right = d[d["duration"] > thr]

            cr_left = left["y"].mean() if not left.empty else float("nan")
            cr_right = right["y"].mean() if not right.empty else float("nan")

            st.markdown("**Ngưỡng tham chiếu (~510.5s ≈ 8.5 phút):**")
            st.write(
                f"- ≤ 510.5s: CR ~ **{cr_left*100:.2f}%** (n={len(left):,})\n"
                f"- > 510.5s: CR ~ **{cr_right*100:.2f}%** (n={len(right):,})"
            )
            st.caption(
                "Lưu ý: duration là biến hậu nghiệm (biết sau cuộc gọi) → dùng để đánh giá chất lượng tương tác, "
                "không dùng để chọn khách hàng trước khi gọi."
            )


# -----------------------------
# Knowledge & Action Rules
# -----------------------------
with tab_rules:
    st.subheader("Tri thức rút ra & Khuyến nghị hành động (Actionable rules)")

    # kiểm tra y
    if (not has(df_f, "y")) or df_f["y"].isna().all():
        st.info("Không có nhãn 'y' (0/1) hợp lệ để rút tri thức.")
        st.stop()

    st.markdown("### Tri thức (auto-summarized từ dữ liệu đã lọc)")
    bullets = []

    # --- Thời gian: dùng YYYY-MM nếu có date ---
    month_col = None
    if has(df_f, "date"):
        tmp_time = df_f.copy()
        tmp_time["year_month"] = tmp_time["date"].dt.to_period("M").astype(str)
        month_col = "year_month"
    elif has(df_f, "month"):
        tmp_time = df_f
        month_col = "month"
    else:
        tmp_time = df_f

    if month_col is not None:
        g = (
            tmp_time.groupby(month_col)
            .agg(n=("y", "size"), conversion_rate=("y", "mean"))
            .reset_index()
            .sort_values("conversion_rate", ascending=False)
        )
        if len(g) >= 2:
            best, worst = g.iloc[0], g.iloc[-1]
            bullets.append(
                f"**Thời điểm hiệu quả nhất/kém nhất**: {best[month_col]} (CR {best['conversion_rate']*100:.2f}%, n={int(best['n']):,}) "
                f"vs {worst[month_col]} (CR {worst['conversion_rate']*100:.2f}%, n={int(worst['n']):,})."
            )

    # --- Kênh liên hệ (nếu có) ---
    if has(df_f, "contact"):
        tmp = df_f.copy()
        tmp["contact_clean"] = tmp["contact"].astype(str).str.strip().str.lower()

        g = (
            tmp.groupby("contact_clean")
            .agg(n=("y", "size"), conversion_rate=("y", "mean"))
            .reset_index()
            .sort_values("conversion_rate", ascending=False)
        )
        if not g.empty:
            top = g.iloc[0]
            bullets.append(
                f"**Kênh liên hệ hiệu quả nhất**: {top['contact_clean']} (CR {top['conversion_rate']*100:.2f}%, n={int(top['n']):,})."
            )
            unk_share = (tmp["contact_clean"] == "unknown").mean()
            bullets.append(f"**Tỷ trọng contact=unknown**: {unk_share*100:.2f}% → tín hiệu chất lượng danh sách thấp.")

    # --- Lịch sử chiến dịch: poutcome ---
    if has(df_f, "poutcome"):
        tmp = df_f.copy()
        tmp["poutcome_clean"] = tmp["poutcome"].astype(str).str.strip().str.lower()

        g = (
            tmp.groupby("poutcome_clean")
            .agg(n=("y", "size"), conversion_rate=("y", "mean"))
            .reset_index()
            .sort_values("conversion_rate", ascending=False)
        )
        if not g.empty:
            bullets.append(
                f"**Lịch sử chiến dịch (poutcome)**: nhóm cao nhất là {g.iloc[0]['poutcome_clean']} "
                f"(CR {g.iloc[0]['conversion_rate']*100:.2f}%, n={int(g.iloc[0]['n']):,})."
            )

    # --- Tần suất liên hệ (campaign): tìm best trong [1..5] ---
    if has(df_f, "campaign"):
        g = (
            df_f.groupby("campaign")
            .agg(n=("y", "size"), conversion_rate=("y", "mean"))
            .reset_index()
            .sort_values("campaign")
        )
        gg = g[g["campaign"].between(1, 5)]
        if not gg.empty:
            best_c = gg.sort_values("conversion_rate", ascending=False).iloc[0]
            bullets.append(
                f"**Tần suất liên hệ (campaign)**: tốt nhất trong [1..5] là campaign={int(best_c['campaign'])} "
                f"(CR {best_c['conversion_rate']*100:.2f}%, n={int(best_c['n']):,})."
            )

    # --- pdays (timing): dùng pdays_bin nếu có ---
    if has(df_f, "pdays"):
        tmp = df_f.copy()
        if "pdays_bin" not in tmp.columns:
            tmp["pdays_bin"] = bin_pdays(tmp["pdays"])
        g = (
            tmp.groupby("pdays_bin")
            .agg(n=("y", "size"), conversion_rate=("y", "mean"))
            .reset_index()
            .sort_values("conversion_rate", ascending=False)
        )
        if not g.empty:
            bullets.append(
                f"**Thời điểm gọi lại (pdays)**: nhóm cao nhất là {g.iloc[0]['pdays_bin']} "
                f"(CR {g.iloc[0]['conversion_rate']*100:.2f}%, n={int(g.iloc[0]['n']):,})."
            )

    # --- Duration insight (after-call) ---
    if include_duration and has(df_f, "duration"):
        d = df_f.copy()
        d["duration"] = pd.to_numeric(d["duration"], errors="coerce")
        d = d.dropna(subset=["duration", "y"])
        if not d.empty:
            thr = 510.5
            cr_hi = d[d["duration"] > thr]["y"].mean()
            cr_lo = d[d["duration"] <= thr]["y"].mean()
            bullets.append(
                f"**Ngưỡng duration (~8.5 phút)**: CR(>510.5s) = {cr_hi*100:.2f}% "
                f"vs CR(≤510.5s) = {cr_lo*100:.2f}%."
            )

    if bullets:
        for b in bullets:
            st.write(f"- {b}")
    else:
        st.info("Chưa đủ cột/nhãn để tự tóm tắt tri thức. Hãy kiểm tra dataset hoặc bật/tắt các filter.")

    st.markdown("### Luật hành động (IF–THEN) gợi ý")
    st.caption("Các luật dưới đây là mẫu *actionable* cho phần chiến dịch. Bạn có thể điều chỉnh ngưỡng theo biểu đồ ở các tab trên.")

    rules = []

    if has(df_f, "poutcome"):
        rules.append("**IF** `poutcome = success` **THEN** ưu tiên gọi lại (nhóm có xác suất chuyển đổi cao).")

    if has(df_f, "contact"):
        rules.append("**IF** `contact = unknown` **THEN** không ưu tiên (cần làm sạch danh sách/kênh liên hệ trước).")

    if has(df_f, "campaign"):
        rules.append("**IF** `campaign > 2` **THEN** cân nhắc dừng/đổi chiến lược (tránh gọi dồn gây giảm hiệu quả).")

    if has(df_f, "pdays"):
        rules.append("**IF** `pdays = -1` **THEN** coi như khách chưa từng liên hệ → dùng kịch bản 'khách mới'.")
        rules.append("**IF** `pdays` thuộc nhóm conversion cao (xem biểu đồ) **THEN** ưu tiên lịch gọi lại trong khoảng đó.")

    if month_col is not None:
        rules.append("**IF** tháng thuộc nhóm conversion cao **THEN** tăng nguồn lực/ưu tiên lead; **IF** tháng thấp **THEN** điều chỉnh thông điệp hoặc giảm volume.")

    if include_duration and has(df_f, "duration"):
        rules.append(
            "**(After-call)** **IF** `duration > 510.5s` **THEN** đánh dấu lead 'nóng' và ưu tiên follow-up/upsell; "
            "**IF** `duration ≤ 510.5s` **THEN** tối ưu kịch bản mở đầu hoặc sàng lọc lead tốt hơn."
        )

    for r in rules:
        st.write(f"- {r}")


    # st.markdown("### Ghi chú học thuật (để đưa vào đồ án)")
    # st.write(
    #     "- Dashboard chia 2 chế độ: **Pre-call (actionable)** và **After-call (insight)** để tránh *data leakage* của biến `duration`.\n"
    #     "- Các tri thức và luật hành động được rút ra từ: xu hướng theo thời gian, kênh liên hệ, tần suất liên hệ, và lịch sử chiến dịch."
    # )
