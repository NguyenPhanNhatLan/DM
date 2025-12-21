import streamlit as st
import plotly.express as px
from services.data import load_clean_data


st.header("📌 Overview")
df= load_clean_data()

n = len(df)
rate = (df["y"] == "yes").mean() if "y" in df else 0.0

c1, c2, c3 = st.columns(3)
c1.metric("Records", f"{n:,}")
c2.metric("Subscribe rate (y=yes)", f"{rate*100:.2f}%")
c3.metric("Class imbalance (no/yes)", f"{((df['y']=='no').sum() / max((df['y']=='yes').sum(),1)):.2f}x")

if "month" in df.columns:
    grp = df.groupby("month")["y"].apply(lambda s: (s=="yes").mean()).reset_index(name="yes_rate")
    fig = px.bar(grp, x="month", y="yes_rate", title="Subscribe rate by month")
    st.plotly_chart(fig, use_container_width=True)

st.caption("Dataset: direct marketing phone campaigns; goal: predict term deposit subscription (y).")
