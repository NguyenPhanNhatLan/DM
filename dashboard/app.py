import streamlit as st
from services.data import load_clean_data

st.set_page_config(layout="wide")
st.title("📊 Bank Marketing Knowledge Dashboard - overview 200 records")

df = load_clean_data()

st.dataframe(df.head(200), use_container_width=True, width='stretch')
