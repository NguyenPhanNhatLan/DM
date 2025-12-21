from pathlib import Path
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"


DATA_FILE = DATA_DIR / "processed" / "bank_marketing_raw.csv"


@st.cache_data(ttl=3600)
def load_clean_data() -> pd.DataFrame:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"CSV not found at: {DATA_FILE}\n"
            f"PROJECT_ROOT={PROJECT_ROOT}\n"
            f"DATA_DIR={DATA_DIR}"
        )

    df = pd.read_csv(DATA_FILE)

    if "y" in df.columns:
        df["y"] = df["y"].astype(str).str.lower().str.strip()

    return df
