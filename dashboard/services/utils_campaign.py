import numpy as np
import pandas as pd
import streamlit as st
import datetime as dt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / ".." / ".." / "data" / "processed" / "bank_marketing_campaign.csv"

def has(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns

def _safe_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def bin_duration_seconds(d: pd.Series) -> pd.Categorical:
    # Bins tuned for call durations (seconds)
    bins = [-1, 60, 180, 300, 480, 510, 900, 1_000_000]
    labels = ["≤1m", "1–3m", "3–5m", "5–8m", "8–8.5m", "8.5–15m", ">15m"]
    return pd.cut(d, bins=bins, labels=labels)


def bin_pdays(p: pd.Series) -> pd.Categorical:
    """
    pdays: days since last contact from a previous campaign.
    -1 means client was not previously contacted.
    """
    p = _safe_num(p)
    out = pd.Series(index=p.index, dtype="object")

    never = (p.isna()) | (p == -1)
    out[never] = "Never contacted (-1)"

    mask = ~never
    out[mask] = pd.cut(
        p[mask],
        bins=[-0.5, 7, 30, 90, 180, 365, 10_000],
        labels=["0–7", "8–30", "31–90", "91–180", "181–365", ">365"],
    ).astype(str)

    cats = ["0–7", "8–30", "31–90", "91–180", "181–365", ">365", "Never contacted (-1)"]
    return pd.Categorical(out, categories=cats)

def conversion_rate(df: pd.DataFrame) -> float:
    if df.empty or "y" not in df.columns:
        return float("nan")
    return df["y"].mean()

def agg_rate_and_n(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if df.empty or col not in df.columns:
        return pd.DataFrame({col: [], "n": [], "conversion_rate": []})

    g = df.groupby(col).agg(
        n=("y", "size"),
        conversion_rate=("y", "mean")   
    )
    return g.reset_index()

def load_data(path) -> pd.DataFrame:
    path = Path(path)  

    suf = path.suffix.lower()
    if suf in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    elif suf == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {suf}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    df["year"] = df["date"].dt.year         
    df["month"] = df["date"].dt.month
    df["month_name"] = df["date"].dt.month_name()
    df["day_of_week"] = df["date"].dt.day_name()
    # y is already 0/1
    df["y"] = pd.to_numeric(df["y"], errors="coerce")

    # numeric campaign variables
    for c in ["campaign", "pdays", "previous"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # string
    if "poutcome" in df.columns:
        df["poutcome"] = df["poutcome"].astype(str)

    # pdays bin
    if "pdays" in df.columns:
        df["pdays_bin"] = bin_pdays(df["pdays"])

    return df
