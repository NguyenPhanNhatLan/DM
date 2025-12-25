# services/segmentation_service.py
from __future__ import annotations

import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, average_precision_score


@dataclass(frozen=True)
class SegConfig:
    target_col: str = "target"
    segment_features: Tuple[str, ...] = (
        "age", "balance", "job", "marital", "education", "default", "housing", "loan"
    )
    test_size: float = 0.2
    seed: int = 42
    max_iter: int = 5000
    call_budget: float = 0.10       # 10% by default
    k: int = 8                      # default k-means



def precision_at_k(y_true, p, k_frac: float):
    y_true = pd.Series(y_true).reset_index(drop=True)
    p = np.asarray(p)
    n = len(p)
    k = max(1, int(n * k_frac))
    idx = np.argsort(-p)[:k]
    return float(y_true.iloc[idx].mean())

def lift_at_k(y_true, p, k_frac: float):
    base = float(pd.Series(y_true).mean())
    prec = precision_at_k(y_true, p, k_frac)
    return prec / base if base > 0 else np.nan

def top_share(series: pd.Series, top_n: int = 3):
    vc = series.value_counts(normalize=True, dropna=False).head(top_n)
    return {str(k): float(v) for k, v in vc.items()}



def build_preprocess(df: pd.DataFrame, features: List[str]) -> Tuple[ColumnTransformer, List[str], List[str]]:
    num_features = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    cat_features = [c for c in features if c not in num_features]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocess = ColumnTransformer([
        ("num", num_pipe, num_features),
        ("cat", cat_pipe, cat_features),
    ])
    return preprocess, num_features, cat_features


def fit_score_model(df: pd.DataFrame, cfg: SegConfig):
    # Validate columns
    need_cols = [cfg.target_col, *cfg.segment_features]
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    X = df[list(cfg.segment_features)].copy()
    y = df[cfg.target_col].astype(int).copy()

    preprocess, _, _ = build_preprocess(df, list(cfg.segment_features))

    pipe = Pipeline([
        ("prep", preprocess),
        ("clf", LogisticRegression(max_iter=cfg.max_iter, class_weight="balanced")),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg.test_size,
        random_state=cfg.seed,
        stratify=y,
    )

    pipe.fit(X_train, y_train)
    p_test = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, p_test)),
        "pr_auc": float(average_precision_score(y_test, p_test)),
        "base_conv": float(y.mean()),
        "top5_prec": float(precision_at_k(y_test, p_test, 0.05)),
        "top10_prec": float(precision_at_k(y_test, p_test, 0.10)),
        "top20_prec": float(precision_at_k(y_test, p_test, 0.20)),
        "top30_prec": float(precision_at_k(y_test, p_test, 0.30)),
        "top5_lift": float(lift_at_k(y_test, p_test, 0.05)),
        "top10_lift": float(lift_at_k(y_test, p_test, 0.10)),
        "top20_lift": float(lift_at_k(y_test, p_test, 0.20)),
        "top30_lift": float(lift_at_k(y_test, p_test, 0.30)),
    }

    df_scored = df.copy()
    df_scored["p_yes"] = pipe.predict_proba(df_scored[list(cfg.segment_features)])[:, 1]

    return pipe, df_scored, metrics


def select_candidates(df_scored: pd.DataFrame, cfg: SegConfig) -> pd.DataFrame:
    thr = df_scored["p_yes"].quantile(1 - cfg.call_budget)
    return df_scored[df_scored["p_yes"] >= thr].copy()


def cluster_candidates(candidates: pd.DataFrame, cfg: SegConfig,fitted_prep: ColumnTransformer, overall_conv: float):
    # Fit preprocess on candidates only
    X_seg = fitted_prep.transform(candidates[list(cfg.segment_features)])
    km = KMeans(n_clusters=cfg.k, random_state=cfg.seed, n_init="auto")
    candidates = candidates.copy()
    candidates["segment"] = km.fit_predict(X_seg)

    base_conv = float(candidates[cfg.target_col].mean())
    # segment summary
    seg_summary = candidates.groupby("segment").agg(
        n=(cfg.target_col, "size"),
        conversion=(cfg.target_col, "mean"),
        avg_score=("p_yes", "mean"),
    ).sort_values("conversion", ascending=False)

    # lift vs overall requires overall base conversion from df_scored, but we can pass it later;
    # here we compute lift vs candidates only
    seg_summary["lift_vs_candidates"] = seg_summary["conversion"] / base_conv
    seg_summary["lift_vs_overall"] = seg_summary["conversion"] / overall_conv
    seg_summary = seg_summary.sort_values("lift_vs_overall", ascending=False)
    return candidates, seg_summary


def build_profiles(candidates: pd.DataFrame, cfg: SegConfig, overall_conv: float) -> pd.DataFrame:
    rows = []
    for seg_id, sub in candidates.groupby("segment"):
        row = {
            "segment": int(seg_id),
            "n": int(len(sub)),
            "conversion": float(sub[cfg.target_col].mean()),
            "lift_vs_overall": float(sub[cfg.target_col].mean() / overall_conv) if overall_conv > 0 else np.nan,
            "avg_score": float(sub["p_yes"].mean()),
            "age_median": float(sub["age"].median()) if "age" in sub.columns else np.nan,
            "balance_median": float(sub["balance"].median()) if "balance" in sub.columns else np.nan,
        }
        for c in ["job", "education", "marital", "default", "housing", "loan"]:
            if c in sub.columns:
                row[f"{c}_top"] = top_share(sub[c], 3)
        rows.append(row)

    prof = pd.DataFrame(rows).sort_values("lift_vs_overall", ascending=False)
    return prof


def tiering_from_profiles(profiles: pd.DataFrame) -> Dict[int, str]:
    # Sort by lift_vs_overall; create 4 tiers similar logic bạn đã dùng
    prof = profiles.sort_values("lift_vs_overall", ascending=False).reset_index(drop=True)
    segs = prof["segment"].tolist()

    tier_map: Dict[int, str] = {}
    if len(segs) >= 8:
        tier1 = segs[:1]
        tier2 = segs[1:3]
        tier3 = segs[3:7]
        tier4 = segs[7:]
    else:
        # fallback
        cut1 = max(1, len(segs)//4)
        cut2 = max(cut1+1, len(segs)//2)
        cut3 = max(cut2+1, int(len(segs)*0.75))
        tier1, tier2, tier3, tier4 = segs[:cut1], segs[cut1:cut2], segs[cut2:cut3], segs[cut3:]

    for s in tier1: tier_map[int(s)] = "Tier 1 (High Priority)"
    for s in tier2: tier_map[int(s)] = "Tier 2 (High Potential)"
    for s in tier3: tier_map[int(s)] = "Tier 3 (Medium Priority)"
    for s in tier4: tier_map[int(s)] = "Tier 4 (Low Priority)"
    return tier_map


def run_segmentation_pipeline(df: pd.DataFrame, cfg: SegConfig):
    pipe, df_scored, metrics = fit_score_model(df, cfg)

    overall_conv = float(df_scored[cfg.target_col].mean())
    candidates = select_candidates(df_scored, cfg)
    cand_conv = float(candidates[cfg.target_col].mean())
    cand_lift = float(cand_conv / overall_conv) if overall_conv > 0 else np.nan

    candidates = select_candidates(df_scored, cfg)
    fitted_prep = pipe.named_steps["prep"]
    candidates, seg_summary = cluster_candidates(candidates, cfg, fitted_prep)
    profiles = build_profiles(candidates, cfg, overall_conv)

    tier_map = tiering_from_profiles(profiles)
    candidates["tier"] = candidates["segment"].map(lambda s: tier_map.get(int(s), "Tier ?"))
    profiles["tier"] = profiles["segment"].map(lambda s: tier_map.get(int(s), "Tier ?"))

    # optional campaign validation (if exists)
    campaign_table = None
    if "has_previous_campaign" in candidates.columns and "pdays_contacted" in candidates.columns:
        campaign_table = candidates.groupby("segment").agg(
            prev_campaign_rate=("has_previous_campaign", "mean"),
            pdays_contacted_median=("pdays_contacted", "median"),
            conversion=(cfg.target_col, "mean"),
            n=(cfg.target_col, "size"),
        ).sort_values("conversion", ascending=False)

    return {
        "pipe": pipe,
        "df_scored": df_scored,
        "metrics": metrics,
        "overall_conv": overall_conv,
        "candidates": candidates,
        "cand_conv": cand_conv,
        "cand_lift": cand_lift,
        "seg_summary": seg_summary,
        "profiles": profiles,
        "tier_map": tier_map,
        "campaign_table": campaign_table,
    }
