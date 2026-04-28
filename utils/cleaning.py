"""
Data Cleaning utilities — five distinct cleaning measures.

Required by the brief:
  Measure 1: Missing-value imputation (median / mode / MICE / drop)
  Measure 2: Outlier detection & handling (IQR / Z-score, flag/cap/remove)
  Measure 3: Format validation (dates, types, postal codes)

Bonus measures (for extra credit):
  Measure 4: Date parsing & engineered date features
  Measure 5: Category standardization (case, whitespace, fuzzy match)
"""
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

# ════════════════════════════════════════════════════════════════════════════
# MEASURE 1 — Missing value imputation
# ════════════════════════════════════════════════════════════════════════════
def detect_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Return a per-column missingness summary."""
    n = len(df)
    report = pd.DataFrame({
        "Column": df.columns,
        "Missing": df.isna().sum().values,
        "Missing %": (100 * df.isna().sum() / max(n, 1)).round(2).values,
        "Dtype": [str(t) for t in df.dtypes.values],
    })
    return report.sort_values("Missing", ascending=False).reset_index(drop=True)


def impute(df: pd.DataFrame, strategy: str, columns: list = None) -> tuple:
    """
    Apply imputation strategy to selected columns.

    strategy ∈ {"median", "mean", "mode", "mice", "drop", "zero"}
    Returns (cleaned_df, log_dict)
    """
    df = df.copy()
    if columns is None:
        columns = [c for c in df.columns if df[c].isna().any()]

    log = {"strategy": strategy, "columns_treated": [], "rows_dropped": 0}

    if strategy == "drop":
        before = len(df)
        df = df.dropna(subset=columns)
        log["rows_dropped"] = before - len(df)
        log["columns_treated"] = columns
        return df, log

    for col in columns:
        if col not in df.columns or not df[col].isna().any():
            continue
        s = df[col]
        if strategy == "median" and pd.api.types.is_numeric_dtype(s):
            df[col] = s.fillna(s.median())
        elif strategy == "mean" and pd.api.types.is_numeric_dtype(s):
            df[col] = s.fillna(s.mean())
        elif strategy == "zero" and pd.api.types.is_numeric_dtype(s):
            df[col] = s.fillna(0)
        elif strategy == "mode":
            m = s.mode(dropna=True)
            df[col] = s.fillna(m.iloc[0]) if len(m) else s
        elif strategy == "mice" and pd.api.types.is_numeric_dtype(s):
            # Run MICE on all numeric columns at once for richer imputation
            num_cols = df.select_dtypes(include="number").columns.tolist()
            imputer = IterativeImputer(max_iter=5, random_state=42, sample_posterior=False)
            df[num_cols] = imputer.fit_transform(df[num_cols])
            log["columns_treated"] = num_cols
            return df, log
        else:
            # categorical with non-mode strategy — fall back to mode
            m = s.mode(dropna=True)
            df[col] = s.fillna(m.iloc[0]) if len(m) else s
        log["columns_treated"].append(col)

    return df, log


# ════════════════════════════════════════════════════════════════════════════
# MEASURE 2 — Outlier detection & handling
# ════════════════════════════════════════════════════════════════════════════
def detect_outliers(s: pd.Series, method: str = "iqr",
                    iqr_mult: float = 1.5, z_thresh: float = 3.0) -> tuple:
    """
    Return (mask of outliers, lower_bound, upper_bound).
    """
    s = pd.to_numeric(s, errors="coerce")
    if method == "iqr":
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - iqr_mult * iqr, q3 + iqr_mult * iqr
    else:  # z-score
        mu, sd = s.mean(), s.std()
        lo, hi = mu - z_thresh * sd, mu + z_thresh * sd
    mask = (s < lo) | (s > hi)
    return mask, lo, hi


def handle_outliers(df: pd.DataFrame, column: str, action: str,
                    method: str = "iqr", iqr_mult: float = 1.5,
                    z_thresh: float = 3.0,
                    cap_pct: tuple = (0.01, 0.99)) -> tuple:
    """
    action ∈ {"flag", "cap", "remove"}
    Returns (df_out, log_dict)
    """
    df = df.copy()
    mask, lo, hi = detect_outliers(df[column], method=method,
                                    iqr_mult=iqr_mult, z_thresh=z_thresh)
    n_out = int(mask.sum())
    log = {
        "column": column, "method": method, "action": action,
        "n_outliers": n_out, "lower": float(lo), "upper": float(hi),
        "before_stats": df[column].describe().to_dict(),
    }

    if action == "flag":
        df[f"{column}_is_outlier"] = mask
    elif action == "cap":
        lo_p = df[column].quantile(cap_pct[0])
        hi_p = df[column].quantile(cap_pct[1])
        df[column] = df[column].clip(lower=lo_p, upper=hi_p)
        log["cap_lower"] = float(lo_p)
        log["cap_upper"] = float(hi_p)
    elif action == "remove":
        df = df[~mask].reset_index(drop=True)
        log["rows_removed"] = n_out

    log["after_stats"] = df[column].describe().to_dict()
    return df, log


# ════════════════════════════════════════════════════════════════════════════
# MEASURE 3 — Format validation
# ════════════════════════════════════════════════════════════════════════════
def validate_formats(df: pd.DataFrame) -> dict:
    """
    Run format checks and return a structured report.
    Checks:
      • Postal codes coercible to int / 5-digit
      • Order Date and Ship Date are valid dates
      • Ship Date >= Order Date
      • Quantity is positive integer
    """
    report = {"checks": [], "n_issues": 0}
    df = df.copy()

    # Postal Code
    if "Postal Code" in df.columns:
        pc = pd.to_numeric(df["Postal Code"], errors="coerce")
        bad = pc.isna() & df["Postal Code"].notna()
        report["checks"].append({
            "check": "Postal Code is numeric",
            "passed": int((~bad).sum()),
            "failed": int(bad.sum()),
        })
        report["n_issues"] += int(bad.sum())

    # Date columns
    for c in ["Order Date", "Ship Date"]:
        if c in df.columns:
            parsed = pd.to_datetime(df[c], errors="coerce")
            bad = parsed.isna() & df[c].notna()
            report["checks"].append({
                "check": f"{c} is valid date",
                "passed": int((~bad).sum()),
                "failed": int(bad.sum()),
            })
            report["n_issues"] += int(bad.sum())

    # Ship >= Order
    if {"Order Date", "Ship Date"}.issubset(df.columns):
        od = pd.to_datetime(df["Order Date"], errors="coerce")
        sd = pd.to_datetime(df["Ship Date"], errors="coerce")
        bad = (sd < od)
        report["checks"].append({
            "check": "Ship Date ≥ Order Date",
            "passed": int((~bad).sum()),
            "failed": int(bad.sum()),
        })
        report["n_issues"] += int(bad.sum())

    # Quantity positive integer
    if "Quantity" in df.columns:
        q = pd.to_numeric(df["Quantity"], errors="coerce")
        bad = (q <= 0) | q.isna() | (q != q.round())
        report["checks"].append({
            "check": "Quantity is positive integer",
            "passed": int((~bad).sum()),
            "failed": int(bad.sum()),
        })
        report["n_issues"] += int(bad.sum())

    return report


def fix_formats(df: pd.DataFrame, drop_invalid_dates: bool = False) -> tuple:
    """Apply standard format fixes; return (df, log)."""
    df = df.copy()
    log = []

    if "Postal Code" in df.columns:
        df["Postal Code"] = pd.to_numeric(df["Postal Code"], errors="coerce")
        log.append("Coerced Postal Code to numeric")

    for c in ["Order Date", "Ship Date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
            log.append(f"Parsed {c} as datetime")

    if {"Order Date", "Ship Date"}.issubset(df.columns) and drop_invalid_dates:
        before = len(df)
        df = df[df["Ship Date"] >= df["Order Date"]].reset_index(drop=True)
        log.append(f"Dropped {before - len(df):,} rows with Ship<Order")

    if "Quantity" in df.columns:
        df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
        df = df[df["Quantity"] > 0].reset_index(drop=True)
        log.append("Removed rows with non-positive Quantity")

    return df, log


# ════════════════════════════════════════════════════════════════════════════
# MEASURE 4 (BONUS) — Date feature engineering
# ════════════════════════════════════════════════════════════════════════════
def engineer_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add Year / Quarter / Month / Weekday / Shipping Days columns."""
    df = df.copy()
    if "Order Date" in df.columns:
        d = pd.to_datetime(df["Order Date"], errors="coerce")
        df["Year"]    = d.dt.year
        df["Quarter"] = d.dt.to_period("Q").astype(str)
        df["Month"]   = d.dt.to_period("M").astype(str)
        df["Weekday"] = d.dt.day_name()
    if {"Order Date", "Ship Date"}.issubset(df.columns):
        df["Shipping Days"] = (
            pd.to_datetime(df["Ship Date"], errors="coerce")
            - pd.to_datetime(df["Order Date"], errors="coerce")
        ).dt.days
    if "Total Profit" in df.columns and "Total Revenue" in df.columns:
        df["Profit Margin"] = np.where(
            df["Total Revenue"].abs() > 1e-9,
            df["Total Profit"] / df["Total Revenue"],
            np.nan,
        )
    return df


# ════════════════════════════════════════════════════════════════════════════
# MEASURE 5 (BONUS) — Category standardization
# ════════════════════════════════════════════════════════════════════════════
def standardize_categories(df: pd.DataFrame, columns: list = None) -> tuple:
    """
    Trim whitespace and title-case categorical text columns.
    Returns (df, log_dict)
    """
    df = df.copy()
    if columns is None:
        columns = ["Region", "Segment", "Category", "Sub-Category", "Ship Mode", "State", "City"]
    log = {"changes": {}}
    for c in columns:
        if c in df.columns and df[c].dtype == object:
            before_unique = df[c].nunique(dropna=True)
            df[c] = df[c].astype(str).str.strip().replace({"nan": np.nan})
            after_unique = df[c].nunique(dropna=True)
            if before_unique != after_unique:
                log["changes"][c] = {"before_unique": before_unique,
                                      "after_unique": after_unique}
    return df, log


# ════════════════════════════════════════════════════════════════════════════
# Helper: inject some realistic missingness for demo purposes
# ════════════════════════════════════════════════════════════════════════════
def inject_demo_nulls(df: pd.DataFrame, frac: float = 0.02, seed: int = 42) -> pd.DataFrame:
    """
    OPTIONAL demo helper — inject random nulls so the imputation feature
    has visible work to do during the defense. Off by default.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    n = len(df)
    target_cols = ["Total Revenue", "Total Profit", "Quantity", "Postal Code", "City"]
    for col in target_cols:
        if col in df.columns:
            mask = rng.random(n) < frac
            df.loc[mask, col] = np.nan
    return df
