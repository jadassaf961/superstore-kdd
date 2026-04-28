"""
Superstore schema + validators.
Both Mode A (file upload) and Mode B (Google Sheets) feed through validate_dataframe()
so the downstream pipeline only ever sees data that conforms to the contract below.
"""
import pandas as pd
import numpy as np

# Canonical Superstore columns (Orders sheet) ────────────────────────────────
REQUIRED_COLUMNS = {
    "Row ID":          "int",
    "Order ID":        "str",
    "Order Date":      "datetime",
    "Ship Date":       "datetime",
    "Ship Mode":       "str",
    "Customer ID":     "str",
    "Customer Name":   "str",
    "Segment":         "str",
    "Country/Region":  "str",
    "City":            "str",
    "State":           "str",
    "Postal Code":     "numeric",
    "Region":          "str",
    "Product ID":      "str",
    "Category":        "str",
    "Sub-Category":    "str",
    "Product Name":    "str",
    "Quantity":        "numeric",
    "Total Revenue":   "numeric",
    "Total Profit":    "numeric",
}

# Common alternate column names mapped to canonical
COLUMN_ALIASES = {
    "Sales":        "Total Revenue",
    "Profit":       "Total Profit",
    "Country":      "Country/Region",
    "Customer":     "Customer Name",
    "Product":      "Product Name",
    "Subcategory":  "Sub-Category",
    "Sub Category": "Sub-Category",
}

EXPECTED_VALUES = {
    "Segment":  {"Consumer", "Corporate", "Home Office"},
    "Region":   {"Central", "East", "South", "West"},
    "Category": {"Furniture", "Office Supplies", "Technology"},
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and apply known aliases."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={k: v for k, v in COLUMN_ALIASES.items() if k in df.columns})
    return df


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Best-effort type coercion to the canonical schema."""
    df = df.copy()
    for col, kind in REQUIRED_COLUMNS.items():
        if col not in df.columns:
            continue
        if kind == "datetime":
            df[col] = pd.to_datetime(df[col], errors="coerce")
        elif kind == "numeric":
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif kind == "int":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        else:
            df[col] = df[col].astype(str).where(df[col].notna(), np.nan)
    return df


def validate_dataframe(df: pd.DataFrame) -> dict:
    """
    Return a validation report:
      ok               — bool, hard-pass
      missing_columns  — list of required cols not present
      type_issues      — list of (col, sample_problem)
      value_issues     — list of (col, unexpected_values)
      warnings         — soft issues (non-blocking)
      n_rows, n_cols
    """
    report = {
        "ok": True,
        "missing_columns": [],
        "type_issues": [],
        "value_issues": [],
        "warnings": [],
        "n_rows": len(df),
        "n_cols": len(df.columns),
    }

    # 1. Required columns
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        report["missing_columns"] = missing
        report["ok"] = False

    # 2. Type sanity (try coercion, report failure rate)
    coerced = coerce_types(df)
    for col, kind in REQUIRED_COLUMNS.items():
        if col not in coerced.columns:
            continue
        if kind in ("datetime", "numeric", "int"):
            n_null_after = coerced[col].isna().sum()
            n_null_before = df[col].isna().sum() if col in df.columns else 0
            new_nulls = n_null_after - n_null_before
            if new_nulls > 0:
                pct = 100 * new_nulls / max(len(df), 1)
                report["type_issues"].append(
                    f"`{col}`: {new_nulls:,} value(s) ({pct:.1f}%) could not be parsed as {kind}"
                )

    # 3. Expected categorical values (warning only — outliers happen)
    for col, allowed in EXPECTED_VALUES.items():
        if col in df.columns:
            unexpected = set(df[col].dropna().astype(str).unique()) - allowed
            if unexpected:
                report["value_issues"].append(
                    f"`{col}`: unexpected value(s) → {sorted(unexpected)[:5]}"
                )

    # 4. Date sanity
    if "Order Date" in coerced.columns and "Ship Date" in coerced.columns:
        bad = (coerced["Ship Date"] < coerced["Order Date"]).sum()
        if bad > 0:
            report["warnings"].append(f"{bad:,} order(s) have Ship Date before Order Date")

    # 5. Negative revenue is suspicious
    if "Total Revenue" in coerced.columns:
        neg = (coerced["Total Revenue"] < 0).sum()
        if neg > 0:
            report["warnings"].append(f"{neg:,} row(s) have negative revenue")

    return report


def schema_summary_table() -> pd.DataFrame:
    """Used in the About / Help panels — shows the contract."""
    return pd.DataFrame(
        [{"Column": c, "Type": t} for c, t in REQUIRED_COLUMNS.items()]
    )
