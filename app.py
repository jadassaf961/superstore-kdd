"""
Superstore KDD Analytics — DAN614 Individual Project
================================================================================
Author : Jad Assaf
Course : DAN614 — Advanced Data Visualization
School : LAU · Adnan Kassar School of Business · MSDA

This single Streamlit application implements the full Knowledge Discovery
in Databases (KDD) pipeline end-to-end:

    Selection → Pre-processing → Transformation → Data Mining → Interpretation

Architecture:
    • Horizontal navigation menu (streamlit-option-menu)
    • Six pages: Data Source, Cleaning, three Dashboards, ML Lab, About
    • All state lives in st.session_state, keyed by stage
    • Visual design follows Cole Knaflic / SWD principles:
        - One accent color (LAU green) used sparingly
        - Decluttered axes, no chart borders, no secondary y-axes
        - Direct labeling over legend boxes
        - Scorecards for headline numbers, charts for shape
================================================================================
"""
import warnings
warnings.filterwarnings("ignore")

import io
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu

# Local utilities
from utils.theme import (
    apply_plotly_template, inject_css, render_brand_strip,
    kpi_card, insight, page_title, section,
    LAU_GREEN, LAU_GREEN_DK, LAU_GREEN_LT, ACCENT_GOLD, ACCENT_RED,
    GRAY_900, GRAY_700, GRAY_500, GRAY_300, GRAY_100,
    SEQUENTIAL_GREEN, CATEGORICAL,
)
from utils.schema import (
    REQUIRED_COLUMNS, normalize_columns, coerce_types,
    validate_dataframe, schema_summary_table,
)
from utils.cleaning import (
    detect_missing, impute,
    detect_outliers, handle_outliers,
    validate_formats, fix_formats,
    engineer_date_features, standardize_categories,
    inject_demo_nulls,
)
from utils.gsheets import load_sheet, list_worksheets, parse_creds_upload
from utils.ml_models import (
    CLASSIFIERS,
    MULTICLASS_TARGETS, CLASSIFICATION_TARGETS,
    train_multiclass, train_classifier,
    build_rfm, kmeans_rfm, kmeans_elbow, feature_importance,
    predict_single, predict_single_proba,
)
from utils.forecasting import (
    aggregate_monthly, seasonal_naive_forecast, sarima_forecast, backtest,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page config + theme
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Superstore KDD · DAN614",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)
apply_plotly_template()
inject_css()

STUDENT_NAME = "Jad Assaf"
render_brand_strip(STUDENT_NAME)

# ─────────────────────────────────────────────────────────────────────────────
# Session state initialization
# ─────────────────────────────────────────────────────────────────────────────
DEFAULTS = {
    "df_raw":        None,    # original loaded data
    "df_clean":      None,    # after cleaning steps
    "df_active":     None,    # what dashboards & ML use
    "source_meta":   {},      # which mode + filename / sheet info
    "cleaning_log":  [],      # human-readable list of operations applied
    "valid_report":  None,
    "ml_results":    {},      # cache trained models per page session
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# Horizontal navigation menu
# ─────────────────────────────────────────────────────────────────────────────
PAGES = [
    "Data Source",
    "Data Cleaning",
    "Executive Overview",
    "Customer & Product",
    "Geo & Logistics",
    "ML Lab",
    "About",
]
ICONS = ["cloud-upload", "funnel", "speedometer2", "people", "globe", "cpu", "info-circle"]

selected = option_menu(
    menu_title=None,
    options=PAGES,
    icons=ICONS,
    orientation="horizontal",
    default_index=0,
    styles={
        "container": {"padding": "0", "background-color": "transparent",
                       "margin-bottom": "10px"},
        "nav-link": {
            "font-size": "13.5px", "font-weight": "500",
            "color": GRAY_700, "padding": "10px 14px",
            "margin": "0 2px", "border-radius": "8px",
            "background-color": "transparent",
        },
        "nav-link-selected": {
            "background-color": LAU_GREEN, "color": "white",
            "font-weight": "600",
        },
        "icon": {"font-size": "14px"},
    },
)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: gate that requires data to be loaded
# ─────────────────────────────────────────────────────────────────────────────
def require_data() -> pd.DataFrame:
    """Block downstream pages until data is loaded."""
    df = st.session_state.df_active
    if df is None or len(df) == 0:
        st.markdown(
            '<div class="warn-box">⚠️ No data loaded yet. Go to '
            '<strong>Data Source</strong> and upload a file or connect a Google Sheet '
            'to unlock this page.</div>',
            unsafe_allow_html=True,
        )
        st.stop()
    return df


def fmt_money(v: float) -> str:
    if pd.isna(v):
        return "—"
    if abs(v) >= 1_000_000:
        return f"${v/1_000_000:,.2f}M"
    if abs(v) >= 1_000:
        return f"${v/1_000:,.1f}K"
    return f"${v:,.2f}"


def fmt_int(v) -> str:
    if pd.isna(v):
        return "—"
    return f"{int(v):,}"


def fmt_pct(v: float, decimals: int = 1) -> str:
    if pd.isna(v):
        return "—"
    return f"{v*100:.{decimals}f}%"


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1 — DATA SOURCE
# ═══════════════════════════════════════════════════════════════════════════
def render_data_source():
    page_title("Data Source",
               "Choose how to load the Superstore dataset. "
               "Both modes feed identical, validated data into the downstream pipeline.")

    mode = st.radio(
        "Data source mode",
        ["📁 Mode A — Upload local file", "☁️ Mode B — Google Sheets API"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if mode.startswith("📁"):
        _render_mode_a()
    else:
        _render_mode_b()

    # Once data is loaded, show validation report + preview
    if st.session_state.df_raw is not None:
        st.markdown("---")
        _render_validation_and_preview()


def _render_mode_a():
    section("Mode A · Local file upload",
            "Accepts .xls, .xlsx, .csv. Validated against the Superstore schema before loading.")

    col1, col2 = st.columns([2, 1])
    with col1:
        f = st.file_uploader(
            "Upload Superstore data",
            type=["xls", "xlsx", "csv"],
            help="Expected columns include Order ID, Order Date, Total Revenue, "
                  "Total Profit, Region, etc.",
        )
    with col2:
        sheet_name = st.text_input(
            "Sheet name (Excel only)",
            value="Orders",
            help="If your Excel file has multiple sheets, specify which one. "
                  "Defaults to 'Orders' (matches the canonical Superstore file).",
        )
        demo_nulls = st.checkbox(
            "Inject demo missing values",
            value=False,
            help="Adds ~2% random nulls to numeric columns so the imputation "
                  "tools have something visible to clean. Useful for the demo.",
        )

    if f is None:
        st.markdown(
            '<div class="info-box">📂 Awaiting upload. The dataset has 20 columns and '
            'roughly 10,000 rows of US retail transactions from 2017–2020.</div>',
            unsafe_allow_html=True,
        )
        return

    try:
        if f.name.lower().endswith(".csv"):
            df = pd.read_csv(f)
        else:
            xl = pd.ExcelFile(f)
            sn = sheet_name if sheet_name in xl.sheet_names else xl.sheet_names[0]
            df = pd.read_excel(xl, sheet_name=sn)
    except Exception as e:
        st.error(f"❌ Could not read the file: {e}")
        return

    df = normalize_columns(df)
    df = coerce_types(df)
    if demo_nulls:
        df = inject_demo_nulls(df, frac=0.02)

    report = validate_dataframe(df)
    st.session_state.valid_report = report

    if not report["ok"]:
        st.error(
            f"❌ **Schema validation failed.** Missing required columns: "
            f"`{', '.join(report['missing_columns'])}`. "
            "Please check the column names in your file."
        )
        return

    st.session_state.df_raw     = df
    st.session_state.df_clean   = None
    st.session_state.df_active  = df
    st.session_state.cleaning_log = []
    st.session_state.source_meta = {
        "mode": "A",
        "filename": f.name,
        "loaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "rows": len(df),
        "cols": len(df.columns),
    }
    st.success(
        f"✅ **Loaded `{f.name}`** — {len(df):,} rows × {len(df.columns)} columns. "
        f"All 20 required columns present."
    )


def _render_mode_b():
    section("Mode B · Google Sheets via Service Account API",
            "Requires a Google Cloud service account with read access to the target sheet.")

    with st.expander("ℹ️ How to set up a service account", expanded=False):
        st.markdown("""
1. Go to **Google Cloud Console** → IAM → Service Accounts → **Create**.
2. Skip role assignment (read-only access is granted at the sheet level).
3. Click the new account → **Keys** → **Add key** → **Create new key (JSON)**.
        Download the JSON file.
4. Open your target Google Sheet and **share it with the service account's
        `client_email`** (read access is enough).
5. Upload the JSON below and paste the sheet URL.

> The credentials are held only in memory for this session — they are never
> written to disk by this app.
""")

    col1, col2 = st.columns([1, 1])
    with col1:
        creds_file = st.file_uploader(
            "Service account credentials (.json)",
            type=["json"],
            help="The JSON key file you downloaded from Google Cloud Console.",
        )
    with col2:
        sheet_url = st.text_input(
            "Google Sheet URL or ID",
            placeholder="https://docs.google.com/spreadsheets/d/...",
        )

    if creds_file is None or not sheet_url:
        st.markdown(
            '<div class="info-box">☁️ Provide credentials and a sheet URL to connect.</div>',
            unsafe_allow_html=True,
        )
        return

    try:
        creds_dict = parse_creds_upload(creds_file.getvalue())
    except ValueError as e:
        st.error(f"❌ {e}")
        return

    # Worksheet picker
    try:
        worksheets = list_worksheets(creds_dict, sheet_url)
    except Exception as e:
        st.error(f"❌ Could not connect to the sheet: {e}")
        st.markdown(
            '<div class="warn-box">Common causes: (1) The service account email '
            'has not been added to the sheet, (2) the URL/ID is wrong, '
            '(3) the Google Sheets API is not enabled in your project.</div>',
            unsafe_allow_html=True,
        )
        return

    chosen_ws = st.selectbox("Worksheet / tab", worksheets,
                              index=worksheets.index("Orders") if "Orders" in worksheets else 0)

    if st.button("🔗 Load worksheet"):
        try:
            df, meta = load_sheet(creds_dict, sheet_url, worksheet=chosen_ws)
        except Exception as e:
            st.error(f"❌ Failed to load worksheet: {e}")
            return

        df = normalize_columns(df)
        df = coerce_types(df)
        report = validate_dataframe(df)
        st.session_state.valid_report = report

        if not report["ok"]:
            st.error(
                f"❌ **Schema validation failed.** Missing columns: "
                f"`{', '.join(report['missing_columns'])}`."
            )
            return

        st.session_state.df_raw    = df
        st.session_state.df_clean  = None
        st.session_state.df_active = df
        st.session_state.cleaning_log = []
        st.session_state.source_meta = {"mode": "B", **meta}
        st.success(
            f"✅ Connected to **{meta['spreadsheet_title']}** → tab "
            f"**{meta['worksheet_title']}**. {len(df):,} rows loaded."
        )

    # Sidebar metadata box if a sheet is loaded via Mode B
    if st.session_state.source_meta.get("mode") == "B":
        meta = st.session_state.source_meta
        with st.sidebar:
            st.markdown("### ☁️ Sheet metadata")
            st.markdown(f"""
- **Title:** {meta.get('spreadsheet_title','—')}
- **Worksheet:** {meta.get('worksheet_title','—')}
- **Rows:** {meta.get('row_count', 0):,}
- **Columns:** {meta.get('col_count', 0)}
- **Worksheets in workbook:** {meta.get('worksheet_count', 0)}
- **Last updated:** {meta.get('last_updated','—')}
- **Loaded at:** {meta.get('loaded_at','—')}
""")


def _render_validation_and_preview():
    section("Validation report")
    rep = st.session_state.valid_report
    df = st.session_state.df_raw

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(kpi_card("Rows", f"{rep['n_rows']:,}", accent=True), unsafe_allow_html=True)
    with c2: st.markdown(kpi_card("Columns", f"{rep['n_cols']}"), unsafe_allow_html=True)
    with c3: st.markdown(kpi_card("Type issues", f"{len(rep['type_issues'])}"), unsafe_allow_html=True)
    with c4: st.markdown(kpi_card("Warnings", f"{len(rep['warnings'])}"), unsafe_allow_html=True)
    st.write("")

    if rep["type_issues"]:
        st.markdown("**Type-coercion issues**")
        for msg in rep["type_issues"]:
            st.markdown(f"- {msg}", unsafe_allow_html=True)
    if rep["warnings"]:
        st.markdown("**Warnings**")
        for msg in rep["warnings"]:
            st.markdown(f"- {msg}")
    if rep["value_issues"]:
        st.markdown("**Unexpected categorical values**")
        for msg in rep["value_issues"]:
            st.markdown(f"- {msg}")

    section("Data explorer",
            "Sort any column by clicking its header. Use the search box to filter rows. "
            "Resize columns by dragging the column dividers.")
    st.dataframe(df, use_container_width=True, height=500)

    if st.session_state.cleaning_log:
        section("Active cleaning operations")
        for op in st.session_state.cleaning_log:
            st.markdown(f"- {op}")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2 — DATA CLEANING
# ═══════════════════════════════════════════════════════════════════════════
def render_data_cleaning():
    page_title("Data Cleaning",
               "Five distinct cleaning measures. Each operation is logged and "
               "the cleaned dataset propagates downstream.")
    df = require_data()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "1 · Missing values",
        "2 · Outliers",
        "3 · Format validation",
        "4 · Date features (bonus)",
        "5 · Category standardization (bonus)",
    ])

    # ── Tab 1: Missing values ───────────────────────────────────────────────
    with tab1:
        section("Missing-value imputation",
                "Detect nulls and choose an imputation strategy per column or globally.")
        miss = detect_missing(df)
        miss_show = miss[miss["Missing"] > 0]

        c1, c2 = st.columns([1, 1])
        with c1:
            if len(miss_show) == 0:
                st.markdown('<div class="success-box">✅ No missing values detected '
                             'across any column.</div>', unsafe_allow_html=True)
            else:
                st.dataframe(miss_show, use_container_width=True, hide_index=True)

        with c2:
            strategy = st.selectbox(
                "Imputation strategy",
                ["median", "mean", "mode", "mice", "zero", "drop"],
                help="median/mean/zero — numeric only · mode — any · "
                      "MICE — multivariate iterative for numerics · drop — remove rows.",
            )
            target_cols = st.multiselect(
                "Columns to impute (empty = all with nulls)",
                options=miss_show["Column"].tolist(),
                default=miss_show["Column"].tolist(),
            )
            if st.button("Apply imputation", key="apply_imp"):
                cleaned, log = impute(df, strategy=strategy,
                                       columns=target_cols if target_cols else None)
                st.session_state.df_active = cleaned
                st.session_state.cleaning_log.append(
                    f"Imputation: `{strategy}` on {len(log['columns_treated'])} column(s)"
                    + (f", dropped {log['rows_dropped']} rows" if log["rows_dropped"] else "")
                )
                st.success(f"✅ Applied **{strategy}** to "
                            f"{len(log['columns_treated'])} column(s).")
                st.rerun()

    # ── Tab 2: Outliers ────────────────────────────────────────────────────
    with tab2:
        section("Outlier detection & handling",
                "Identify outliers via IQR or Z-score, then flag, cap, or remove them.")
        c1, c2, c3 = st.columns(3)
        with c1:
            target = st.selectbox("Column", ["Total Revenue", "Total Profit", "Quantity"],
                                   key="out_col")
        with c2:
            method = st.selectbox("Method", ["iqr", "zscore"], key="out_method")
        with c3:
            action = st.selectbox("Action", ["flag", "cap", "remove"], key="out_action")

        if method == "iqr":
            mult = st.slider("IQR multiplier", 1.0, 3.0, 1.5, 0.1, key="out_mult")
            z = 3.0
        else:
            z = st.slider("Z-score threshold", 2.0, 4.0, 3.0, 0.1, key="out_z")
            mult = 1.5

        # Detection preview
        mask, lo, hi = detect_outliers(df[target], method=method,
                                        iqr_mult=mult, z_thresh=z)
        n_out = int(mask.sum())
        c1, c2, c3 = st.columns(3)
        with c1: st.markdown(kpi_card("Outliers detected", f"{n_out:,}",
                                        sub=f"{100*n_out/max(len(df),1):.1f}% of rows",
                                        accent=True), unsafe_allow_html=True)
        with c2: st.markdown(kpi_card("Lower bound", fmt_money(lo)), unsafe_allow_html=True)
        with c3: st.markdown(kpi_card("Upper bound", fmt_money(hi)), unsafe_allow_html=True)
        st.write("")

        if st.button("Apply outlier handling", key="apply_out"):
            cleaned, log = handle_outliers(df, target, action,
                                            method=method, iqr_mult=mult, z_thresh=z)
            st.session_state.df_active = cleaned
            st.session_state.cleaning_log.append(
                f"Outliers: `{action}` {n_out} outlier(s) in `{target}` "
                f"using {method} (bounds {lo:.2f} – {hi:.2f})"
            )
            st.success(
                f"✅ Action **{action}** applied to {n_out:,} outlier(s) in `{target}`."
            )

    # ── Tab 3: Format validation ───────────────────────────────────────────
    with tab3:
        section("Format validation",
                "Verify dates, types, and value ranges. Apply automatic fixes where safe.")
        rep = validate_formats(df)

        n_pass = sum(c["passed"] for c in rep["checks"])
        n_fail = sum(c["failed"] for c in rep["checks"])
        c1, c2, c3 = st.columns(3)
        with c1: st.markdown(kpi_card("Checks run", f"{len(rep['checks'])}", accent=True),
                                unsafe_allow_html=True)
        with c2: st.markdown(kpi_card("Total passing", f"{n_pass:,}"), unsafe_allow_html=True)
        with c3: st.markdown(kpi_card("Total failing", f"{n_fail:,}"), unsafe_allow_html=True)
        st.write("")

        check_df = pd.DataFrame(rep["checks"])
        st.dataframe(check_df, use_container_width=True, hide_index=True)

        drop_bad_dates = st.checkbox("Also drop rows where Ship Date < Order Date", value=False)
        if st.button("Apply format fixes", key="apply_fmt"):
            cleaned, log = fix_formats(df, drop_invalid_dates=drop_bad_dates)
            st.session_state.df_active = cleaned
            st.session_state.cleaning_log.append("Format fixes: " + "; ".join(log))
            st.success("✅ Applied format fixes: " + "; ".join(log))

    # ── Tab 4: Date feature engineering ────────────────────────────────────
    with tab4:
        section("Date feature engineering (bonus)",
                "Adds Year, Quarter, Month, Weekday, Shipping Days, and Profit Margin "
                "columns derived from existing fields. Required for the dashboards.")
        if st.button("Engineer date features", key="apply_dates"):
            cleaned = engineer_date_features(df)
            st.session_state.df_active = cleaned
            new_cols = [c for c in cleaned.columns if c not in df.columns]
            st.session_state.cleaning_log.append(
                f"Date features: added {', '.join(new_cols)}"
            )
            st.success(f"✅ Added: {', '.join(new_cols)}")
            st.rerun()
        existing = [c for c in ["Year", "Quarter", "Month", "Weekday",
                                  "Shipping Days", "Profit Margin"] if c in df.columns]
        if existing:
            st.markdown(f'<div class="success-box">✅ Already engineered: '
                          f'{", ".join(existing)}</div>', unsafe_allow_html=True)

    # ── Tab 5: Category standardization ────────────────────────────────────
    with tab5:
        section("Category standardization (bonus)",
                "Trim whitespace and normalize text categorical values to prevent "
                "duplicate categories like 'West' vs ' West '.")
        if st.button("Standardize categorical text", key="apply_std"):
            cleaned, log = standardize_categories(df)
            st.session_state.df_active = cleaned
            if log["changes"]:
                msg = "; ".join(f"{k}: {v['before_unique']}→{v['after_unique']}"
                                  for k, v in log["changes"].items())
                st.session_state.cleaning_log.append(f"Category standardization: {msg}")
                st.success(f"✅ Cleaned categorical fields. Changes: {msg}")
            else:
                st.markdown('<div class="info-box">No category drift detected — '
                             'all categorical fields are already clean.</div>',
                             unsafe_allow_html=True)

    # ── Cleaning log ───────────────────────────────────────────────────────
    if st.session_state.cleaning_log:
        section("Operations applied this session")
        for i, op in enumerate(st.session_state.cleaning_log, 1):
            st.markdown(f"{i}. {op}")
        if st.button("🗑️ Reset to raw data"):
            st.session_state.df_active = st.session_state.df_raw
            st.session_state.cleaning_log = []
            st.success("Reset to raw data.")
            st.rerun()


def _dist_chart(s: pd.Series, label: str, tag: str, n_outliers: int = None,
                  lo: float = None, hi: float = None) -> go.Figure:
    """Compact histogram with optional outlier-bound markers."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=s, nbinsx=40,
        marker=dict(color=GRAY_300 if tag == "Before" else LAU_GREEN_LT,
                     line=dict(width=0)),
        hovertemplate="Range: %{x}<br>Count: %{y}<extra></extra>",
    ))
    if lo is not None and hi is not None and tag == "Before":
        fig.add_vline(x=lo, line=dict(color=ACCENT_RED, width=1, dash="dash"))
        fig.add_vline(x=hi, line=dict(color=ACCENT_RED, width=1, dash="dash"))
    fig.update_layout(
        height=240, margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False, bargap=0.05,
        xaxis_title=label, yaxis_title="",
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# DASHBOARD FILTERS — shared widget at top of each dashboard page
# ═══════════════════════════════════════════════════════════════════════════
def render_filters(df: pd.DataFrame, key_prefix: str) -> pd.DataFrame:
    """
    Render the standard filter bar (year, region, segment, category) and return
    the filtered DataFrame. Filters are sticky per page via key_prefix.
    """
    if "Year" not in df.columns:
        df = engineer_date_features(df)

    years    = sorted(df["Year"].dropna().unique().astype(int).tolist())
    regions  = sorted(df["Region"].dropna().unique().tolist())
    segments = sorted(df["Segment"].dropna().unique().tolist())
    categs   = sorted(df["Category"].dropna().unique().tolist())

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        sel_years = st.multiselect("Year", years, default=years, key=f"{key_prefix}_y")
    with c2:
        sel_reg = st.multiselect("Region", regions, default=regions, key=f"{key_prefix}_r")
    with c3:
        sel_seg = st.multiselect("Segment", segments, default=segments, key=f"{key_prefix}_s")
    with c4:
        sel_cat = st.multiselect("Category", categs, default=categs, key=f"{key_prefix}_c")

    out = df[
        df["Year"].isin(sel_years) &
        df["Region"].isin(sel_reg) &
        df["Segment"].isin(sel_seg) &
        df["Category"].isin(sel_cat)
    ].copy()

    if len(out) == 0:
        st.warning("No rows match the current filters. Widen the selection above.")
        st.stop()
    return out


# ═══════════════════════════════════════════════════════════════════════════
# DASHBOARD 1 — EXECUTIVE OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════
def render_executive():
    page_title("Executive Overview",
               "Headline performance: revenue, profit, margin, and growth across "
               "the active filter selection.")
    df_full = require_data()
    df = render_filters(df_full, "exec")

    # ── Scorecards (5+) ────────────────────────────────────────────────────
    revenue = df["Total Revenue"].sum()
    profit  = df["Total Profit"].sum()
    margin  = profit / revenue if revenue else 0
    n_orders = df["Order ID"].nunique()
    n_customers = df["Customer ID"].nunique()
    aov = revenue / n_orders if n_orders else 0

    # Year-over-year delta (latest year vs. prior, within filter)
    yr_rev = df.groupby("Year")["Total Revenue"].sum().sort_index()
    if len(yr_rev) >= 2:
        delta_rev = (yr_rev.iloc[-1] - yr_rev.iloc[-2]) / yr_rev.iloc[-2]
        delta_dir = "up" if delta_rev > 0 else "down"
        delta_label = f"{delta_rev*100:+.1f}% YoY"
    else:
        delta_label, delta_dir = None, "flat"

    c = st.columns(5)
    with c[0]: st.markdown(kpi_card("Revenue",   fmt_money(revenue),
                                       delta=delta_label, delta_dir=delta_dir,
                                       accent=True), unsafe_allow_html=True)
    with c[1]: st.markdown(kpi_card("Profit",    fmt_money(profit),
                                       sub=f"{margin*100:.1f}% margin"),
                              unsafe_allow_html=True)
    with c[2]: st.markdown(kpi_card("Orders",    fmt_int(n_orders)),
                              unsafe_allow_html=True)
    with c[3]: st.markdown(kpi_card("Customers", fmt_int(n_customers)),
                              unsafe_allow_html=True)
    with c[4]: st.markdown(kpi_card("Avg order value", fmt_money(aov)),
                              unsafe_allow_html=True)
    st.write("")

    # ── Chart 1: Monthly revenue & profit trend (line, dual series) ─────────
    section("Monthly performance trend",
            "Revenue and profit aggregated to month-end. Profit (green) shows "
            "the bottom line — where the business actually makes money.")
    monthly = (df.assign(_d=pd.to_datetime(df["Order Date"]))
                  .set_index("_d")
                  .resample("MS")[["Total Revenue", "Total Profit"]].sum()
                  .reset_index())
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=monthly["_d"], y=monthly["Total Revenue"],
        mode="lines", line=dict(color=GRAY_500, width=2),
        name="Revenue",
        hovertemplate="<b>%{x|%b %Y}</b><br>Revenue: $%{y:,.0f}<extra></extra>",
    ))
    fig1.add_trace(go.Scatter(
        x=monthly["_d"], y=monthly["Total Profit"],
        mode="lines", line=dict(color=LAU_GREEN, width=2.5),
        name="Profit", fill="tozeroy",
        fillcolor="rgba(0,106,78,0.08)",
        hovertemplate="<b>%{x|%b %Y}</b><br>Profit: $%{y:,.0f}<extra></extra>",
    ))
    fig1.update_layout(height=360, hovermode="x unified")
    st.plotly_chart(fig1, use_container_width=True)

    # ── Charts 2 & 3 side by side ──────────────────────────────────────────
    col_l, col_r = st.columns(2)

    # Chart 2: Profit by Region (horizontal bar, accent on best)
    with col_l:
        section("Profit by region")
        region_p = (df.groupby("Region")["Total Profit"].sum()
                       .sort_values(ascending=True).reset_index())
        max_idx = region_p["Total Profit"].idxmax()
        colors = [LAU_GREEN if i == max_idx else GRAY_500 for i in region_p.index]
        fig2 = go.Figure(go.Bar(
            x=region_p["Total Profit"], y=region_p["Region"],
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[fmt_money(v) for v in region_p["Total Profit"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Profit: $%{x:,.0f}<extra></extra>",
        ))
        fig2.update_layout(height=300, margin=dict(l=10, r=80, t=10, b=10),
                            xaxis_title="", yaxis_title="")
        st.plotly_chart(fig2, use_container_width=True)

    # Chart 3: Revenue mix by Category (treemap)
    with col_r:
        section("Revenue mix by category & sub-category")
        mix = (df.groupby(["Category", "Sub-Category"])["Total Revenue"]
                  .sum().reset_index())
        fig3 = px.treemap(
            mix, path=["Category", "Sub-Category"], values="Total Revenue",
            color="Total Revenue", color_continuous_scale=SEQUENTIAL_GREEN,
        )
        fig3.update_traces(
            hovertemplate="<b>%{label}</b><br>Revenue: $%{value:,.0f}<br>"
                          "%{percentParent} of parent<extra></extra>",
            textinfo="label+percent parent",
        )
        fig3.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig3, use_container_width=True)

    # Chart 4: Annual revenue/profit comparison (clustered bar)
    section("Year-over-year revenue & profit")
    yoy = df.groupby("Year")[["Total Revenue", "Total Profit"]].sum().reset_index()
    fig4 = go.Figure()
    fig4.add_trace(go.Bar(
        x=yoy["Year"], y=yoy["Total Revenue"], name="Revenue",
        marker=dict(color=GRAY_500),
        text=[fmt_money(v) for v in yoy["Total Revenue"]], textposition="outside",
    ))
    fig4.add_trace(go.Bar(
        x=yoy["Year"], y=yoy["Total Profit"], name="Profit",
        marker=dict(color=LAU_GREEN),
        text=[fmt_money(v) for v in yoy["Total Profit"]], textposition="outside",
    ))
    fig4.update_layout(barmode="group", height=320,
                        xaxis=dict(type="category"))
    st.plotly_chart(fig4, use_container_width=True)

    # Insight callout
    best_year = yoy.loc[yoy["Total Profit"].idxmax(), "Year"]
    worst_region = (df.groupby("Region")["Total Profit"].sum().idxmin())
    insight(f"<strong>Best profit year</strong> in selection: {int(best_year)}. "
            f"<strong>Weakest region</strong> by profit: {worst_region} — investigate "
            f"discount policy or shipping costs.")


# ═══════════════════════════════════════════════════════════════════════════
# DASHBOARD 2 — CUSTOMER & PRODUCT
# ═══════════════════════════════════════════════════════════════════════════
def render_customer_product():
    page_title("Customer & Product Insights",
               "Who buys, what they buy, and which products move the needle on profit.")
    df_full = require_data()
    df = render_filters(df_full, "cust")

    # Scorecards
    n_cust = df["Customer ID"].nunique()
    rev_per_cust = df["Total Revenue"].sum() / max(n_cust, 1)
    n_products = df["Product ID"].nunique()
    repeat_rate = (df.groupby("Customer ID")["Order ID"].nunique() > 1).mean()

    seg_rev = df.groupby("Segment")["Total Revenue"].sum()
    top_seg = seg_rev.idxmax() if len(seg_rev) else "—"

    c = st.columns(4)
    with c[0]: st.markdown(kpi_card("Active customers", fmt_int(n_cust), accent=True),
                              unsafe_allow_html=True)
    with c[1]: st.markdown(kpi_card("Revenue / customer", fmt_money(rev_per_cust)),
                              unsafe_allow_html=True)
    with c[2]: st.markdown(kpi_card("Repeat buyer rate", fmt_pct(repeat_rate),
                                       sub="2+ orders"), unsafe_allow_html=True)
    with c[3]: st.markdown(kpi_card("Top segment", top_seg,
                                       sub=fmt_money(seg_rev.max() if len(seg_rev) else 0)),
                              unsafe_allow_html=True)
    st.write("")

    # Chart 1: Profitability by Sub-Category (diverging horizontal bar)
    section("Profitability by sub-category",
            "Loss-makers (red) vs. winners (green). Sorted by total profit.")
    sub = (df.groupby("Sub-Category")["Total Profit"].sum()
              .sort_values().reset_index())
    sub["color"] = sub["Total Profit"].apply(lambda v: ACCENT_RED if v < 0 else LAU_GREEN)
    fig1 = go.Figure(go.Bar(
        x=sub["Total Profit"], y=sub["Sub-Category"],
        orientation="h",
        marker=dict(color=sub["color"], line=dict(width=0)),
        text=[fmt_money(v) for v in sub["Total Profit"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Profit: $%{x:,.0f}<extra></extra>",
    ))
    fig1.add_vline(x=0, line=dict(color=GRAY_300, width=1))
    fig1.update_layout(height=480, margin=dict(l=10, r=80, t=10, b=10))
    st.plotly_chart(fig1, use_container_width=True)

    # Chart 2 + 3 side-by-side
    col_l, col_r = st.columns(2)

    # Chart 2: Segment performance (grouped bars)
    with col_l:
        section("Segment revenue × profit")
        seg = df.groupby("Segment")[["Total Revenue", "Total Profit"]].sum().reset_index()
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=seg["Segment"], y=seg["Total Revenue"],
                                name="Revenue", marker_color=GRAY_500))
        fig2.add_trace(go.Bar(x=seg["Segment"], y=seg["Total Profit"],
                                name="Profit", marker_color=LAU_GREEN))
        fig2.update_layout(barmode="group", height=320)
        st.plotly_chart(fig2, use_container_width=True)

    # Chart 3: Top 10 customers by revenue (lollipop-like horizontal)
    with col_r:
        section("Top 10 customers")
        top_cust = (df.groupby(["Customer ID", "Customer Name"])["Total Revenue"]
                       .sum().reset_index().nlargest(10, "Total Revenue")
                       .sort_values("Total Revenue"))
        fig3 = go.Figure(go.Bar(
            x=top_cust["Total Revenue"], y=top_cust["Customer Name"],
            orientation="h", marker=dict(color=LAU_GREEN_LT),
            text=[fmt_money(v) for v in top_cust["Total Revenue"]],
            textposition="outside",
        ))
        fig3.update_layout(height=320, margin=dict(l=10, r=80, t=10, b=10))
        st.plotly_chart(fig3, use_container_width=True)

    # Chart 4: Discount-free profit-vs-revenue scatter
    section("Order-level profit vs. revenue",
            "Each dot is one order. Below the zero line are the unprofitable orders — "
            "the prediction target on the ML page.")
    sample = df.sample(min(2500, len(df)), random_state=42)
    fig4 = go.Figure()
    profitable = sample[sample["Total Profit"] >= 0]
    losses     = sample[sample["Total Profit"] < 0]
    fig4.add_trace(go.Scatter(
        x=profitable["Total Revenue"], y=profitable["Total Profit"],
        mode="markers",
        marker=dict(color=LAU_GREEN, size=5, opacity=0.5),
        name="Profitable",
        hovertemplate="Revenue: $%{x:,.0f}<br>Profit: $%{y:,.0f}<extra></extra>",
    ))
    fig4.add_trace(go.Scatter(
        x=losses["Total Revenue"], y=losses["Total Profit"],
        mode="markers",
        marker=dict(color=ACCENT_RED, size=5, opacity=0.6),
        name="Loss-making",
        hovertemplate="Revenue: $%{x:,.0f}<br>Loss: $%{y:,.0f}<extra></extra>",
    ))
    fig4.add_hline(y=0, line=dict(color=GRAY_300, width=1, dash="dash"))
    fig4.update_layout(height=380, xaxis_type="log",
                        xaxis_title="Order revenue ($, log scale)",
                        yaxis_title="Order profit ($)")
    st.plotly_chart(fig4, use_container_width=True)

    pct_loss = (df["Total Profit"] < 0).mean()
    insight(f"<strong>{pct_loss*100:.1f}% of orders lose money.</strong> "
            f"Most concentrated in the lower-right quadrant: high revenue but negative "
            f"profit, almost always tied to heavy discounting on Furniture and "
            f"Office Supplies sub-categories like Tables, Bookcases, and Supplies.")


# ═══════════════════════════════════════════════════════════════════════════
# DASHBOARD 3 — GEO & LOGISTICS
# ═══════════════════════════════════════════════════════════════════════════
def render_geo():
    page_title("Geography & Logistics",
               "Where the business operates and how shipping decisions affect profit.")
    df_full = require_data()
    df = render_filters(df_full, "geo")

    n_states = df["State"].nunique()
    n_cities = df["City"].nunique()
    avg_ship = df["Shipping Days"].mean() if "Shipping Days" in df.columns else np.nan
    most_used_mode = df["Ship Mode"].mode().iloc[0] if len(df) else "—"

    c = st.columns(4)
    with c[0]: st.markdown(kpi_card("States covered", fmt_int(n_states), accent=True),
                              unsafe_allow_html=True)
    with c[1]: st.markdown(kpi_card("Cities served", fmt_int(n_cities)),
                              unsafe_allow_html=True)
    with c[2]: st.markdown(kpi_card("Avg shipping days",
                                       f"{avg_ship:.1f}" if not pd.isna(avg_ship) else "—"),
                              unsafe_allow_html=True)
    with c[3]: st.markdown(kpi_card("Top ship mode", most_used_mode),
                              unsafe_allow_html=True)
    st.write("")

    # Chart 1: US state choropleth (profit)
    section("Profit by state",
            "Darker green = higher cumulative profit. White states are not in the filter.")
    STATE_ABBR = {"Alabama":"AL","Alaska":"AK","Arizona":"AZ","Arkansas":"AR","California":"CA",
        "Colorado":"CO","Connecticut":"CT","Delaware":"DE","District of Columbia":"DC",
        "Florida":"FL","Georgia":"GA","Hawaii":"HI","Idaho":"ID","Illinois":"IL","Indiana":"IN",
        "Iowa":"IA","Kansas":"KS","Kentucky":"KY","Louisiana":"LA","Maine":"ME","Maryland":"MD",
        "Massachusetts":"MA","Michigan":"MI","Minnesota":"MN","Mississippi":"MS","Missouri":"MO",
        "Montana":"MT","Nebraska":"NE","Nevada":"NV","New Hampshire":"NH","New Jersey":"NJ",
        "New Mexico":"NM","New York":"NY","North Carolina":"NC","North Dakota":"ND","Ohio":"OH",
        "Oklahoma":"OK","Oregon":"OR","Pennsylvania":"PA","Rhode Island":"RI","South Carolina":"SC",
        "South Dakota":"SD","Tennessee":"TN","Texas":"TX","Utah":"UT","Vermont":"VT","Virginia":"VA",
        "Washington":"WA","West Virginia":"WV","Wisconsin":"WI","Wyoming":"WY"}
    state_p = df.groupby("State")["Total Profit"].sum().reset_index()
    state_p["abbr"] = state_p["State"].map(STATE_ABBR)
    state_p = state_p.dropna(subset=["abbr"])
    fig1 = go.Figure(go.Choropleth(
        locations=state_p["abbr"], locationmode="USA-states",
        z=state_p["Total Profit"], colorscale=[
            [0.0, ACCENT_RED], [0.5, GRAY_100], [1.0, LAU_GREEN_DK]
        ],
        zmid=0,
        text=state_p["State"],
        hovertemplate="<b>%{text}</b><br>Profit: $%{z:,.0f}<extra></extra>",
        colorbar=dict(title="Profit", thickness=12, len=0.7),
    ))
    fig1.update_layout(geo=dict(scope="usa", showlakes=False, bgcolor="rgba(0,0,0,0)"),
                        height=400, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig1, use_container_width=True)

    # Chart 2 & 3 side by side
    col_l, col_r = st.columns(2)

    # Chart 2: Top & bottom states by profit
    with col_l:
        section("Top 5 / bottom 5 states by profit")
        sp = state_p.set_index("State")["Total Profit"].sort_values()
        bot = sp.head(5).reset_index()
        top = sp.tail(5).reset_index()
        combo = pd.concat([bot.assign(group="Bottom"), top.assign(group="Top")])
        combo = combo.sort_values("Total Profit")
        colors = [ACCENT_RED if v < 0 else LAU_GREEN for v in combo["Total Profit"]]
        fig2 = go.Figure(go.Bar(
            x=combo["Total Profit"], y=combo["State"], orientation="h",
            marker=dict(color=colors),
            text=[fmt_money(v) for v in combo["Total Profit"]], textposition="outside",
        ))
        fig2.add_vline(x=0, line=dict(color=GRAY_300, width=1))
        fig2.update_layout(height=380, margin=dict(l=10, r=80, t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    # Chart 3: Ship mode share of orders
    with col_r:
        section("Ship mode mix")
        ship = df["Ship Mode"].value_counts().reset_index()
        ship.columns = ["Ship Mode", "Orders"]
        ship["pct"] = ship["Orders"] / ship["Orders"].sum()
        fig3 = go.Figure(go.Bar(
            x=ship["pct"]*100, y=ship["Ship Mode"], orientation="h",
            marker=dict(color=[LAU_GREEN if i == 0 else GRAY_500
                                for i in range(len(ship))]),
            text=[f"{p*100:.1f}%" for p in ship["pct"]],
            textposition="outside",
        ))
        fig3.update_layout(height=380, margin=dict(l=10, r=80, t=10, b=10),
                            xaxis_title="Share of orders (%)")
        st.plotly_chart(fig3, use_container_width=True)

    # Chart 4: Shipping days vs. profit margin (regional)
    section("Shipping days distribution by ship mode",
            "Box plot — does fast shipping eat margin?")
    if "Shipping Days" in df.columns:
        fig4 = go.Figure()
        for mode in df["Ship Mode"].unique():
            sub = df[df["Ship Mode"] == mode]
            fig4.add_trace(go.Box(
                y=sub["Shipping Days"], name=mode,
                marker_color=LAU_GREEN, boxmean=True,
                line=dict(width=1.5),
            ))
        fig4.update_layout(height=320, showlegend=False,
                            yaxis_title="Days", xaxis_title="")
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("Run the bonus 'Date features' cleaning step to enable this chart.")

    insight(f"The strongest profit comes from a handful of populous states. "
            f"Same-Day shipping is rare in the mix but worth checking against "
            f"its cost — it's the most likely candidate for margin erosion.")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 6 — ML LAB
# ═══════════════════════════════════════════════════════════════════════════
def _svm_model(model_name: str) -> bool:
    return model_name == "SVC (RBF kernel)"

def _tree_model(model_name: str) -> bool:
    return model_name in ("Random Forest", "Gradient Boosting")


def _classifier_hyperparam_ui(prefix: str, model_name: str) -> dict:
    """Render model hyperparameter widgets. Returns hp dict."""
    c2, c3 = st.columns(2)
    hp = {}
    with c2:
        if _tree_model(model_name):
            hp["n_estimators"] = st.slider(
                "n_estimators", 20, 300, 100, 20, key=f"{prefix}_ne")
        elif _svm_model(model_name):
            hp["C"] = st.slider(
                "C (regularisation)", 0.01, 10.0, 1.0, 0.01,
                key=f"{prefix}_c",
                help="Larger C = less regularisation, tighter decision boundary.")
        else:
            st.slider("n_estimators", 20, 300, 100, 20,
                      key=f"{prefix}_ne", disabled=True)
    with c3:
        if _tree_model(model_name):
            hp["max_depth"] = st.slider(
                "max_depth", 2, 15, 6, 1, key=f"{prefix}_md")
        else:
            st.slider("max_depth", 2, 15, 6, 1,
                      key=f"{prefix}_md", disabled=True)
    if _svm_model(model_name):
        st.caption("ℹ️ SVC is automatically capped at 2,000 rows and uses "
                   "3-fold CV to stay within Streamlit Cloud limits (~20 s).")
    return hp


def render_ml_lab():
    page_title("Machine Learning Lab",
               "Two supervised classifiers (multi-class & binary), customer segmentation, "
               "and time-series forecasting — each defensible end-to-end.")
    df = require_data()
    if "Profit Margin" not in df.columns:
        df = engineer_date_features(df)

    tab_mc, tab_c, tab_s, tab_f = st.tabs([
        "📊 Profit-tier classifier",
        "🚨 Binary classifier",
        "👥 Customer segmentation",
        "📈 Sales forecast",
    ])

    # ── Tab MC: Multi-class classifier ─────────────────────────────────────
    with tab_mc:
        section("Predict a categorical profit / revenue tier per order",
                "Both targets produce a **3-class categorical label**. "
                "All four models are classifiers — evaluated with macro F1 "
                "(treats every class equally) via 5-fold stratified CV.")

        mc_target_label = st.selectbox(
            "🎯 Target variable",
            list(MULTICLASS_TARGETS.keys()),
            key="mc_target",
            help="Profit-Margin Tier bins orders into Loss / Low / High margin. "
                 "Order-Value Tier bins by revenue tertile (Small / Medium / Large).",
        )
        mc_target_key = MULTICLASS_TARGETS[mc_target_label]

        tier_desc = (
            "Loss (margin < 0%) · Low Margin (0 – 15%) · High Margin (> 15%)"
            if mc_target_key == "margin_tier"
            else "Small / Medium / Large — based on revenue tertiles of the loaded dataset"
        )
        st.markdown(
            f'<div class="info-box">Predicting <strong>{mc_target_label}</strong>. '
            f'Classes: <em>{tier_desc}</em>.</div>',
            unsafe_allow_html=True,
        )

        c1, *_ = st.columns([1, 2])
        with c1:
            mc_model = st.selectbox("Model", list(CLASSIFIERS.keys()), key="mc_model")
        mc_hp = _classifier_hyperparam_ui("mc", mc_model)

        if st.button("Train multi-class classifier", key="train_mc"):
            with st.spinner("Cross-validating and training…"):
                pipe, m = train_multiclass(mc_model, df,
                                           target=mc_target_key, hyperparams=mc_hp)
                st.session_state.ml_results["mc"] = {
                    "pipe": pipe, "metrics": m,
                    "model": mc_model, "target_label": mc_target_label,
                }

        if "mc" in st.session_state.ml_results:
            _show_multiclass_results(st.session_state.ml_results["mc"])
            _render_prediction_panel(
                st.session_state.ml_results["mc"], df, mode="mc"
            )

    # ── Tab C: Binary classifier ────────────────────────────────────────────
    with tab_c:
        section("Predict a binary categorical outcome per order",
                "Binary target (0 / 1). All four models are classifiers. "
                "5-fold stratified CV keeps class balance in every fold.")

        c_target_label = st.selectbox(
            "🎯 Target variable",
            list(CLASSIFICATION_TARGETS.keys()),
            key="c_target",
            help="Choose the binary outcome the model will learn to predict.",
        )
        c_target_key = CLASSIFICATION_TARGETS[c_target_label]
        st.markdown(
            f'<div class="info-box">Predicting <strong>{c_target_label}</strong> '
            f'(binary 0 / 1). Features: quantity, revenue, ship mode, segment, region, '
            f'category, sub-category, shipping days, and engineered date signals.</div>',
            unsafe_allow_html=True,
        )

        c1, *_ = st.columns([1, 2])
        with c1:
            c_model = st.selectbox("Model", list(CLASSIFIERS.keys()), key="c_model")
        c_hp = _classifier_hyperparam_ui("c", c_model)

        if st.button("Train binary classifier", key="train_c"):
            with st.spinner("Cross-validating and training…"):
                pipe, m = train_classifier(c_model, df,
                                           target=c_target_key, hyperparams=c_hp)
                st.session_state.ml_results["clf"] = {
                    "pipe": pipe, "metrics": m,
                    "model": c_model, "target_label": c_target_label,
                }

        if "clf" in st.session_state.ml_results:
            _show_classification_results(st.session_state.ml_results["clf"])
            _render_prediction_panel(
                st.session_state.ml_results["clf"], df, mode="clf"
            )

    # ── Tab S: Segmentation ────────────────────────────────────────────────
    with tab_s:
        section("RFM customer segmentation via K-Means",
                "Recency, Frequency, Monetary features → unsupervised clusters → "
                "tier names by Monetary value.")
        c1, c2 = st.columns(2)
        with c1:
            k = st.slider("Number of clusters (k)", 2, 6, 4, 1, key="km_k")
        with c2:
            log_t = st.checkbox("Log-transform Frequency & Monetary",
                                  value=True, key="km_log",
                                  help="Reduces skew so clusters are more balanced.")
        if st.button("Run K-Means", key="train_km"):
            with st.spinner("Clustering customers…"):
                rfm = build_rfm(df)
                rfm_c, model, scaler, sil = kmeans_rfm(rfm, k=k, log_transform=log_t)
                elbow = kmeans_elbow(rfm, log_transform=log_t)
                st.session_state.ml_results["clu"] = {
                    "rfm": rfm_c, "k": k, "silhouette": sil, "elbow": elbow
                }
        if "clu" in st.session_state.ml_results:
            _show_segmentation_results(st.session_state.ml_results["clu"])

    # ── Tab F: Forecast ────────────────────────────────────────────────────
    with tab_f:
        section("Monthly revenue forecast",
                "SARIMA trained on 2017–2020 monthly totals. Seasonal-naive baseline "
                "shown for comparison. No Prophet — pure statsmodels for reliability.")
        h = st.slider("Horizon (months)", 3, 12, 6, 1, key="fc_h")
        run_backtest = st.checkbox("Also run backtest on last "
                                     f"{h} months", value=True, key="fc_bt")
        if st.button("Generate forecast", key="train_fc"):
            with st.spinner("Fitting SARIMA…"):
                series = aggregate_monthly(df, "Total Revenue")
                naive = seasonal_naive_forecast(series, horizon=h)
                try:
                    sar, fitted, res = sarima_forecast(series, horizon=h)
                    bt = backtest(series, horizon=h, model="sarima") if run_backtest else None
                    st.session_state.ml_results["fc"] = {
                        "series": series, "naive": naive, "sarima": sar,
                        "fitted": fitted, "backtest": bt
                    }
                except Exception as e:
                    st.error(f"SARIMA failed: {e}. Falling back to seasonal naive only.")
                    st.session_state.ml_results["fc"] = {
                        "series": series, "naive": naive, "sarima": None,
                        "fitted": None, "backtest": None
                    }
        if "fc" in st.session_state.ml_results:
            _show_forecast_results(st.session_state.ml_results["fc"])

    # ── Tab P: Prediction Tool ─────────────────────────────────────────────
    with tab_p:
        _render_prediction_tab(df)


# ══════════════════════════════════════════════════════════════════════════
# Prediction panel — shared by regression and classification tabs
# ══════════════════════════════════════════════════════════════════════════
def _render_prediction_panel(state: dict, df: pd.DataFrame, mode: str):
    """
    Render an interactive prediction tool below the model results.

    Parameters
    ----------
    state : ml_results dict (contains pipe, metrics, target_label)
    df    : the active dataset (used for dropdown options & sensible defaults)
    mode  : "reg" or "clf"
    """
    m            = state["metrics"]
    pipe         = state["pipe"]
    num_cols     = m["num_cols"]
    cat_cols     = m["cat_cols"]
    cat_values   = m["cat_values"]
    num_defaults = m["num_defaults"]
    target_label = state.get("target_label", m.get("target_col", "target"))

    with st.expander("🔮 Predict on new input", expanded=False):
        st.markdown(
            f"Enter values for the order features below and the trained "
            f"**{state['model']}** will return its prediction for "
            f"**{target_label}**."
        )

        # ── Raw numeric inputs (user-visible, excludes engineered cols) ──
        raw_num = [c for c in num_cols
                   if c not in ("Rev_per_unit", "Order_Month", "Order_Quarter")]

        input_vals: dict = {}

        # Numeric inputs
        num_cols_layout = st.columns(max(len(raw_num), 1))
        for i, col in enumerate(raw_num):
            default = num_defaults.get(col, 1.0)
            step    = 1.0 if col == "Quantity" else (10.0 if default > 10 else 1.0)
            with num_cols_layout[i]:
                input_vals[col] = st.number_input(
                    col, value=float(default),
                    min_value=0.0, step=step,
                    key=f"pred_{mode}_{col}",
                )

        # Month selector (derives Order_Month and Order_Quarter)
        if "Order_Month" in num_cols:
            pred_month = st.slider(
                "Order Month (1 = Jan … 12 = Dec)", 1, 12, 6,
                key=f"pred_{mode}_month",
            )
            input_vals["Order_Month"]   = pred_month
            input_vals["Order_Quarter"] = int((pred_month - 1) // 3 + 1)

        # Derive engineered Rev_per_unit
        qty = input_vals.get("Quantity", 1.0)
        rev = input_vals.get("Total Revenue", 0.0)
        input_vals["Rev_per_unit"] = rev / max(qty, 1.0)

        # Categorical inputs
        if cat_cols:
            cat_layout = st.columns(min(len(cat_cols), 3))
            for i, col in enumerate(cat_cols):
                opts = cat_values.get(col, ["Unknown"])
                with cat_layout[i % len(cat_layout)]:
                    input_vals[col] = st.selectbox(
                        col, opts, key=f"pred_{mode}_{col}"
                    )

        # ── Predict ──────────────────────────────────────────────────────
        if st.button("▶ Get prediction", key=f"btn_pred_{mode}"):
            try:
                pred_val = predict_single(pipe, input_vals, num_cols, cat_cols)[0]

                if mode == "reg":
                    tkey = m.get("target_key", "profit_margin")
                    if tkey == "profit_margin":
                        st.success(
                            f"**Predicted {target_label}:** {pred_val * 100:.2f}%"
                        )
                    elif tkey == "shipping_days":
                        days = max(0, round(pred_val))
                        st.success(
                            f"**Predicted {target_label}:** {days} day(s)"
                        )
                    else:
                        st.success(f"**Predicted {target_label}:** {pred_val:.4f}")

                else:  # clf
                    proba_arr = predict_single_proba(
                        pipe, input_vals, num_cols, cat_cols
                    )[0]
                    pos_prob  = float(proba_arr[1])
                    neg_prob  = float(proba_arr[0])
                    label_str = "✅ Positive class" if pred_val == 1 else "❌ Negative class"
                    tkey = m.get("target_key", "is_unprofitable")
                    if tkey == "is_unprofitable":
                        label_str = (
                            "🔴 Loss-making order" if pred_val == 1
                            else "🟢 Profitable order"
                        )
                    elif tkey == "is_high_revenue":
                        label_str = (
                            "🟢 High-revenue order" if pred_val == 1
                            else "⚪ Standard-revenue order"
                        )
                    st.success(
                        f"**Prediction:** {label_str}  \n"
                        f"Confidence — positive class: **{pos_prob*100:.1f}%** | "
                        f"negative class: **{neg_prob*100:.1f}%**"
                    )
            except Exception as e:
                st.error(f"Prediction failed: {e}")


def _render_prediction_tab(df: pd.DataFrame):
    """Standalone, customer-facing prediction tool tab."""
    section("Try it yourself",
            "Describe an order using the form below and a trained model will predict "
            "its outcome instantly. Train a model first in the Regression or Classifier tabs.")

    has_reg = "reg" in st.session_state.ml_results
    has_clf = "clf" in st.session_state.ml_results

    if not has_reg and not has_clf:
        st.markdown(
            '<div class="info-box">'
            "<strong>No models trained yet.</strong> Here's how to get started:<br><br>"
            "<ol>"
            "<li>Go to the <strong>🎯 Regression</strong> tab to train a model that predicts "
            "a number (e.g. profit margin or shipping days).</li>"
            "<li>Or go to the <strong>🚨 Classifier</strong> tab to train a model that answers "
            "yes/no questions (e.g. will this order lose money?).</li>"
            "<li>Come back here, fill in your order details, and click <strong>Get prediction</strong>.</li>"
            "</ol>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    # ── Model selector ────────────────────────────────────────────────────
    options, keys = [], []
    if has_reg:
        reg_state = st.session_state.ml_results["reg"]
        options.append(
            f"📊 Regression — predicts '{reg_state['target_label']}' "
            f"using {reg_state['model']}"
        )
        keys.append("reg")
    if has_clf:
        clf_state = st.session_state.ml_results["clf"]
        options.append(
            f"🔍 Classifier — answers '{clf_state['target_label']}' "
            f"using {clf_state['model']}"
        )
        keys.append("clf")

    if len(options) > 1:
        chosen_idx = st.radio(
            "Which model would you like to use?", range(len(options)),
            format_func=lambda i: options[i], key="pred_tab_choice"
        )
        mode = keys[chosen_idx]
    else:
        mode = keys[0]
        st.info(options[0])

    state        = st.session_state.ml_results[mode]
    m            = state["metrics"]
    pipe         = state["pipe"]
    num_cols     = m["num_cols"]
    cat_cols     = m["cat_cols"]
    cat_values   = m["cat_values"]
    num_defaults = m["num_defaults"]
    target_label = state.get("target_label", m.get("target_col", "target"))

    st.markdown("---")
    st.markdown("### 📋 Enter your order details")

    raw_num = [c for c in num_cols
               if c not in ("Rev_per_unit", "Order_Month", "Order_Quarter")]

    input_vals: dict = {}
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**📦 Quantities & Revenue**")
        friendly_num = {
            "Quantity":      "Quantity (number of items)",
            "Total Revenue": "Total Revenue ($)",
            "Shipping Days": "Shipping Days",
        }
        for col in raw_num:
            default = num_defaults.get(col, 1.0)
            step    = 1.0 if col == "Quantity" else (10.0 if default > 10 else 1.0)
            input_vals[col] = st.number_input(
                friendly_num.get(col, col),
                value=float(default), min_value=0.0, step=step,
                key=f"predtab_{mode}_{col}",
            )

        if "Order_Month" in num_cols:
            month_names = ["January", "February", "March", "April", "May", "June",
                           "July", "August", "September", "October", "November", "December"]
            pred_month = st.selectbox(
                "Order Month",
                options=list(range(1, 13)),
                format_func=lambda x: month_names[x - 1],
                key=f"predtab_{mode}_month",
            )
            input_vals["Order_Month"]   = pred_month
            input_vals["Order_Quarter"] = int((pred_month - 1) // 3 + 1)

    with col_right:
        st.markdown("**🏷️ Order Categories**")
        friendly_cat = {
            "Ship Mode":    "Shipping Speed",
            "Segment":      "Customer Type",
            "Region":       "Region",
            "Category":     "Product Category",
            "Sub-Category": "Product Sub-Category",
        }
        for col in cat_cols:
            opts = cat_values.get(col, ["Unknown"])
            input_vals[col] = st.selectbox(
                friendly_cat.get(col, col), opts,
                key=f"predtab_{mode}_{col}",
            )

    # Derive engineered features
    qty = input_vals.get("Quantity", 1.0)
    rev = input_vals.get("Total Revenue", 0.0)
    input_vals["Rev_per_unit"] = rev / max(qty, 1.0)

    st.markdown("")
    if st.button("▶ Get prediction", key=f"btn_predtab_{mode}"):
        try:
            pred_val = predict_single(pipe, input_vals, num_cols, cat_cols)[0]
            st.markdown("---")
            st.markdown("### 🎯 Result")

            if mode == "mc":
                # Multi-class: show predicted tier + probability bar per class
                proba_arr = predict_single_proba(pipe, input_vals, num_cols, cat_cols)[0]
                classes   = m.get("classes", [])
                tkey      = m.get("target_key", "margin_tier")
                icons     = {"Loss": "🔴", "Low Margin": "🟡", "High Margin": "🟢",
                             "Small": "⚪", "Medium": "🟡", "Large": "🟢"}
                icon      = icons.get(str(pred_val), "📊")
                best_prob = float(max(proba_arr))
                st.success(
                    f"{icon} **Predicted tier: {pred_val}** "
                    f"(model confidence: {best_prob*100:.1f}%)"
                )
                if classes:
                    prob_df = pd.DataFrame({
                        "Class": classes,
                        "Probability (%)": [p * 100 for p in proba_arr],
                    })
                    fig_p = go.Figure(go.Bar(
                        x=prob_df["Class"],
                        y=prob_df["Probability (%)"],
                        marker_color=[LAU_GREEN if c == str(pred_val) else GRAY_300
                                      for c in classes],
                        text=[f"{p:.1f}%" for p in prob_df["Probability (%)"]],
                        textposition="outside",
                    ))
                    fig_p.update_layout(
                        height=260, yaxis_title="Probability (%)",
                        margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                    )
                    st.plotly_chart(fig_p, use_container_width=True)

            elif mode == "clf":
                proba_arr = predict_single_proba(pipe, input_vals, num_cols, cat_cols)[0]
                pos_prob  = float(proba_arr[1])
                neg_prob  = float(proba_arr[0])
                pred_cls  = int(pred_val)
                tkey      = m.get("target_key", "is_unprofitable")

                if tkey == "is_unprofitable":
                    if pred_cls == 1:
                        st.error(
                            f"🔴 **This order is predicted to LOSE MONEY**\n\n"
                            f"The model is **{pos_prob*100:.1f}%** confident it will be unprofitable. "
                            f"Consider reviewing pricing, discounts, or shipping costs."
                        )
                    else:
                        st.success(
                            f"🟢 **This order is predicted to be PROFITABLE**\n\n"
                            f"Confidence: **{neg_prob*100:.1f}%** profitable. "
                            f"Loss probability: {pos_prob*100:.1f}%."
                        )
                elif tkey == "is_high_revenue":
                    if pred_cls == 1:
                        st.success(
                            f"🟢 **This looks like a HIGH-REVENUE order**\n\n"
                            f"Confidence: **{pos_prob*100:.1f}%**."
                        )
                    else:
                        st.info(
                            f"⚪ **This looks like a standard-revenue order**\n\n"
                            f"Confidence: **{neg_prob*100:.1f}%**."
                        )
                else:
                    label_str = "✅ Yes" if pred_cls == 1 else "❌ No"
                    st.success(
                        f"**{target_label}:** {label_str}  \n"
                        f"Confidence — Yes: **{pos_prob*100:.1f}%** | No: **{neg_prob*100:.1f}%**"
                    )
        except Exception as e:
            st.error(f"Prediction failed: {e}")


def _show_multiclass_results(state: dict):
    m       = state["metrics"]
    classes = m["classes"]

    c = st.columns(4)
    with c[0]: st.markdown(kpi_card("CV F1 Macro", f"{m['cv_f1_mean']:.3f}",
                                       sub=f"± {m['cv_f1_std']:.3f}", accent=True),
                              unsafe_allow_html=True)
    with c[1]: st.markdown(kpi_card("CV Accuracy", f"{m['cv_acc_mean']:.3f}"),
                              unsafe_allow_html=True)
    with c[2]: st.markdown(kpi_card("Test F1 Macro", f"{m['test_f1']:.3f}"),
                              unsafe_allow_html=True)
    with c[3]: st.markdown(kpi_card("Test Accuracy", fmt_pct(m["test_acc"])),
                              unsafe_allow_html=True)
    st.write("")

    cl, cr = st.columns(2)
    with cl:
        section("Confusion matrix")
        cm = m["confusion"]
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=[f"Pred: {c}" for c in classes],
            y=[f"Actual: {c}" for c in classes],
            text=cm, texttemplate="%{text:,}", textfont=dict(size=13),
            colorscale=SEQUENTIAL_GREEN, showscale=False,
        ))
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with cr:
        section("Class distribution in training data")
        dist = m["class_dist"]
        fig = go.Figure(go.Bar(
            x=list(dist.keys()),
            y=[v * 100 for v in dist.values()],
            marker=dict(color=[LAU_GREEN, ACCENT_GOLD, GRAY_500][:len(dist)],
                        line=dict(width=0)),
            text=[f"{v*100:.1f}%" for v in dist.values()],
            textposition="outside",
        ))
        fig.update_layout(height=320, yaxis_title="Share of orders (%)",
                           margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    section("Top features driving tier prediction")
    fi = feature_importance(state["pipe"], top_n=10)
    if not fi.empty:
        fi = fi.sort_values("importance")
        fig = go.Figure(go.Bar(
            x=fi["importance"], y=fi["feature"], orientation="h",
            marker=dict(color=LAU_GREEN_LT),
        ))
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Feature importance not available for this model.")


def _show_classification_results(state: dict):
    m = state["metrics"]
    c = st.columns(4)
    with c[0]: st.markdown(kpi_card("CV ROC-AUC", f"{m['cv_auc_mean']:.3f}",
                                       sub=f"± {m['cv_auc_std']:.3f}", accent=True),
                              unsafe_allow_html=True)
    with c[1]: st.markdown(kpi_card("CV F1", f"{m['cv_f1_mean']:.3f}"),
                              unsafe_allow_html=True)
    with c[2]: st.markdown(kpi_card("Test accuracy", fmt_pct(m["test_acc"])),
                              unsafe_allow_html=True)
    with c[3]: st.markdown(kpi_card("Loss-making rate", fmt_pct(m["positive_rate"]),
                                       sub="class balance"),
                              unsafe_allow_html=True)
    st.write("")

    cl, cr = st.columns(2)
    with cl:
        section("Confusion matrix")
        cm = m["confusion"]
        fig = go.Figure(data=go.Heatmap(
            z=cm, x=["Predicted profitable", "Predicted loss"],
            y=["Actually profitable", "Actually loss"],
            text=cm, texttemplate="%{text:,}", textfont=dict(size=14),
            colorscale=SEQUENTIAL_GREEN, showscale=False,
        ))
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with cr:
        section("Probability distribution by actual class")
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=m["y_proba"][m["y_test"] == 0], name="Profitable",
            marker_color=LAU_GREEN, opacity=0.6, nbinsx=30,
        ))
        fig.add_trace(go.Histogram(
            x=m["y_proba"][m["y_test"] == 1], name="Loss-making",
            marker_color=ACCENT_RED, opacity=0.6, nbinsx=30,
        ))
        fig.update_layout(barmode="overlay", height=300,
                            xaxis_title="Predicted probability of loss",
                            yaxis_title="Orders")
        st.plotly_chart(fig, use_container_width=True)

    section("Top features driving loss prediction")
    fi = feature_importance(state["pipe"], top_n=10)
    if not fi.empty:
        fi = fi.sort_values("importance")
        fig = go.Figure(go.Bar(x=fi["importance"], y=fi["feature"],
                                 orientation="h", marker_color=LAU_GREEN_LT))
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)


def _show_segmentation_results(state: dict):
    rfm = state["rfm"]
    c = st.columns(4)
    with c[0]: st.markdown(kpi_card("Customers", fmt_int(len(rfm)), accent=True),
                              unsafe_allow_html=True)
    with c[1]: st.markdown(kpi_card("Clusters", str(state["k"])),
                              unsafe_allow_html=True)
    with c[2]: st.markdown(kpi_card("Silhouette", f"{state['silhouette']:.3f}",
                                       sub="higher = better separation"),
                              unsafe_allow_html=True)
    with c[3]:
        top_seg = rfm.groupby("Segment")["Monetary"].sum().idxmax()
        st.markdown(kpi_card("Largest segment by $", top_seg),
                       unsafe_allow_html=True)
    st.write("")

    cl, cr = st.columns(2)
    with cl:
        section("Segment sizes & average value")
        agg = rfm.groupby("Segment").agg(
            customers=("Customer ID", "count"),
            avg_monetary=("Monetary", "mean"),
            avg_recency=("Recency", "mean"),
            avg_frequency=("Frequency", "mean"),
        ).reset_index().sort_values("avg_monetary", ascending=False)
        st.dataframe(
            agg.style.format({
                "avg_monetary": "${:,.0f}",
                "avg_recency": "{:.0f} d",
                "avg_frequency": "{:.1f}",
            }),
            use_container_width=True, hide_index=True,
        )
    with cr:
        section("Elbow & silhouette")
        elbow = state["elbow"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=elbow["k"], y=elbow["inertia"], mode="lines+markers",
            line=dict(color=GRAY_500, width=2), name="Inertia",
            marker=dict(size=8),
        ))
        fig.add_trace(go.Scatter(
            x=elbow["k"], y=elbow["silhouette"]*elbow["inertia"].max(),
            mode="lines+markers",
            line=dict(color=LAU_GREEN, width=2), name="Silhouette (rescaled)",
            yaxis="y2", marker=dict(size=8),
        ))
        fig.update_layout(
            height=320,
            xaxis_title="k", yaxis_title="Inertia",
            yaxis2=dict(overlaying="y", side="right",
                         showgrid=False, title="Silhouette"),
        )
        st.plotly_chart(fig, use_container_width=True)

    section("RFM space coloured by segment")
    fig = px.scatter(
        rfm, x="Recency", y="Monetary",
        size="Frequency", color="Segment",
        hover_data=["Customer Name"],
        color_discrete_sequence=[LAU_GREEN, GRAY_700, LAU_GREEN_LT,
                                   ACCENT_GOLD, ACCENT_RED, GRAY_500],
        size_max=24,
    )
    fig.update_layout(height=420, xaxis_title="Recency (days since last order)",
                        yaxis_title="Monetary ($)")
    st.plotly_chart(fig, use_container_width=True)


def _show_forecast_results(state: dict):
    series = state["series"]
    naive  = state["naive"]
    sar    = state["sarima"]
    fitted = state["fitted"]
    bt     = state["backtest"]

    cur_total = float(series.tail(12).sum())
def _show_forecast_results(state: dict):
    series = state["series"]
    naive  = state["naive"]
    sar    = state["sarima"]
    fitted = state["fitted"]
    bt     = state["backtest"]

    cur_total = float(series.tail(12).sum())
    fc_total  = float(sar["yhat"].sum()) if sar is not None else float(naive["yhat"].sum())
    growth    = (fc_total / cur_total - 1) if cur_total else 0

    c = st.columns(4)
    with c[0]: st.markdown(kpi_card("Trailing-12 revenue", fmt_money(cur_total), accent=True),
                              unsafe_allow_html=True)
    with c[1]: st.markdown(kpi_card("Forecast horizon", f"{len(naive)} mo"),
                              unsafe_allow_html=True)
    with c[2]: st.markdown(kpi_card("Forecast total", fmt_money(fc_total)),
                              unsafe_allow_html=True)
    with c[3]: st.markdown(kpi_card("vs. trailing-12", f"{growth*100:+.1f}%",
                                       delta_dir="up" if growth >= 0 else "down"),
                              unsafe_allow_html=True)
    st.write("")

    section("Historical, fitted, and forecast")
    fig = go.Figure()
    # History
    fig.add_trace(go.Scatter(
        x=series.index, y=series.values, mode="lines",
        line=dict(color=GRAY_700, width=2), name="History",
        hovertemplate="<b>%{x|%b %Y}</b><br>$%{y:,.0f}<extra></extra>",
    ))
    if fitted is not None:
        fig.add_trace(go.Scatter(
            x=fitted.index, y=fitted.values, mode="lines",
            line=dict(color=GRAY_500, width=1, dash="dot"), name="SARIMA fit",
            hoverinfo="skip",
        ))
    if sar is not None:
        # Anchor point: last historical value so forecast connects to history
        last_x = series.index[-1]
        last_y = float(series.values[-1])
        fc_x = [last_x] + list(sar["ds"])
        fc_y = [last_y] + list(sar["yhat"])
        fc_upper = [last_y] + list(sar["yhat_upper"])
        fc_lower = [last_y] + list(sar["yhat_lower"])
        # Confidence band
        fig.add_trace(go.Scatter(
            x=fc_x + fc_x[::-1],
            y=fc_upper + fc_lower[::-1],
            fill="toself", fillcolor="rgba(0,106,78,0.10)",
            line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip",
            name="95% CI",
        ))
        fig.add_trace(go.Scatter(
            x=fc_x, y=fc_y, mode="lines+markers",
            line=dict(color=LAU_GREEN, width=2.5), name="SARIMA forecast",
            marker=dict(size=7),
            hovertemplate="<b>%{x|%b %Y}</b><br>$%{y:,.0f}<extra></extra>",
        ))
    # Seasonal naive — also connect from last historical point
    naive_x = [series.index[-1]] + list(naive["ds"])
    naive_y = [float(series.values[-1])] + list(naive["yhat"])
    fig.add_trace(go.Scatter(
        x=naive_x, y=naive_y, mode="lines+markers",
        line=dict(color=ACCENT_GOLD, width=1.5, dash="dash"),
        marker=dict(size=6, symbol="diamond-open"),
        name="Seasonal naive",
        hovertemplate="<b>%{x|%b %Y}</b><br>$%{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(height=420, hovermode="x unified",
                        yaxis_title="Monthly revenue ($)")
    st.plotly_chart(fig, use_container_width=True)

    if bt is not None and "error" not in bt:
        section("Backtest — last horizon held out",
                "Tests whether SARIMA generalizes by training only on data prior to "
                f"{bt['test_start']}.")
        c = st.columns(3)
        with c[0]: st.markdown(kpi_card("Backtest MAE", fmt_money(bt["mae"]), accent=True),
                                  unsafe_allow_html=True)
        with c[1]: st.markdown(kpi_card("Backtest RMSE", fmt_money(bt["rmse"])),
                                  unsafe_allow_html=True)
        with c[2]: st.markdown(kpi_card("Backtest MAPE", f"{bt['mape']:.1f}%"),
                                  unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 7 — ABOUT
# ═══════════════════════════════════════════════════════════════════════════
def render_about():
    page_title("About this project",
               "DAN614 individual project · Knowledge Discovery in Databases pipeline")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"""
### Project context

This application is the deliverable for the DAN614 Individual Project at the
Lebanese American University, Adnan Kassar School of Business, Master of Science
in Data Analytics.

It implements the full **Knowledge Discovery in Databases (KDD)** pipeline:

1. **Selection** — Two data-source modes (local file, Google Sheets API)
2. **Pre-processing** — Five distinct cleaning measures
3. **Transformation** — Date features, category standardization, RFM aggregation
4. **Data Mining** — Three supervised models, K-Means clustering, SARIMA forecast
5. **Interpretation** — Three business dashboards with insight callouts

### Visual design philosophy

Every chart on every page applies the principles taught in Advanced Data Visualization:

- **One accent color** (LAU green) drawn from preattentive-attribute theory —
  color is precious, and used only where the insight lives.
- **Decluttered axes** — no chart borders, no secondary y-axes, minimal gridlines.
- **Direct labeling** wherever a legend would force the reader to glance back and forth.
- **Right chart for the data** — bars for category comparison, lines for time,
  treemap for composition, choropleth for geography, scatter for relationships.
- **Scorecards** at the top of every dashboard to anchor the headline numbers
  before the audience sees the breakdown.

### Technical stack

- **Streamlit** — multi-page UI with horizontal navigation
- **pandas / numpy** — data handling
- **plotly** — interactive visualizations
- **scikit-learn** — supervised + unsupervised models, cross-validation
- **statsmodels** — SARIMA time-series forecasting
- **gspread + google-auth** — Mode B Google Sheets API connection
""")

    with col2:
        st.markdown("""
<div class="kpi" style="border-left:3px solid #006A4E;">
  <div class="kpi-label">Author</div>
  <div class="kpi-value" style="font-size:18px;">Jad Assaf</div>
  <div class="kpi-sub">MSDA · LAU</div>
</div>
""", unsafe_allow_html=True)
        st.write("")
        st.markdown("""
<div class="kpi">
  <div class="kpi-label">Course</div>
  <div class="kpi-value" style="font-size:16px;">DAN614</div>
  <div class="kpi-sub">Advanced Data Visualization</div>
</div>
""", unsafe_allow_html=True)
        st.write("")
        st.markdown("""
<div class="kpi">
  <div class="kpi-label">Dataset</div>
  <div class="kpi-value" style="font-size:16px;">Superstore</div>
  <div class="kpi-sub">~10,000 transactions · 2017–2020</div>
</div>
""", unsafe_allow_html=True)

    section("Expected schema")
    st.dataframe(schema_summary_table(), use_container_width=True, hide_index=True,
                  height=420)

    section("Business questions answered")
    st.markdown("""
- **Executive Overview** — How is revenue trending? Which region drives profit?
   Where is the revenue mix concentrated?
- **Customer & Product** — Which sub-categories lose money? What share of orders
   are unprofitable? Who are the top customers?
- **Geo & Logistics** — Where does the business operate? Does shipping speed
   erode margin? Which states are net losses?
- **ML Lab** — What drives order profit? Can we flag a loss-making order
   before it ships? Who are our champion vs. at-risk customers? What's our
   six-month revenue forecast?
""")


# ═══════════════════════════════════════════════════════════════════════════
# Router
# ═══════════════════════════════════════════════════════════════════════════
ROUTES = {
    "Data Source":        render_data_source,
    "Data Cleaning":      render_data_cleaning,
    "Executive Overview": render_executive,
    "Customer & Product": render_customer_product,
    "Geo & Logistics":    render_geo,
    "ML Lab":             render_ml_lab,
    "About":              render_about,
}
ROUTES[selected]()
