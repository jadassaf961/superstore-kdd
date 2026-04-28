"""
LAU Academic Theme — DAN614 Project
Color palette and Plotly template inspired by:
  • Cole Knaflic / SWD principles (preattentive attributes, decluttering)
  • LAU Adnan Kassar School of Business brand (deep green)

All charts use a SINGLE accent color (LAU green) on a NEUTRAL gray base —
this implements the preattentive-attribute principle: color is precious,
use it once, use it where the insight lives.
"""
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ── Color tokens ────────────────────────────────────────────────────────────
LAU_GREEN     = "#006A4E"   # primary accent — used SPARINGLY
LAU_GREEN_DK  = "#00533D"
LAU_GREEN_LT  = "#1F8A6E"
ACCENT_GOLD   = "#B8860B"   # for "warning" / negative deltas
ACCENT_RED    = "#C0392B"   # for losses / risk

# Neutral grays — Cole Knaflic palette
GRAY_900 = "#1A1A1A"
GRAY_700 = "#4A4A4A"
GRAY_500 = "#9CA3AF"
GRAY_300 = "#D1D5DB"
GRAY_100 = "#F3F4F6"
GRAY_50  = "#F9FAFB"
WHITE    = "#FFFFFF"

# Sequential palettes (when categories are unavoidable)
SEQUENTIAL_GREEN = ["#E6F1ED", "#B3D5C8", "#80B9A3", "#4D9D7E", "#1A8159", "#006A4E"]
CATEGORICAL = ["#006A4E", "#9CA3AF", "#4A4A4A", "#1F8A6E", "#B8860B", "#C0392B"]

# ── Plotly template ─────────────────────────────────────────────────────────
def build_plotly_template():
    """Custom Plotly template enforcing decluttering rules from the course."""
    tpl = go.layout.Template()
    tpl.layout = go.Layout(
        font=dict(family="Inter, system-ui, sans-serif", size=13, color=GRAY_700),
        title=dict(font=dict(size=16, color=GRAY_900, family="Inter"), x=0.0, xanchor="left"),
        paper_bgcolor=WHITE,
        plot_bgcolor=WHITE,
        xaxis=dict(
            showgrid=False,
            showline=True,
            linewidth=1,
            linecolor=GRAY_300,
            ticks="outside",
            tickcolor=GRAY_300,
            tickfont=dict(color=GRAY_700, size=12),
            title=dict(font=dict(color=GRAY_700, size=12)),
            zeroline=False,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=GRAY_100,
            gridwidth=1,
            showline=False,
            ticks="",
            tickfont=dict(color=GRAY_700, size=12),
            title=dict(font=dict(color=GRAY_700, size=12)),
            zeroline=False,
        ),
        colorway=CATEGORICAL,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=12, color=GRAY_700),
            bgcolor="rgba(0,0,0,0)",
        ),
        hoverlabel=dict(
            bgcolor=WHITE,
            font=dict(family="Inter", size=12, color=GRAY_900),
            bordercolor=GRAY_300,
        ),
    )
    return tpl

def apply_plotly_template():
    pio.templates["lau"] = build_plotly_template()
    pio.templates.default = "lau"

# ── Streamlit page config + CSS ─────────────────────────────────────────────
def inject_css():
    """Inject custom CSS for a clean, academic dashboard look."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"], .stMarkdown, .stText, p, span, div, label {
      font-family: 'Inter', system-ui, sans-serif !important;
    }

    /* Restore Material Symbols font for Streamlit icons (overridden by rule above) */
    [data-testid="stIconMaterial"] {
      font-family: 'Material Symbols Rounded' !important;
    }

    /* Hide Streamlit's default chrome */
    #MainMenu, footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }

    /* Reduce default padding so the horizontal menu sits up top */
    .block-container {
      padding-top: 1.2rem !important;
      padding-bottom: 2rem !important;
      max-width: 1400px;
    }

    /* ── Brand strip ──────────────────────────────────────────────────────── */
    .brand-strip {
      display: flex; align-items: center; justify-content: space-between;
      padding: 8px 0 14px 0;
      border-bottom: 1px solid #E5E7EB;
      margin-bottom: 8px;
    }
    .brand-left { display: flex; align-items: center; gap: 14px; }
    .brand-mark {
      width: 38px; height: 38px; border-radius: 9px;
      background: linear-gradient(135deg, #006A4E 0%, #00533D 100%);
      display: flex; align-items: center; justify-content: center;
      color: white; font-weight: 700; font-size: 16px;
      box-shadow: 0 2px 8px rgba(0,106,78,0.20);
      letter-spacing: -0.5px;
    }
    .brand-text h1 {
      margin: 0; font-size: 17px; font-weight: 700; color: #1A1A1A; letter-spacing: -0.2px;
    }
    .brand-text p {
      margin: 0; font-size: 11.5px; color: #6B7280; font-weight: 500;
      letter-spacing: 0.4px; text-transform: uppercase;
    }
    .brand-right {
      text-align: right; font-size: 11px; color: #9CA3AF;
      font-weight: 500; letter-spacing: 0.4px; text-transform: uppercase;
    }
    .brand-right strong { color: #4A4A4A; font-weight: 600; }

    /* ── Page title ───────────────────────────────────────────────────────── */
    .page-title {
      margin: 18px 0 4px 0;
      font-size: 26px; font-weight: 700; color: #1A1A1A;
      letter-spacing: -0.4px;
    }
    .page-sub {
      margin: 0 0 22px 0;
      font-size: 14px; color: #6B7280; font-weight: 400;
    }

    /* ── KPI scorecards ───────────────────────────────────────────────────── */
    .kpi {
      background: white;
      border: 1px solid #E5E7EB;
      border-radius: 10px;
      padding: 16px 18px;
      box-shadow: 0 1px 2px rgba(0,0,0,0.04);
      transition: border-color .15s, box-shadow .15s;
      height: 100%;
    }
    .kpi:hover { border-color: #006A4E; box-shadow: 0 2px 8px rgba(0,106,78,0.08); }
    .kpi-label {
      font-size: 11px; color: #6B7280; font-weight: 600;
      letter-spacing: 0.5px; text-transform: uppercase; margin-bottom: 8px;
    }
    .kpi-value {
      font-size: 26px; font-weight: 700; color: #1A1A1A;
      letter-spacing: -0.5px; line-height: 1.1;
    }
    .kpi-delta {
      font-size: 12px; font-weight: 600; margin-top: 6px;
    }
    .kpi-delta.up   { color: #006A4E; }
    .kpi-delta.down { color: #C0392B; }
    .kpi-delta.flat { color: #9CA3AF; }
    .kpi-sub {
      font-size: 11px; color: #9CA3AF; font-weight: 500; margin-top: 4px;
    }
    .kpi-accent { border-left: 3px solid #006A4E; }

    /* ── Section heads ────────────────────────────────────────────────────── */
    .sec-head {
      font-size: 15px; font-weight: 700; color: #1A1A1A;
      margin: 24px 0 4px 0; letter-spacing: -0.2px;
    }
    .sec-sub {
      font-size: 12.5px; color: #6B7280; margin: 0 0 14px 0;
    }

    /* ── Insight callout (the "so what") ──────────────────────────────────── */
    .insight {
      background: #F0F7F4;
      border-left: 3px solid #006A4E;
      border-radius: 0 8px 8px 0;
      padding: 11px 16px;
      margin: 12px 0;
      font-size: 13px; color: #1A1A1A; line-height: 1.5;
    }
    .insight strong { color: #006A4E; }

    /* ── Info / warning / success boxes ───────────────────────────────────── */
    .info-box {
      background: #F9FAFB; border: 1px solid #E5E7EB;
      border-radius: 8px; padding: 12px 16px;
      font-size: 13px; color: #4A4A4A; margin: 10px 0;
    }
    .warn-box {
      background: #FFFBEB; border: 1px solid #FDE68A;
      border-radius: 8px; padding: 12px 16px;
      font-size: 13px; color: #92400E; margin: 10px 0;
    }
    .success-box {
      background: #ECFDF5; border: 1px solid #A7F3D0;
      border-radius: 8px; padding: 12px 16px;
      font-size: 13px; color: #065F46; margin: 10px 0;
    }

    /* ── Streamlit widget overrides ───────────────────────────────────────── */
    .stButton > button {
      background: #006A4E; color: white; font-weight: 600;
      border: none; border-radius: 8px; padding: 8px 18px;
      transition: background .15s;
    }
    .stButton > button:hover { background: #00533D; }
    .stDownloadButton > button {
      background: white; color: #006A4E; font-weight: 600;
      border: 1px solid #006A4E; border-radius: 8px; padding: 8px 18px;
    }
    .stDownloadButton > button:hover { background: #F0F7F4; }

    .stTabs [data-baseweb="tab-list"] {
      gap: 4px; background: #F3F4F6; border-radius: 9px; padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
      border-radius: 7px; padding: 7px 14px; font-weight: 500;
      color: #6B7280; font-size: 13px;
    }
    .stTabs [aria-selected="true"] {
      background: white !important; color: #006A4E !important;
      box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }

    .stProgress > div > div { background: #006A4E; }

    /* DataFrame styling */
    [data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

    /* Sidebar */
    [data-testid="stSidebar"] {
      background: #F9FAFB; border-right: 1px solid #E5E7EB;
    }
    [data-testid="stSidebar"] .block-container { padding-top: 2rem; }
    </style>
    """, unsafe_allow_html=True)

def render_brand_strip(student_name: str = "Jad Assaf"):
    """Top brand strip — LAU + MSDA + course + student."""
    st.markdown(f"""
    <div class="brand-strip">
      <div class="brand-left">
        <div class="brand-mark">L</div>
        <div class="brand-text">
          <h1>Superstore KDD Analytics</h1>
          <p>LAU · Adnan Kassar School of Business</p>
        </div>
      </div>
      <div class="brand-right">
        <div><strong>DAN614</strong> · Data Visualization for Executives</div>
        <div>MSDA · {student_name}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

def kpi_card(label: str, value: str, delta: str = None, delta_dir: str = "flat",
             sub: str = None, accent: bool = False):
    """Render one KPI scorecard via raw HTML for full styling control."""
    accent_class = "kpi-accent" if accent else ""
    delta_html = ""
    if delta is not None:
        arrow = "▲" if delta_dir == "up" else ("▼" if delta_dir == "down" else "—")
        delta_html = f'<div class="kpi-delta {delta_dir}">{arrow} {delta}</div>'
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return f"""
    <div class="kpi {accent_class}">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      {delta_html}
      {sub_html}
    </div>
    """

def insight(text: str):
    """Render the 'so what' insight box under a chart."""
    st.markdown(f'<div class="insight">{text}</div>', unsafe_allow_html=True)

def page_title(title: str, subtitle: str = ""):
    st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="page-sub">{subtitle}</div>', unsafe_allow_html=True)

def section(title: str, subtitle: str = ""):
    st.markdown(f'<div class="sec-head">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="sec-sub">{subtitle}</div>', unsafe_allow_html=True)
