# Superstore KDD Analytics — DAN614 Individual Project

**Author:** Jad Assaf · MSDA · LAU Adnan Kassar School of Business
**Course:** DAN614 — Data Visualization for Executives

A multi-page Streamlit application implementing the full Knowledge Discovery
in Databases pipeline on the Superstore retail dataset.

---

## What's inside

```
superstore_kdd/
├── app.py                 # main entry — horizontal menu router
├── utils/
│   ├── theme.py           # LAU color palette, Plotly template, CSS, KPI cards
│   ├── schema.py          # Superstore schema + validators
│   ├── cleaning.py        # 5 cleaning measures (imputation, outliers, formats, dates, categories)
│   ├── gsheets.py         # Mode B — Google Sheets service account loader
│   ├── ml_models.py       # supervised regression/classification + KMeans
│   └── forecasting.py     # SARIMA + seasonal-naive baseline (NO Prophet)
├── assets/
├── requirements.txt
└── README.md              # this file
```

## Quick start

```bash
# 1. (recommended) Make a clean virtual environment
python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`) and use the
horizontal menu at the top of the page to navigate.

## Two ways to load data

### Mode A — Local file
Drop a `.xls`, `.xlsx`, or `.csv` file. The app validates against the canonical
Superstore schema and shows a preview. There is also a **"inject demo nulls"**
toggle that adds ~2% random missing values to the numeric columns so the
imputation tools have visible work to do during a demo.

### Mode B — Google Sheets via service account
1. Create a Google Cloud service account, download the JSON key.
2. Share the target Google Sheet with the service account's `client_email`.
3. In the app, upload the JSON file and paste the sheet URL.
4. Select the worksheet (defaults to `Orders`).

The credentials file is held only in memory for the session — it is never
written to disk.

## Pages

1. **Data Source** — pick a load mode, see validation report, preview rows
2. **Data Cleaning** — five tabs of cleaning measures, with a running operations log
3. **Executive Overview** — revenue, profit, margin, YoY, regional and category mix
4. **Customer & Product** — segments, top customers, sub-category profitability
5. **Geo & Logistics** — US choropleth, ship modes, shipping-day distributions
6. **ML Lab** — four sub-tabs:
   - Profit regression (Linear / Random Forest / Gradient Boosting, 5-fold CV)
   - Loss-making classifier (Logistic / RF / GB, ROC-AUC, F1)
   - K-Means RFM customer segmentation with elbow + silhouette
   - SARIMA monthly revenue forecast with backtest
7. **About** — methodology, schema, business questions

## Design notes

Every chart applies the visual-design principles from the course:
- One accent color (LAU green `#006A4E`) used sparingly
- Decluttered axes, no chart borders, no secondary y-axes
- Direct labeling instead of legend boxes where possible
- Charts chosen for the question (bars for category, lines for time, treemap for composition, choropleth for geography, scatter for relationships)

## Troubleshooting

**`ModuleNotFoundError: streamlit_option_menu`**
Run `pip install -r requirements.txt` again — make sure the venv is activated.

**Mode B fails with "permission denied"**
The service account email must be added as a viewer on the target sheet.
Find the email inside the JSON file under `client_email`.

**SARIMA forecast is slow**
The first run fits a SARIMA(1,1,1)(1,1,1,12) — about 1–2 seconds on the
Superstore monthly series. Subsequent runs use cached state.
