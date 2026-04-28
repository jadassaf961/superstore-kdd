"""
Mode B — Google Sheets API connection via service account.

The user provides:
  • service-account JSON credentials (uploaded via Streamlit file_uploader)
  • Sheet ID or full URL
  • Worksheet/tab name (defaults to first sheet)

We use gspread + google-auth (no heavy gcloud SDK).
Returns a pandas DataFrame ready for downstream validation.
"""
import io
import json
import re
from datetime import datetime
import pandas as pd

# Lazy imports so the rest of the app still runs if these aren't installed yet
def _load_gspread():
    import gspread
    from google.oauth2.service_account import Credentials
    return gspread, Credentials


SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
]


def extract_sheet_id(url_or_id: str) -> str:
    """Accept either a full URL or a bare sheet ID."""
    s = url_or_id.strip()
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", s)
    if m:
        return m.group(1)
    return s


def load_sheet(creds_dict: dict, url_or_id: str, worksheet: str = None) -> tuple:
    """
    Connect, load the worksheet, return (DataFrame, metadata_dict).
    """
    gspread, Credentials = _load_gspread()

    creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
    gc = gspread.authorize(creds)

    sheet_id = extract_sheet_id(url_or_id)
    sh = gc.open_by_key(sheet_id)

    ws = sh.worksheet(worksheet) if worksheet else sh.sheet1
    records = ws.get_all_records()
    df = pd.DataFrame(records)

    # Metadata
    meta = {
        "spreadsheet_title": sh.title,
        "worksheet_title":   ws.title,
        "row_count":         len(df),
        "col_count":         len(df.columns),
        "sheet_id":          sheet_id,
        "worksheet_count":   len(sh.worksheets()),
        "loaded_at":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "url":               sh.url,
    }
    try:
        # spreadsheet metadata via Drive API (timestamp)
        meta["last_updated"] = sh.lastUpdateTime
    except Exception:
        meta["last_updated"] = "unknown"

    return df, meta


def list_worksheets(creds_dict: dict, url_or_id: str) -> list:
    """Return list of worksheet/tab names."""
    gspread, Credentials = _load_gspread()
    creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(extract_sheet_id(url_or_id))
    return [w.title for w in sh.worksheets()]


def parse_creds_upload(file_bytes: bytes) -> dict:
    """Parse uploaded JSON credentials, raise ValueError if malformed."""
    try:
        data = json.loads(file_bytes.decode("utf-8"))
    except Exception as e:
        raise ValueError(f"Could not parse credentials JSON: {e}")

    required = {"type", "project_id", "private_key", "client_email"}
    missing = required - set(data.keys())
    if missing:
        raise ValueError(f"Credentials JSON missing fields: {sorted(missing)}")
    if data.get("type") != "service_account":
        raise ValueError(f"Expected 'service_account' credentials, got '{data.get('type')}'")
    return data
