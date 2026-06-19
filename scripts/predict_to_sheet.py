"""
CatBoost Tuned — Predict & write results back to Google Sheet.

Reads borrower rows from the Google Sheet, runs the CatBoost model,
computes CIBIL score + SHAP explanations, and writes results back
to new columns in the same sheet.

Requires a Google Service Account JSON key file.
Setup:
  1. Create a service account at https://console.cloud.google.com/iam-admin/serviceaccounts
  2. Enable the Google Sheets API
  3. Download the JSON key → save as  secrets/gsheet_credentials.json
  4. Share the Google Sheet with the service account email (Editor access)

Usage:
    python predict_to_sheet.py
    python predict_to_sheet.py --creds path/to/creds.json
    python predict_to_sheet.py --sheet-id <ID> --gid 398026000
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import gspread
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from predict import (
    COL_MAP,
    _feature_cols,
    _FEATURE_LABELS,
    predict,
)

DEFAULT_SHEET_ID = "1uTb3CsFbyO6TKJBkXLxNNmjhGep8NPgozKgf2GHwbBg"
DEFAULT_GID = 398026000
DEFAULT_CREDS = SCRIPT_DIR.parent / "secrets" / "gsheet_credentials.json"

# SHAP column headers: "SHAP_<FeatureLabel>"
SHAP_HEADERS = [f"SHAP_{_FEATURE_LABELS.get(c, c)}" for c in _feature_cols]

# Output columns to write (after the last input column)
OUTPUT_HEADERS = [
    "CreditScore",
    "CIBILGrade",
    "DefaultProbability",
    "Prediction",
    "TopRiskDriver1",
    "TopRiskDriver2",
    "TopRiskDriver3",
    "Recommendation1",
    "Recommendation2",
    "Recommendation3",
] + SHAP_HEADERS


def get_worksheet(creds_path: str, sheet_id: str, gid: int) -> gspread.Worksheet:
    """Authenticate and return the worksheet matching the given gid."""
    gc = gspread.service_account(filename=creds_path)
    spreadsheet = gc.open_by_key(sheet_id)
    for ws in spreadsheet.worksheets():
        if ws.id == gid:
            return ws
    # Fallback to first sheet
    return spreadsheet.sheet1


def row_to_raw(headers: list[str], row: list) -> dict | None:
    """Convert a sheet row to the raw dict expected by predict()."""
    row_dict = {}
    for i, h in enumerate(headers):
        if i < len(row):
            row_dict[h] = row[i]
        else:
            row_dict[h] = ""

    raw = {}
    for sheet_col, internal_name in COL_MAP.items():
        val = row_dict.get(sheet_col, "")
        if val == "" or val is None:
            raw[internal_name] = None
        else:
            try:
                raw[internal_name] = float(val)
            except (ValueError, TypeError):
                raw[internal_name] = None
    return raw


def build_output_row(result: dict) -> list:
    """Build the output cell values from a prediction result."""
    risk_drivers = result["shap_explanation"]["risk_drivers"]

    def driver_name(i):
        return risk_drivers[i]["feature"] if i < len(risk_drivers) else ""

    def driver_rec(i):
        if i < len(risk_drivers):
            return risk_drivers[i].get("recommendation") or ""
        return ""

    prediction_label = "DEFAULT" if result["prediction_f1"] else "NO DEFAULT"

    # Build SHAP values in _feature_cols order
    # feature_contributions is sorted by |SHAP|, so index by feature name
    shap_by_feature = {
        item["feature"]: item["shap_value"]
        for item in result["shap_explanation"]["feature_contributions"]
    }
    shap_values = [round(shap_by_feature.get(col, 0.0), 6) for col in _feature_cols]

    return [
        result["cibil_score"],
        result["cibil_grade"],
        round(result["default_probability"], 6),
        prediction_label,
        driver_name(0),
        driver_name(1),
        driver_name(2),
        driver_rec(0),
        driver_rec(1),
        driver_rec(2),
    ] + shap_values


def main():
    parser = argparse.ArgumentParser(
        description="Run predictions and write results back to Google Sheet"
    )
    parser.add_argument("--creds", default=str(DEFAULT_CREDS),
                        help="Path to service account JSON key")
    parser.add_argument("--sheet-id", default=DEFAULT_SHEET_ID,
                        help="Google Sheet ID")
    parser.add_argument("--gid", type=int, default=DEFAULT_GID,
                        help="Sheet tab gid (default: 398026000)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print results without writing to sheet")
    args = parser.parse_args()

    if not args.dry_run and not Path(args.creds).exists():
        print(f"ERROR: Credentials file not found: {args.creds}", file=sys.stderr)
        print("See script docstring for setup instructions.", file=sys.stderr)
        sys.exit(1)

    # --- Connect to sheet ---
    if not args.dry_run:
        ws = get_worksheet(args.creds, args.sheet_id, args.gid)
        all_data = ws.get_all_values()
    else:
        # Dry run: read via public CSV. fillna("") so empty cells match the
        # live ws.get_all_values() path (which returns "" not NaN); otherwise
        # row_to_raw would treat NaN as a real value and mis-impute.
        import pandas as pd
        url = f"https://docs.google.com/spreadsheets/d/{args.sheet_id}/export?format=csv&gid={args.gid}"
        df = pd.read_csv(url, dtype=str).fillna("")
        all_data = [df.columns.tolist()] + df.values.tolist()

    if len(all_data) < 2:
        print("Sheet has no data rows.")
        return

    headers = all_data[0]
    data_rows = all_data[1:]

    # Find where input columns end (CreditScore column or after last input col)
    input_cols = list(COL_MAP.keys())
    # Determine the output start column
    if "CreditScore" in headers:
        out_start_idx = headers.index("CreditScore")
    else:
        out_start_idx = len(headers)

    results_to_write = []

    print(f"Processing {len(data_rows)} row(s)...")

    for i, row in enumerate(data_rows):
        raw = row_to_raw(headers, row)
        if raw is None:
            print(f"  Row {i + 2}: skipped (invalid data)")
            results_to_write.append([""] * len(OUTPUT_HEADERS))
            continue

        result = predict(raw)
        output_cells = build_output_row(result)
        results_to_write.append(output_cells)

        print(f"  Row {i + 2}: CIBIL={result['cibil_score']} "
              f"({result['cibil_grade']}), "
              f"P(default)={result['default_probability']:.4f}")

    if args.dry_run:
        print("\n--- DRY RUN (not writing to sheet) ---")
        print(f"Headers: {OUTPUT_HEADERS}")
        for i, row in enumerate(results_to_write):
            print(f"  Row {i + 2}: {row}")
        return

    # --- Write headers if needed ---
    # Update header row to include output columns
    header_range_start = gspread.utils.rowcol_to_a1(1, out_start_idx + 1)
    header_range_end = gspread.utils.rowcol_to_a1(1, out_start_idx + len(OUTPUT_HEADERS))
    ws.update(values=[OUTPUT_HEADERS], range_name=f"{header_range_start}:{header_range_end}")

    # --- Write results ---
    for i, output_cells in enumerate(results_to_write):
        row_num = i + 2  # sheet rows are 1-indexed, row 1 is header
        start_cell = gspread.utils.rowcol_to_a1(row_num, out_start_idx + 1)
        end_cell = gspread.utils.rowcol_to_a1(row_num, out_start_idx + len(OUTPUT_HEADERS))
        ws.update(values=[output_cells], range_name=f"{start_cell}:{end_cell}")

    print(f"\nDone — wrote {len(results_to_write)} row(s) to sheet.")


if __name__ == "__main__":
    main()
