"""
Update Google Sheet — fill missing input features, then compute PD, Credit Score, SHAP.

For rows with incomplete data (missing MonthlyIncome, Dependents, etc.), this script:
  1. Applies the same imputation/clipping pipeline used during training
  2. Writes the cleaned feature values back to the input columns
  3. Computes and writes PD, CIBIL score, SHAP values to the output columns

Usage:
    python update_sheet.py
    python update_sheet.py --dry-run
    python update_sheet.py --creds path/to/creds.json --sheet-id <ID> --gid 398026000
"""

import argparse
import sys
from pathlib import Path

import gspread
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from predict import (
    COL_MAP,
    MEDIAN_INCOME,
    _feature_cols,
    _FEATURE_LABELS,
    _model,
    _scaler,
    _explainer,
    _thresh_f1,
    _thresh_prec,
    _IMPROVEMENT_TIPS,
    predict,
    cibil_score,
    explain,
)

DEFAULT_SHEET_ID = "1uTb3CsFbyO6TKJBkXLxNNmjhGep8NPgozKgf2GHwbBg"
DEFAULT_GID = 398026000
DEFAULT_CREDS = SCRIPT_DIR.parent / "secrets" / "gsheet_credentials.json"

# Reverse map: internal name → sheet column name
INTERNAL_TO_SHEET = {v: k for k, v in COL_MAP.items()}

# SHAP column headers
SHAP_HEADERS = [f"SHAP_{_FEATURE_LABELS.get(c, c)}" for c in _feature_cols]

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


# ---------------------------------------------------------------------------
# Impute / clean raw inputs (same logic as predict.py preprocess)
# ---------------------------------------------------------------------------
def clean_raw(raw: dict) -> dict:
    """Apply imputation, clipping, and derive flags. Returns cleaned dict with
    both the 10 input features (using internal names) and the 3 derived features."""
    monthly_income = raw.get("monthly_income")
    dependents = raw.get("dependents")

    monthly_income_missing = int(monthly_income is None)
    dependents_missing = int(dependents is None)

    if monthly_income is None:
        monthly_income = MEDIAN_INCOME
    if dependents is None:
        dependents = 0.0

    delinq_90_raw = raw.get("delinq_90") or 0
    delinq_sentinel = int(delinq_90_raw >= 96)

    delinq_30_59 = min(raw.get("delinq_30_59") or 0, 10)
    delinq_60_89 = min(raw.get("delinq_60_89") or 0, 10)
    delinq_90 = min(delinq_90_raw, 10)

    return {
        "unsecured_credit": raw.get("unsecured_credit") or 0,
        "age": raw.get("age") or 0,
        "delinq_30_59": delinq_30_59,
        "debt_ratio": raw.get("debt_ratio") or 0,
        "monthly_income": monthly_income,
        "open_credit": raw.get("open_credit") or 0,
        "delinq_90": delinq_90,
        "real_estate_loans": raw.get("real_estate_loans") or 0,
        "delinq_60_89": delinq_60_89,
        "dependents": dependents,
        "monthly_income_missing": monthly_income_missing,
        "dependents_missing": dependents_missing,
        "delinq_sentinel": delinq_sentinel,
    }


# ---------------------------------------------------------------------------
# Sheet helpers
# ---------------------------------------------------------------------------
def get_worksheet(creds_path: str, sheet_id: str, gid: int) -> gspread.Worksheet:
    gc = gspread.service_account(filename=creds_path)
    spreadsheet = gc.open_by_key(sheet_id)
    for ws in spreadsheet.worksheets():
        if ws.id == gid:
            return ws
    return spreadsheet.sheet1


def row_to_raw(headers: list[str], row: list) -> dict:
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


def has_missing_inputs(raw: dict) -> bool:
    """Check if any of the 10 input features are missing/None."""
    return any(v is None for v in raw.values())


def build_input_cells(cleaned: dict, headers: list[str]) -> dict[int, object]:
    """Map cleaned internal values back to sheet column indices.
    Returns {col_index: value} for the 10 input features only."""
    updates = {}
    for internal_name in [
        "unsecured_credit", "age", "delinq_30_59", "debt_ratio",
        "monthly_income", "open_credit", "delinq_90",
        "real_estate_loans", "delinq_60_89", "dependents",
    ]:
        sheet_col = INTERNAL_TO_SHEET.get(internal_name)
        if sheet_col and sheet_col in headers:
            col_idx = headers.index(sheet_col)
            updates[col_idx] = cleaned[internal_name]
    return updates


def build_output_cells(result: dict) -> list:
    """Build output column values from a prediction result."""
    risk_drivers = result["shap_explanation"]["risk_drivers"]

    def driver_name(i):
        return risk_drivers[i]["feature"] if i < len(risk_drivers) else ""

    def driver_rec(i):
        if i < len(risk_drivers):
            return risk_drivers[i].get("recommendation") or ""
        return ""

    prediction_label = "DEFAULT" if result["prediction_f1"] else "NO DEFAULT"

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Update sheet: fill missing inputs, then predict PD/CIBIL/SHAP"
    )
    parser.add_argument("--creds", default=str(DEFAULT_CREDS))
    parser.add_argument("--sheet-id", default=DEFAULT_SHEET_ID)
    parser.add_argument("--gid", type=int, default=DEFAULT_GID)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print results without writing to sheet")
    args = parser.parse_args()

    if not args.dry_run and not Path(args.creds).exists():
        print(f"ERROR: Credentials file not found: {args.creds}", file=sys.stderr)
        sys.exit(1)

    # --- Read sheet ---
    if not args.dry_run:
        ws = get_worksheet(args.creds, args.sheet_id, args.gid)
        all_data = ws.get_all_values()
    else:
        import pandas as pd
        url = f"https://docs.google.com/spreadsheets/d/{args.sheet_id}/export?format=csv&gid={args.gid}"
        df = pd.read_csv(url, dtype=str)
        all_data = [df.columns.tolist()] + df.values.tolist()

    if len(all_data) < 2:
        print("Sheet has no data rows.")
        return

    headers = all_data[0]
    data_rows = all_data[1:]

    # Output column start index
    if "CreditScore" in headers:
        out_start_idx = headers.index("CreditScore")
    else:
        out_start_idx = len(headers)

    print(f"Processing {len(data_rows)} row(s)...\n")

    input_updates = []   # (row_num, col_idx, value)
    output_updates = []  # (row_num, output_cells)

    for i, row in enumerate(data_rows):
        row_num = i + 2  # 1-indexed, row 1 is header
        raw = row_to_raw(headers, row)

        was_incomplete = has_missing_inputs(raw)
        cleaned = clean_raw(raw)

        # Track which fields were filled
        filled_fields = []
        if was_incomplete:
            input_cells = build_input_cells(cleaned, headers)
            for col_idx, value in input_cells.items():
                # Only update cells that were actually empty
                original_val = row[col_idx] if col_idx < len(row) else ""
                if original_val == "" or original_val is None:
                    input_updates.append((row_num, col_idx + 1, value))  # gspread is 1-indexed
                    filled_fields.append(headers[col_idx])

        # Run prediction using the cleaned raw dict (with imputed values)
        result = predict(raw)
        output_cells = build_output_cells(result)
        output_updates.append((row_num, output_cells))

        status = "INCOMPLETE → filled" if filled_fields else "complete"
        print(f"  Row {row_num}: {status}")
        if filled_fields:
            print(f"           Filled: {', '.join(filled_fields)}")
        print(f"           CIBIL={result['cibil_score']} ({result['cibil_grade']}), "
              f"P(default)={result['default_probability']:.4f}")

    if args.dry_run:
        print(f"\n--- DRY RUN ---")
        print(f"Input cells to fill: {len(input_updates)}")
        for row_num, col_idx, value in input_updates:
            print(f"  Row {row_num}, Col {col_idx}: {value}")
        print(f"Output rows to write: {len(output_updates)}")
        return

    # --- Write input fixes (individual cells for missing values) ---
    if input_updates:
        print(f"\nFilling {len(input_updates)} missing input cell(s)...")
        # Batch into cell_list for efficiency
        cells_to_update = []
        for row_num, col_idx, value in input_updates:
            cells_to_update.append(gspread.Cell(row_num, col_idx, value))
        ws.update_cells(cells_to_update)

    # --- Write output headers ---
    header_start = gspread.utils.rowcol_to_a1(1, out_start_idx + 1)
    header_end = gspread.utils.rowcol_to_a1(1, out_start_idx + len(OUTPUT_HEADERS))
    ws.update(values=[OUTPUT_HEADERS], range_name=f"{header_start}:{header_end}")

    # --- Write output columns ---
    print(f"Writing predictions for {len(output_updates)} row(s)...")
    for row_num, output_cells in output_updates:
        start_cell = gspread.utils.rowcol_to_a1(row_num, out_start_idx + 1)
        end_cell = gspread.utils.rowcol_to_a1(row_num, out_start_idx + len(OUTPUT_HEADERS))
        ws.update(values=[output_cells], range_name=f"{start_cell}:{end_cell}")

    total_filled = len(input_updates)
    print(f"\nDone — processed {len(output_updates)} row(s), filled {total_filled} missing input cell(s).")


if __name__ == "__main__":
    main()
