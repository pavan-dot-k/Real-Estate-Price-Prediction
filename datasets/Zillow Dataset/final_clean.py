# final_clean.py
# Fill missing year data for 2018–2023.
# - Wide table (columns named '2018'...'2023'): ensure all year columns exist and fill NaN/empty with "N/A".
# - Long table (has 'Year' column): add missing (key, year) rows; non-key fields in new rows -> "N/A".

import os
import re
import sys
import pandas as pd
import numpy as np

BASE_DIR = r"C:\Users\16377\Downloads\Zillow"
IN_FILE  = os.path.join(BASE_DIR, "zhvi_2018_2023_clean.csv")
OUT_FILE = os.path.join(BASE_DIR, "zillow_2018_2023_final.csv")
YEARS = [str(y) for y in range(2018, 2024)]  # as strings for column matching

# Optional manual overrides (leave None to auto-detect)
KEY_COLS = None  # e.g., ["Zipcode"] or ["RegionName"]
YEAR_COL = None  # e.g., "Year"

def detect_key_cols(df: pd.DataFrame, year_col_name: str | None):
    if KEY_COLS is not None:
        for c in KEY_COLS:
            if c not in df.columns:
                raise ValueError(f"KEY_COL '{c}' not found. Columns: {list(df.columns)}")
        return KEY_COLS

    # Prefer Zipcode, then RegionName, else any id-like column
    for cand in [["Zipcode"], ["RegionName"], ["ZCTA"], ["ZIP"], ["zip"]]:
        if all(c in df.columns for c in cand):
            return cand
    id_like = [c for c in df.columns if c != year_col_name and re.search(r"(zip|region|id|name)", c, flags=re.I)]
    if id_like:
        return [id_like[0]]
    # As a last resort, use the first non-year column
    non_year_cols = [c for c in df.columns if c not in YEARS and c != year_col_name]
    if non_year_cols:
        return [non_year_cols[0]]
    raise ValueError("Could not infer key columns; please set KEY_COLS at the top.")

def to_long_year(val):
    """Parse 'MM/YYYY' or 'YYYY' to int year; return None if not matched."""
    s = str(val).strip()
    # pure year
    if re.fullmatch(r"\d{4}", s):
        return int(s)
    # MM/YYYY
    m = re.match(r"^\s*(\d{1,2})\s*/\s*(\d{4})\s*$", s)
    return int(m.group(2)) if m else None

def process_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure year columns exist and fill missing with 'N/A'."""
    # Add missing year columns
    for y in YEARS:
        if y not in df.columns:
            df[y] = np.nan

    # Fill only in the year columns (leave other columns untouched)
    for y in YEARS:
        df[y] = df[y].where(pd.notna(df[y]) & (df[y].astype(str).str.strip() != ""), "N/A")

    # Sort columns: key cols first, then 2018..2023, then others
    key_cols = detect_key_cols(df, year_col_name=None)
    ordered = key_cols + [y for y in YEARS] + [c for c in df.columns if c not in set(key_cols + YEARS)]
    df = df[ordered]
    return df

def process_long(df: pd.DataFrame, year_col: str) -> pd.DataFrame:
    """Add missing (key, year) rows 2018–2023; fill non-key fields with 'N/A'."""
    key_cols = detect_key_cols(df, year_col_name=year_col)

    # Normalize Year
    parsed_year = pd.to_numeric(df[year_col], errors="coerce")
    if parsed_year.isna().all():
        # maybe Month column like MM/YYYY
        parsed_year = df[year_col].apply(to_long_year)
    df = df[~pd.isna(parsed_year)].copy()
    df[year_col] = parsed_year.astype(int)

    # keep only target range
    df = df[df[year_col].between(2018, 2023, inclusive="both")].copy()

    # Build skeleton keys x YEARS
    keys_df = df[key_cols].drop_duplicates().reset_index(drop=True)
    years_df = pd.DataFrame({year_col: list(range(2018, 2024))})
    keys_df["_k"] = 1; years_df["_k"] = 1
    skeleton = keys_df.merge(years_df, on="_k").drop(columns="_k")

    merged = skeleton.merge(df, on=key_cols + [year_col], how="left")

    # Fill non-key, non-year columns with "N/A" where missing
    non_key_cols = [c for c in merged.columns if c not in key_cols + [year_col]]
    for c in non_key_cols:
        merged.loc[merged[c].isna() | (merged[c].astype(str).str.strip() == ""), c] = "N/A"

    # Sort and return
    sort_cols = key_cols + [year_col]
    merged = merged.sort_values(sort_cols).reset_index(drop=True)
    return merged

def main():
    print(">>> Fill missing years (2018–2023) for Zillow file")
    print(f"[INFO] Input : {IN_FILE}")
    print(f"[INFO] Output: {OUT_FILE}")

    if not os.path.exists(IN_FILE):
        raise FileNotFoundError(f"Input not found: {IN_FILE}")

    df = pd.read_csv(IN_FILE, encoding="utf-8-sig")

    # Detect wide vs long
    has_year_col = YEAR_COL in df.columns if YEAR_COL else any(c.lower() == "year" for c in df.columns)
    has_wide_year_cols = any(y in df.columns for y in YEARS)

    if not has_year_col and has_wide_year_cols:
        # WIDE MODE
        print("[INFO] Detected wide format (year columns 2018..2023).")
        out = process_wide(df)
    else:
        # LONG MODE
        # pick year col
        ycol = YEAR_COL if YEAR_COL in df.columns else next((c for c in df.columns if c.lower()=="year"), None)
        if not ycol:
            raise ValueError(f"Year column not found. Columns: {list(df.columns)}")
        print(f"[INFO] Detected long format with year column: {ycol}")
        out = process_long(df, ycol)

    out.to_csv(OUT_FILE, index=False, encoding="utf-8-sig", lineterminator="\n")
    print(f"[OK] Wrote: {OUT_FILE}  (rows: {len(out)})")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(">>> ERROR")
        print(e)
        sys.exit(1)
