# make_yearly_rates.py
# Build yearly crime rates (/1,000) from monthly results.
# Rule: treat monthly N/A as 0 when summing; BUT if all 12 months are N/A for a (Zip, Year),
# the yearly value is "N/A".
# Input: Merged_PD_SO_Rate_All.csv (Month, Zipcode, Crime rate per 1000)
# Output: Merged_PD_SO_Rate_Yearly.csv (Year, Zipcode, Crime rate per 1000)

import os
import re
import sys
import numpy as np
import pandas as pd

# -------- CONFIG --------
RATE_DIR = r"C:\Users\16377\Downloads\RATE"
IN_FILE  = os.path.join(RATE_DIR, "Merged_PD_SO_Rate_All.csv")  # you can switch to PD_Rate_All.csv or SO_Rate_All.csv if needed
OUT_FILE = os.path.join(RATE_DIR, "Merged_PD_SO_Rate_Yearly.csv")
YEAR_MIN, YEAR_MAX = 2018, 2023

# -------- HELPERS --------
def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize to: Month, Zipcode, Crime rate per 1000."""
    colmap = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl == "month":
            colmap[c] = "Month"
        elif cl in {"zipcode", "zip", "zip_code", "zcta"}:
            colmap[c] = "Zipcode"
        elif cl in {"crime rate per 1000", "crime_rate_per_1000", "crimerateper1000"}:
            colmap[c] = "Crime rate per 1000"
    df = df.rename(columns=colmap)
    needed = ["Month", "Zipcode", "Crime rate per 1000"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing}. Columns present: {list(df.columns)}")
    return df[needed].copy()

def parse_month_to_year(s: str):
    """'MM/YYYY' -> int year; returns None if not matched."""
    s = str(s).strip()
    m = re.match(r"^\s*(\d{1,2})\s*/\s*(\d{4})\s*$", s)
    if not m:
        return None
    return int(m.group(2))

def main():
    print(">>> Build yearly rates from monthly")
    print(f"[INFO] INPUT  : {IN_FILE}")
    print(f"[INFO] OUTPUT : {OUT_FILE}")

    df = normalize_cols(load_csv(IN_FILE))

    # Normalize ZIP to 5 digits and drop invalid
    df["Zipcode"] = df["Zipcode"].astype(str).str.extract(r"(\d{5})", expand=False)
    df = df.dropna(subset=["Zipcode"]).copy()

    # Extract year and filter to 2018–2023
    df["Year"] = df["Month"].apply(parse_month_to_year)
    df = df[df["Year"].between(YEAR_MIN, YEAR_MAX, inclusive="both")].copy()

    # Convert monthly rate to numeric; keep NaN for N/A
    rate_num = pd.to_numeric(df["Crime rate per 1000"], errors="coerce")

    # Group by (Zip, Year):
    # Use sum(min_count=1) so that "all-NaN" returns NaN; otherwise it sums available numbers (N/A treated as 0).
    grp = df.assign(_num=rate_num).groupby(["Zipcode", "Year"], dropna=False)["_num"]
    yearly_sum = grp.sum(min_count=1).reset_index(name="Crime rate per 1000")

    # Format: if NaN (meaning all months were N/A), keep "N/A"; else round to 4 decimals
    mask_nan = yearly_sum["Crime rate per 1000"].isna()
    yearly_sum.loc[~mask_nan, "Crime rate per 1000"] = np.around(yearly_sum.loc[~mask_nan, "Crime rate per 1000"].astype(float), 4)
    yearly_sum.loc[mask_nan, "Crime rate per 1000"] = "N/A"

    # Sort by Zip ascending (numeric) then Year ascending
    yearly_sum["_zip_int"] = yearly_sum["Zipcode"].astype(int)
    yearly_sum = yearly_sum.sort_values(["_zip_int", "Year"]).drop(columns=["_zip_int"])

    # Save
    yearly_sum.to_csv(OUT_FILE, index=False, encoding="utf-8-sig", lineterminator="\n")
    print(f"[OK] Yearly rows: {len(yearly_sum)}")
    print(">>> DONE")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(">>> ERROR")
        print(e)
        sys.exit(1)
