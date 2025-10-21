# fill_missing_income.py
# Fill empty cells in wide year columns (2018–2023) with "N/A"
# and remove spaces from all column names (replace with underscores).

import os
import pandas as pd
import numpy as np

BASE_DIR = r"C:\Users\16377\Downloads\MedianHousehold"
IN_FILE  = os.path.join(BASE_DIR, "median_income_pivoted_2018_2023_final.csv")
OUT_FILE = os.path.join(BASE_DIR, "median_income_pivoted_2018_2023_final_no_space.csv")

YEAR_COLS = [str(y) for y in range(2018, 2024)]  # "2018"..."2023"

def main():
    print(">>> Clean Median Household Income (fill blanks with 'N/A', drop spaces in headers)")
    print(f"[INFO] Input : {IN_FILE}")
    print(f"[INFO] Output: {OUT_FILE}")

    if not os.path.exists(IN_FILE):
        raise FileNotFoundError(f"Input file not found: {IN_FILE}")

    # Read
    df = pd.read_csv(IN_FILE, encoding="utf-8-sig")

    # Remove spaces from column names (use underscores)
    df = df.rename(columns=lambda c: str(c).replace(" ", "_"))

    # Ensure all year columns exist (create if missing)
    for y in YEAR_COLS:
        if y not in df.columns:
            df[y] = np.nan

    # For each year column, fill NA/blank with "N/A"
    for y in YEAR_COLS:
        col = df[y].astype(str)
        # Treat "", " ", and "nan"/"NaN" as missing
        is_empty = col.str.strip().eq("") | col.str.lower().eq("nan")
        df.loc[is_empty, y] = "N/A"

    # Optional: order columns -> key columns first, then year columns
    # Heuristic key columns: RegionName/Zipcode/ZCTA if present
    preferred_keys = [c for c in ["RegionName","Zipcode","ZCTA","ZIP","State","City","County"] if c in df.columns]
    other_cols = [c for c in df.columns if c not in set(preferred_keys + YEAR_COLS)]
    ordered_cols = preferred_keys + YEAR_COLS + other_cols
    df = df[ordered_cols]

    # Save
    df.to_csv(OUT_FILE, index=False, encoding="utf-8-sig", lineterminator="\n")
    print(f"[OK] Rows: {len(df)}, Columns: {len(df.columns)}")
    print(">>> DONE")

if __name__ == "__main__":
    main()
