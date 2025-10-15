# merge_pd_so_rates.py
# Merge PD_Rate_All.csv and SO_Rate_All.csv by (Zipcode, Month)
# Sum "Crime rate per 1000" with rule: N/A counts as 0, BUT if both files
# have no numeric data for the same key, the final value is "N/A".
# Sort by Zipcode asc, then by Year->Month asc.

import os
import re
import sys
import numpy as np
import pandas as pd
from typing import Tuple

# -------- CONFIG --------
RATE_DIR  = r"C:\Users\16377\Downloads\RATE"
PD_FILE   = os.path.join(RATE_DIR, "PD_Rate_All.csv")
SO_FILE   = os.path.join(RATE_DIR, "SO_Rate_All.csv")
OUT_FILE  = os.path.join(RATE_DIR, "Merged_PD_SO_Rate_All.csv")
YEAR_MIN, YEAR_MAX = 2018, 2023

# -------- HELPERS --------
def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    # utf-8-sig 兼容你的导出
    return pd.read_csv(path, encoding="utf-8-sig")

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize to columns: Month, Zipcode, Crime rate per 1000
    """
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
    need = ["Month", "Zipcode", "Crime rate per 1000"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing}. Columns present: {list(df.columns)}")
    return df[need].copy()

def parse_month_tuple(s: str) -> Tuple[int, int]:
    """
    Expect "MM/YYYY" -> (year, month); if mismatch returns (None, None)
    """
    s = str(s).strip()
    m = re.match(r"^\s*(\d{1,2})\s*/\s*(\d{4})\s*$", s)
    if not m:
        return (None, None)
    mm = int(m.group(1)); yy = int(m.group(2))
    return (yy, mm)

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize zip, parse month, keep 2018–2023 and valid months."""
    # 5-digit zip
    df["Zipcode"] = df["Zipcode"].astype(str).str.extract(r"(\d{5})", expand=False)
    df = df.dropna(subset=["Zipcode"]).copy()

    # parse month to (year, month)
    parsed = df["Month"].apply(parse_month_tuple)
    df["_year"]  = [t[0] for t in parsed]
    df["_month"] = [t[1] for t in parsed]

    # filter year/month
    df = df[df["_year"].between(YEAR_MIN, YEAR_MAX, inclusive="both") &
            df["_month"].between(1, 12, inclusive="both")].copy()
    return df

def agg_one_source(df: pd.DataFrame, src_label: str) -> pd.DataFrame:
    """
    Aggregate a single source (PD or SO) first:
    - Sum numeric rates within the file per (Zipcode, Month)
    - Track whether any numeric existed (has_num)
    """
    # numeric with N/A->NaN
    rate_num = pd.to_numeric(df["Crime rate per 1000"], errors="coerce")
    df["_num"] = rate_num

    # has numeric flag per group
    grp = df.groupby(["Zipcode", "_year", "_month", "Month"], dropna=False)
    has_num = grp["_num"].apply(lambda s: s.notna().any()).reset_index(name=f"{src_label}_has_num")

    # sum numeric treating NaN as 0 inside the sum; but if all NaN, sum should be 0 here,
    # and has_num flag will be False to distinguish later
    sum_num = grp["_num"].sum(min_count=1).fillna(0.0).reset_index(name=f"{src_label}_sum")

    out = has_num.merge(sum_num, on=["Zipcode","_year","_month","Month"], how="left")
    return out

# -------- MAIN --------
def main():
    print(">>> Merge PD & SO monthly rates with NA rule")
    print(f"[INFO] INPUT 1: {PD_FILE}")
    print(f"[INFO] INPUT 2: {SO_FILE}")
    print(f"[INFO] OUTPUT : {OUT_FILE}")

    # load & normalize
    pd_df = normalize_cols(load_csv(PD_FILE))
    so_df = normalize_cols(load_csv(SO_FILE))

    # preprocess (zip parse, month parse, filter years)
    pd_df = preprocess(pd_df)
    so_df = preprocess(so_df)

    # aggregate within each file first
    pd_agg = agg_one_source(pd_df, "pd")
    so_agg = agg_one_source(so_df, "so")

    # full outer join on keys
    keys = ["Zipcode","_year","_month","Month"]
    merged = pd.merge(pd_agg, so_agg, on=keys, how="outer")

    # fill flags and sums for missing sides
    for c in ["pd_has_num","so_has_num"]:
        if c not in merged:
            merged[c] = False
    for c in ["pd_sum","so_sum"]:
        if c not in merged:
            merged[c] = 0.0
    merged["pd_has_num"] = merged["pd_has_num"].fillna(False)
    merged["so_has_num"] = merged["so_has_num"].fillna(False)
    merged["pd_sum"]     = merged["pd_sum"].fillna(0.0).astype(float)
    merged["so_sum"]     = merged["so_sum"].fillna(0.0).astype(float)

    # final rule:
    # - if neither side has numeric -> "N/A"
    # - else final = pd_sum + so_sum (N/A treated as 0 already)
    neither = (~merged["pd_has_num"]) & (~merged["so_has_num"])
    merged["Crime rate per 1000"] = np.where(
        neither,
        "N/A",
        np.around(merged["pd_sum"] + merged["so_sum"], 4)
    )

    # sort: by zip asc (numeric), then year, month
    merged["_zip_int"] = merged["Zipcode"].astype(int)
    merged = merged.sort_values(["_zip_int","_year","_month"]).copy()

    out = merged[["Month","Zipcode","Crime rate per 1000"]]
    out.to_csv(OUT_FILE, index=False, encoding="utf-8-sig", lineterminator="\n")
    print(f"[OK] Merged rows: {len(out)}")
    print(">>> DONE")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(">>> ERROR")
        print(e)
        sys.exit(1)
