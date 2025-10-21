# build_training_dataset.py
# Merge four sources into a training dataset keyed by (year, zip),
# keeping ONLY zips/years that appear in the crime-rate yearly file.

import os
import re
import numpy as np
import pandas as pd

# ---------------- CONFIG ----------------
BASE = r"C:\Users\16377\Downloads\Final_work"

# Inputs
CRIME_FILE   = os.path.join(BASE, "Merged_PD_SO_Rate_Yearly.csv")                        # Zipcode, Year, Crime rate per 1000
INCOME_FILE  = os.path.join(BASE, "median_income_pivoted_2018_2023_final_no_space.csv")  # wide: zipcode + 2018..2023
BIZ_FILE     = os.path.join(BASE, "business_by_zip_year_final.csv")                      # long: postal_code, year, business_count
ZILLOW_FILE  = os.path.join(BASE, "zillow_2018_2023_final.csv")                          # wide: RegionName/RRegionName + 2018..2023

# Output
OUT_FILE     = os.path.join(BASE, "dataset_for_training.csv")

YEAR_MIN, YEAR_MAX = 2018, 2023
YEARS = [str(y) for y in range(YEAR_MIN, YEAR_MAX + 1)]

# ------------- HELPERS -------------
def to_5digit_zip(series: pd.Series) -> pd.Series:
    """Extract first 5 digits and keep leading zeros as string."""
    return series.astype(str).str.extract(r"(\d{5})", expand=False)

def melt_wide(df: pd.DataFrame, zip_col: str, value_name: str) -> pd.DataFrame:
    """
    Melt a wide table with columns [zip_col, '2018',...,'2023'] -> long (zip, year, value).
    Ensures all year columns exist.
    """
    for y in YEARS:
        if y not in df.columns:
            df[y] = np.nan

    out = (
        df[[zip_col] + YEARS]
        .melt(id_vars=[zip_col], value_vars=YEARS, var_name="year", value_name=value_name)
    )
    out.rename(columns={zip_col: "zip"}, inplace=True)  # <-- unify to 'zip'
    # year as Int64
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    return out

def coerce_numeric_or_na(series: pd.Series):
    """Convert to numeric; non-numeric -> NaN."""
    return pd.to_numeric(series, errors="coerce")

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip spaces from column names."""
    return df.rename(columns=lambda c: str(c).strip())

# ------------- LOAD & NORMALIZE -------------
print(">>> Building training dataset")

# ----- Crime (master keys) -----
crime_raw = clean_columns(pd.read_csv(CRIME_FILE, encoding="utf-8-sig"))

crime_zip_col = next((c for c in crime_raw.columns if c.lower() in {"zipcode","zip","zip_code","zcta"}), None)
crime_year_col = next((c for c in crime_raw.columns if c.lower() == "year"), None)
crime_rate_col = next(
    (c for c in crime_raw.columns if c.lower().replace(" ", "") in {"crimerateper1000","crime_rate_per_1000"}),
    None
)

if not (crime_zip_col and crime_year_col and crime_rate_col):
    raise ValueError(f"[CRIME] Missing required columns. Columns: {list(crime_raw.columns)}")

crime = crime_raw[[crime_zip_col, crime_year_col, crime_rate_col]].copy()
crime.rename(columns={crime_zip_col: "zip", crime_year_col: "year", crime_rate_col: "crime_rate_per_1000"}, inplace=True)
crime["zip"] = to_5digit_zip(crime["zip"])
crime = crime.dropna(subset=["zip"]).copy()
crime["year"] = pd.to_numeric(crime["year"], errors="coerce").astype("Int64")
crime = crime[(crime["year"] >= YEAR_MIN) & (crime["year"] <= YEAR_MAX)].copy()

# keep "N/A" strings if present; numeric otherwise
crime_num = coerce_numeric_or_na(crime["crime_rate_per_1000"])
is_na_like = crime["crime_rate_per_1000"].astype(str).str.strip().str.upper().eq("N/A") | crime_num.isna()
crime.loc[~is_na_like, "crime_rate_per_1000"] = crime_num[~is_na_like].astype(float)
crime.loc[is_na_like,  "crime_rate_per_1000"] = "N/A"

# ----- Median income (wide -> long) -----
inc_raw = clean_columns(pd.read_csv(INCOME_FILE, encoding="utf-8-sig"))
inc_zip_col = next((c for c in inc_raw.columns if c.lower() in {"zipcode","zip","zcta","zip_code"}), None)
if not inc_zip_col:
    raise ValueError(f"[INCOME] No zipcode-like column. Columns: {list(inc_raw.columns)}")
inc_raw[inc_zip_col] = to_5digit_zip(inc_raw[inc_zip_col])
inc_raw = inc_raw.dropna(subset=[inc_zip_col]).copy()
income_long = melt_wide(inc_raw, zip_col=inc_zip_col, value_name="median_income")  # has 'zip','year','median_income'

inc_num = coerce_numeric_or_na(income_long["median_income"])
income_long.loc[inc_num.notna(), "median_income"] = inc_num[inc_num.notna()].astype(float)
income_long.loc[inc_num.isna(),  "median_income"] = "N/A"

# ----- Business count (long) -----
biz_raw = clean_columns(pd.read_csv(BIZ_FILE, encoding="utf-8-sig"))
biz_zip_col  = next((c for c in biz_raw.columns if c.lower() in {"postal_code","zipcode","zip","zcta","zip_code"}), None)
biz_year_col = next((c for c in biz_raw.columns if c.lower() == "year"), None)
biz_cnt_col  = next((c for c in biz_raw.columns if c.lower() in {"business_count","count","businesses"}), None)
if not (biz_zip_col and biz_year_col and biz_cnt_col):
    raise ValueError(f"[BUSINESS] Missing needed columns. Columns: {list(biz_raw.columns)}")
biz = biz_raw[[biz_zip_col, biz_year_col, biz_cnt_col]].copy()
biz.rename(columns={biz_zip_col:"zip", biz_year_col:"year", biz_cnt_col:"business_count"}, inplace=True)
biz["zip"]  = to_5digit_zip(biz["zip"])
biz = biz.dropna(subset=["zip"]).copy()
biz["year"] = pd.to_numeric(biz["year"], errors="coerce").astype("Int64")
biz = biz[(biz["year"] >= YEAR_MIN) & (biz["year"] <= YEAR_MAX)].copy()
biz_num = coerce_numeric_or_na(biz["business_count"])
biz.loc[biz_num.notna(), "business_count"] = biz_num[biz_num.notna()].astype(float)
biz.loc[biz_num.isna(),  "business_count"] = "N/A"

# ----- Zillow (wide -> long) -----
zil_raw = clean_columns(pd.read_csv(ZILLOW_FILE, encoding="utf-8-sig"))
z_zip_col = next((c for c in zil_raw.columns if c.lower() in {"regionname","rregionname","zipcode","zip","zcta","zip_code"}), None)
if not z_zip_col:
    raise ValueError(f"[ZILLOW] Missing RegionName/RRegionName/zip column. Columns: {list(zil_raw.columns)}")
zil_raw[z_zip_col] = to_5digit_zip(zil_raw[z_zip_col])
zil_raw = zil_raw.dropna(subset=[z_zip_col]).copy()
zillow_long = melt_wide(zil_raw, zip_col=z_zip_col, value_name="avg_house_price")  # has 'zip','year','avg_house_price'
zil_num = coerce_numeric_or_na(zillow_long["avg_house_price"])
zillow_long.loc[zil_num.notna(), "avg_house_price"] = np.round(zil_num[zil_num.notna()].astype(float), 2)
zillow_long.loc[zil_num.isna(),  "avg_house_price"] = "N/A"

# ------------- MERGE (left joins on the crime keys) -------------
base = crime[["year","zip","crime_rate_per_1000"]].copy()

# income
income_long["year"] = income_long["year"].astype("Int64")
base = base.merge(income_long[["zip","year","median_income"]], on=["zip","year"], how="left")

# business
base = base.merge(biz[["zip","year","business_count"]], on=["zip","year"], how="left")

# zillow
zillow_long["year"] = zillow_long["year"].astype("Int64")
base = base.merge(zillow_long[["zip","year","avg_house_price"]], on=["zip","year"], how="left")

# ------------- Final formatting -------------
# Fill missing from merges with "N/A"
for col in ["median_income","business_count","avg_house_price"]:
    base[col] = base[col].where(pd.notna(base[col]), "N/A")

# Types & order
base["year"] = base["year"].astype(int)
base["zip"] = base["zip"].astype(str).str.zfill(5)
base = base[["year","zip","median_income","crime_rate_per_1000","business_count","avg_house_price"]]
base = base.sort_values(["zip","year"]).reset_index(drop=True)

# Save
base.to_csv(OUT_FILE, index=False, encoding="utf-8-sig", lineterminator="\n")
print(f"[OK] Wrote {OUT_FILE} with {len(base)} rows.")
