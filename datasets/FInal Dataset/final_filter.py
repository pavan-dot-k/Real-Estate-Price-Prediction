# build_training_dataset_monthly.py
# Merge four sources by (month/year, zip), keeping ONLY keys present in the monthly crime-rate file.
# Output column order: month/year, zip, median_income, crime_rate_per_1000, business_count, ZHVI
# Missing data in joined sources is written as "N/A" (crime_rate_per_1000 preserves its existing values incl. "N/A").

import os
import re
import numpy as np
import pandas as pd

# ---------------- CONFIG ----------------
BASE = r"C:\Users\16377\Downloads\Final_work"

# Inputs
CRIME_FILE   = os.path.join(BASE, "Merged_PD_SO_Rate_All.csv")                          # columns: Month, Zipcode, Crime rate per 1000
INCOME_FILE  = os.path.join(BASE, "median_income_pivoted_2018_2023_final_no_space.csv") # wide: zipcode + 2018..2023
BIZ_FILE     = os.path.join(BASE, "business_by_zip_year_final.csv")                     # long: postal_code, year, business_count
ZILLOW_FILE  = os.path.join(BASE, "zhvi_2018_2023_monthly.csv")                         # long: ZIP, Year, Month, ZHVI (or ZHCVI)

# Output
OUT_FILE     = os.path.join(BASE, "dataset_for_training.csv")

YEAR_MIN, YEAR_MAX = 2018, 2023
YEARS = [str(y) for y in range(YEAR_MIN, YEAR_MAX + 1)]

# ------------- HELPERS -------------
def to_5digit_zip(series: pd.Series) -> pd.Series:
    """Extract the first 5 digits from any ZIP-like string and keep leading zeros as a string."""
    return series.astype(str).str.extract(r"(\d{5})", expand=False)

def make_month_str(year: int, month: int) -> str:
    """Return 'MM/YYYY' from (year, month)."""
    return f"{int(month):02d}/{int(year)}"

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace around column names."""
    return df.rename(columns=lambda c: str(c).strip())

# ------------- LOAD & NORMALIZE -------------
print(">>> Building monthly training dataset by (month/year, zip)")

# ----- Crime (master keys: Month 'MM/YYYY', Zipcode) -----
crime_raw = normalize_cols(pd.read_csv(CRIME_FILE, encoding="utf-8-sig"))

crime_zip_col = next((c for c in crime_raw.columns if c.lower() in {"zipcode","zip","zip_code","zcta"}), None)
crime_month_col = next((c for c in crime_raw.columns if c.lower() == "month"), None)
crime_rate_col = next(
    (c for c in crime_raw.columns if c.lower().replace(" ", "") in {"crimerateper1000","crime_rate_per_1000"}),
    None
)
if not (crime_zip_col and crime_month_col and crime_rate_col):
    raise ValueError(f"[CRIME] Missing required columns. Columns: {list(crime_raw.columns)}")

# Keep only the needed columns and standardize names
crime = crime_raw[[crime_month_col, crime_zip_col, crime_rate_col]].copy()
crime.rename(columns={
    crime_month_col: "month_year",
    crime_zip_col: "zip",
    crime_rate_col: "crime_rate_per_1000"
}, inplace=True)

# Ensure Month is standardized as MM/YYYY and within 2018–2023
def parse_and_format_month(s: str) -> str | None:
    m = re.match(r"^\s*(\d{1,2})\s*/\s*(\d{4})\s*$", str(s).strip())
    if not m: 
        return None
    mm, yy = int(m.group(1)), int(m.group(2))
    if yy < YEAR_MIN or yy > YEAR_MAX or mm < 1 or mm > 12:
        return None
    return f"{mm:02d}/{yy}"

crime["month_year"] = crime["month_year"].apply(parse_and_format_month)
crime["zip"] = to_5digit_zip(crime["zip"])
crime = crime.dropna(subset=["month_year","zip"]).copy()
crime = crime[crime["month_year"].str[-4:].astype(int).between(YEAR_MIN, YEAR_MAX)]

# Preserve crime_rate_per_1000 as numeric when possible; keep literal 'N/A' as 'N/A'
cr_num = pd.to_numeric(crime["crime_rate_per_1000"], errors="coerce")
is_na_like = crime["crime_rate_per_1000"].astype(str).str.strip().str.upper().eq("N/A") | cr_num.isna()
crime.loc[~is_na_like, "crime_rate_per_1000"] = cr_num[~is_na_like].astype(float)
crime.loc[is_na_like,  "crime_rate_per_1000"] = "N/A"

# Base frame (left-join target): only keys that exist in the crime file
df = crime[["month_year","zip","crime_rate_per_1000"]].copy()

# ----- Median income (wide → annual long → monthly expand) -----
inc_raw = normalize_cols(pd.read_csv(INCOME_FILE, encoding="utf-8-sig"))
inc_zip_col = next((c for c in inc_raw.columns if c.lower() in {"zipcode","zip","zcta","zip_code"}), None)
if not inc_zip_col:
    raise ValueError(f"[INCOME] No zipcode-like column. Columns: {list(inc_raw.columns)}")

# Ensure the 2018–2023 columns exist (create empty if missing)
for y in YEARS:
    if y not in inc_raw.columns:
        inc_raw[y] = np.nan

inc_raw[inc_zip_col] = to_5digit_zip(inc_raw[inc_zip_col])
inc_raw = inc_raw.dropna(subset=[inc_zip_col]).copy()

# Melt to long (zip, year, median_income)
income_long = (
    inc_raw[[inc_zip_col] + YEARS]
    .melt(id_vars=[inc_zip_col], value_vars=YEARS, var_name="year", value_name="median_income")
    .rename(columns={inc_zip_col: "zip"})
)
income_long["year"] = pd.to_numeric(income_long["year"], errors="coerce").astype("Int64")

# Convert to numeric when possible (NaN otherwise); then expand to months
inc_num = pd.to_numeric(income_long["median_income"], errors="coerce")
income_long["median_income"] = inc_num

months = pd.DataFrame({"m": list(range(1, 13))})
income_monthly = (
    income_long.dropna(subset=["zip","year"])
              .merge(months, how="cross")
)
income_monthly["month_year"] = income_monthly.apply(lambda r: make_month_str(int(r["year"]), int(r["m"])), axis=1)
income_monthly = income_monthly[["month_year","zip","median_income"]]

# ----- Business count (annual long → monthly expand) -----
biz_raw = normalize_cols(pd.read_csv(BIZ_FILE, encoding="utf-8-sig"))
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
biz = biz[biz["year"].between(YEAR_MIN, YEAR_MAX)].copy()

# Convert to numeric when possible; then expand to months
biz_num = pd.to_numeric(biz["business_count"], errors="coerce")
biz["business_count"] = biz_num

biz_monthly = biz.merge(months, how="cross")
biz_monthly["month_year"] = biz_monthly.apply(lambda r: make_month_str(int(r["year"]), int(r["m"])), axis=1)
biz_monthly = biz_monthly[["month_year","zip","business_count"]]

# ----- ZHVI (monthly long) -----
zil_raw = normalize_cols(pd.read_csv(ZILLOW_FILE, encoding="utf-8-sig"))
z_zip_col = next((c for c in zil_raw.columns if c.lower() in {"zip","zipcode","zcta","zip_code","regionname","rregionname"}), None)
z_year_col = next((c for c in zil_raw.columns if c.lower() == "year"), None)
z_month_col = next((c for c in zil_raw.columns if c.lower() == "month"), None)
# value column may be ZHVI or ZHCVI
z_value_col = next((c for c in zil_raw.columns if c.upper() in {"ZHVI","ZHCVI"}), None)

if not (z_zip_col and z_year_col and z_month_col and z_value_col):
    raise ValueError(f"[ZILLOW] Missing columns. Columns: {list(zil_raw.columns)}")

zillow = zil_raw[[z_zip_col, z_year_col, z_month_col, z_value_col]].copy()
zillow.rename(columns={z_zip_col:"zip", z_year_col:"year", z_month_col:"month", z_value_col:"ZHVI"}, inplace=True)
zillow["zip"] = to_5digit_zip(zillow["zip"])
zillow = zillow.dropna(subset=["zip"]).copy()
zillow["year"]  = pd.to_numeric(zillow["year"], errors="coerce").astype("Int64")
zillow["month"] = pd.to_numeric(zillow["month"], errors="coerce").astype("Int64")
zillow = zillow[zillow["year"].between(YEAR_MIN, YEAR_MAX) & zillow["month"].between(1,12)].copy()

# Convert to numeric and keep 2 decimals
zhvi_num = pd.to_numeric(zillow["ZHVI"], errors="coerce")
zillow["ZHVI"] = np.round(zhvi_num, 2)
zillow["month_year"] = zillow.apply(lambda r: make_month_str(int(r["year"]), int(r["month"])), axis=1)
zillow = zillow[["month_year","zip","ZHVI"]]

# ------------- MERGE (left joins on crime keys: month_year + zip) -------------
df = df.merge(income_monthly, on=["month_year","zip"], how="left")
df = df.merge(biz_monthly,    on=["month_year","zip"], how="left")
df = df.merge(zillow,         on=["month_year","zip"], how="left")

# ------------- Final formatting -------------
# Fill missing values from joined sources with "N/A"
for col in ["median_income","business_count","ZHVI"]:
    df[col] = df[col].where(pd.notna(df[col]), "N/A")

# Final column order and naming
df = df.rename(columns={"month_year":"month/year"})
df = df[["month/year","zip","median_income","crime_rate_per_1000","business_count","ZHVI"]]

# Sort by zip asc, then by year-month asc
def sort_key(s):
    m = re.match(r"^\s*(\d{2})/(\d{4})\s*$", str(s).strip())
    if not m:
        return (9999, 99)
    return (int(m.group(2)), int(m.group(1)))

sort_vals = df["month/year"].apply(sort_key)
df["_yy"] = [t[0] for t in sort_vals]
df["_mm"] = [t[1] for t in sort_vals]
df = df.sort_values(["zip","_yy","_mm"]).drop(columns=["_yy","_mm"]).reset_index(drop=True)

# Save
df.to_csv(OUT_FILE, index=False, encoding="utf-8-sig", lineterminator="\n")
print(f"[OK] Wrote {OUT_FILE} with {len(df)} rows.")
