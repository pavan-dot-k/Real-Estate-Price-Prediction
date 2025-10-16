import pandas as pd
import re

files = [
    "ACSST5Y2018.S1903-Data.csv",
    "ACSST5Y2019.S1903-Data.csv",
    "ACSST5Y2020.S1903-Data.csv",
    "ACSST5Y2021.S1903-Data.csv",
    "ACSST5Y2022.S1903-Data.csv",
    "ACSST5Y2023.S1903-Data.csv"
]

merged_data = []

for file in files:
    year = re.search(r"20\d{2}", file).group(0)


    df = pd.read_csv(file, dtype=str, low_memory=False)
    df.columns = df.columns.str.strip()  # remove stray spaces in headers

    if "NAME" in df.columns:
        df["zipcode"] = df["NAME"].str.extract(r"ZCTA5\s*(\d{5})")
    elif "GEO_ID" in df.columns:
        df["zipcode"] = df["GEO_ID"].str.extract(r"(\d{5})")

    else:
        raise KeyError(f"Neither GEO_ID nor NAME column found in {file}")

    if "S1903_C03_001E" not in df.columns:
        raise KeyError(f"'S1903_C03_001E' not found in {file}")
    
    df = df[["zipcode", "S1903_C03_001E"]].copy()

    df["S1903_C03_001E"] = (
        df["S1903_C03_001E"]
        .str.replace(",", "", regex=False)
        .str.replace("+", "", regex=False)
        .str.strip()
    )
    df["median_income"] = pd.to_numeric(df["S1903_C03_001E"], errors="coerce")

    # ---- Add year ----
    df["year"] = year

    # ---- Drop invalid rows ----
    df = df.dropna(subset=["zipcode", "median_income"])
    
    print(f"✅ {year}: {len(df)} rows, {df['zipcode'].nunique()} ZIPs with data")

    merged_data.append(df[["zipcode", "year", "median_income"]])

# ---- Combine all years ----
final_df = pd.concat(merged_data, ignore_index=True)

final_df = final_df.groupby(["zipcode", "year"], as_index=False)["median_income"].mean()

# ---- Save merged dataset ----
final_df.to_csv("merged_median_income_2018_2023_clean_final.csv", index=False)

print("\nSaved cleaned merged file: 'merged_median_income_2018_2023_clean_final.csv'")
print(final_df.head(10))
