import pandas as pd

df = pd.read_csv("merged_median_income_2018_2023_clean_final.csv", dtype={"zipcode": str, "year": str})

# ---- Clean column names ----
df.columns = df.columns.str.strip().str.lower()

# ---- Check what columns we have ----
print("Columns found:", df.columns.tolist())
print(df.head())

# ---- Remove rows missing ZIP or year ----
df = df.dropna(subset=["zipcode", "year"])

# ---- Remove duplicates: average income if multiple entries per (zipcode, year) ----
df = (
    df.groupby(["zipcode", "year"], as_index=False)["median_income"]
    .mean()
)

print(f"After deduplication: {len(df)} rows, {df['zipcode'].nunique()} unique ZIP codes")

# ---- Pivot: years as columns ----
pivoted = df.pivot(index="zipcode", columns="year", values="median_income").reset_index()

# ---- Sort by ZIP code ----
pivoted = pivoted.sort_values("zipcode", kind="stable")

# ---- Save pivoted dataset ----
pivoted.to_csv("median_income_pivoted_2018_2023_final.csv", index=False)

