import pandas as pd

# Load the Zillow dataset
file_path = "zhvi.csv"
df = pd.read_csv(file_path, low_memory=False)

# --- Keep only RegionName and yearly columns from 2018–2023 ---
# Identify columns dynamically (some files may have different column orders)
cols_to_keep = ["RegionName"]

# Add columns that start with years 2018–2023
for year in range(2018, 2024):
    year_cols = [col for col in df.columns if str(year) in col]
    cols_to_keep.extend(year_cols)

df_clean = df[cols_to_keep].copy()

# --- Drop rows with missing or invalid RegionName ---
df_clean = df_clean.dropna(subset=["RegionName"])
df_clean["RegionName"] = df_clean["RegionName"].astype(str).str.zfill(5)  # ensure 5-digit ZIPs

# --- Optional: Aggregate monthly data into yearly averages ---
df_yearly = df_clean.melt(id_vars="RegionName", var_name="date", value_name="zhvi")
df_yearly["year"] = df_yearly["date"].str.extract(r"(\d{4})").astype(int)
df_yearly = df_yearly.groupby(["RegionName", "year"])["zhvi"].mean().reset_index()

# --- Pivot so each year becomes a column ---
df_pivoted = df_yearly.pivot(index="RegionName", columns="year", values="zhvi").reset_index()

# --- Save cleaned dataset ---
df_pivoted.to_csv("zhvi_2018_2023_clean.csv", index=False)

print("Cleaned Zillow dataset saved as 'zhvi_zipcode_2018_2023_clean.csv'")
print(df_pivoted.head())
