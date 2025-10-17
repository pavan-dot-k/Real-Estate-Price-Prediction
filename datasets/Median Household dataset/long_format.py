import pandas as pd

income_df = pd.read_csv("median_income_pivoted_2018_2023_final.csv", dtype={"zipcode": str})
income_df["zipcode"] = income_df["zipcode"].str.zfill(5)

income_long = income_df.melt(
    id_vars="zipcode",             
    var_name="Year",                 
    value_name="Income"
)

income_long.rename(columns={"zipcode": "ZIP"}, inplace=True)
income_long["Year"] = income_long["Year"].astype(str)

income_long.to_csv("median_income_2018_2023_rowwise.csv", index=False)

print(income_long.head())
