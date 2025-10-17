import pandas as pd

zhvi_df = pd.read_csv("zhvi.csv")  

zhvi_filtered = zhvi_df.loc[:, ['RegionName'] + [col for col in zhvi_df.columns if col.startswith(('2018', '2019', '2020', '2021', '2022', '2023'))]]

zhvi_long = zhvi_filtered.melt(id_vars="RegionName", var_name="Date", value_name="ZHVI")

zhvi_long.rename(columns={"RegionName": "ZIP"}, inplace=True)

zhvi_long["Date"] = pd.to_datetime(zhvi_long["Date"])

zhvi_long["Year"] = zhvi_long["Date"].dt.year
zhvi_long["Month"] = zhvi_long["Date"].dt.month

zhvi_long.sort_values(by=["ZIP", "Date"], inplace=True)

zhvi_long.to_csv("zhvi_2018_2023_monthly.csv", index=False)

print(zhvi_long.head())
