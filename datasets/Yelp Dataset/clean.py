import pandas as pd

# Input and output file paths
input_path = r"C:\Users\16377\Downloads\YelpData\business_by_zip_year.csv"
output_path = r"C:\Users\16377\Downloads\YelpData\business_by_zip_year_final.csv"

# Read the CSV file
df = pd.read_csv(input_path)

# Keep only rows where 'year' is between 2018 and 2023 (inclusive)
# If the column name is not exactly 'year', change it below
df_filtered = df[(df['year'] >= 2018) & (df['year'] <= 2023)]

# Save the filtered data to a new file
df_filtered.to_csv(output_path, index=False)

print("Filtering complete. The final file has been saved as:", output_path)
