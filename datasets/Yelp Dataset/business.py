import pandas as pd
import json
from datetime import datetime
import re

# Step 1: Load business.json (map business_id to valid US postal_code)
business_data = []
with open("business.json", 'r', encoding='utf-8') as f:
    for line in f:
        entry = json.loads(line)
        raw_zip = str(entry.get("postal_code", "")).strip()
        
        # Only keep valid U.S. 5-digit ZIPs
        if re.fullmatch(r"\d{5}", raw_zip):
            business_data.append({
                "business_id": entry["business_id"],
                "postal_code": raw_zip.zfill(5)
            })

biz_df = pd.DataFrame(business_data)

# Step 2: Load review.json and get first review date for each business
review_dates = {}
with open("review.json", 'r', encoding='utf-8') as f:
    for line in f:
        entry = json.loads(line)
        b_id = entry["business_id"]
        date = entry["date"]
        if b_id not in review_dates or date < review_dates[b_id]:
            review_dates[b_id] = date

# Convert to DataFrame
review_df = pd.DataFrame([
    {"business_id": b_id, "first_review_date": date}
    for b_id, date in review_dates.items()
])

# Convert date and extract year
review_df["first_review_date"] = pd.to_datetime(review_df["first_review_date"], errors='coerce')
review_df = review_df.dropna(subset=["first_review_date"])  # remove bad dates
review_df["year"] = review_df["first_review_date"].dt.year

# Step 3: Merge with business ZIP info
merged_df = pd.merge(review_df, biz_df, on="business_id")

# Step 4: Group by ZIP and year and count businesses
business_count_by_year = merged_df.groupby(["postal_code", "year"]).size().reset_index(name="business_count")

# Step 5: Save output
business_count_by_year = business_count_by_year.sort_values(by=["postal_code", "year"])

business_count_by_year.to_csv("business_by_zip_year.csv", index=False)
print("✅ Saved as 'business_by_zip_year.csv'")
print(business_count_by_year.head())
