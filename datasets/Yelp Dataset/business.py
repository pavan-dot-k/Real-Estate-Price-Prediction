import pandas as pd
import json
from datetime import datetime
import re

business_data = []
with open("business.json", 'r', encoding='utf-8') as f:
    for line in f:
        entry = json.loads(line)
        raw_zip = str(entry.get("postal_code", "")).strip()
        
        if re.fullmatch(r"\d{5}", raw_zip):
            business_data.append({
                "business_id": entry["business_id"],
                "postal_code": raw_zip.zfill(5)
            })

biz_df = pd.DataFrame(business_data)

review_dates = {}
with open("review.json", 'r', encoding='utf-8') as f:
    for line in f:
        entry = json.loads(line)
        b_id = entry["business_id"]
        date = entry["date"]
        if b_id not in review_dates or date < review_dates[b_id]:
            review_dates[b_id] = date

review_df = pd.DataFrame([
    {"business_id": b_id, "first_review_date": date}
    for b_id, date in review_dates.items()
])

review_df["first_review_date"] = pd.to_datetime(review_df["first_review_date"], errors='coerce')
review_df = review_df.dropna(subset=["first_review_date"])
review_df["year"] = review_df["first_review_date"].dt.year

merged_df = pd.merge(review_df, biz_df, on="business_id")

business_count_by_year = merged_df.groupby(["postal_code", "year"]).size().reset_index(name="business_count")

business_count_by_year = business_count_by_year.sort_values(by=["postal_code", "year"])

business_count_by_year.to_csv("business_by_zip_year.csv", index=False)
print("Saved as 'business_by_zip_year.csv'")
print(business_count_by_year.head())
