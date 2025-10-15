Data sources (population \& ZIPs)



Population: U.S. Census Bureau — primarily ACS 1-year and ACS 5-year estimates (2018–2023), with a fallback to the 2020 Decennial (PL) table where needed.



ZIP code lists: Aggregated from zip-codes.com and Zippopotam.us (all ZIP types included—Standard, P.O. Box, Unique).



City overrides (when helpful for modeling consistency): A small Zillow-aligned ZIP list (e.g., Flagstaff) is used as an explicit override.



Scripts



make\_azpd\_monthly\_rate\_batch.py

Batch-processes city Police Department (PD) CSVs (2018–2023).

Uses annual city population; computes monthly crime rates per 1,000 and assigns the same rate to all ZIPs covered by that PD.

Output: PD\_Rate\_All.csv



make\_az\_sheriff\_rate.py

Batch-processes county Sheriff’s Office (SO) CSVs (2018–2023).

Uses annual county population; computes monthly crime rates per 1,000 and assigns the same rate to all ZIPs in that county.

Output: SO\_Rate\_All.csv



merge\_pd\_so\_rates.py

Merges PD \& SO outputs. For matching (Month, Zipcode), sums rates (treat N/A as 0; if both sides are N/A, keeps N/A), then sorts by ZIP and month.

Output: Merged\_PD\_SO\_Rate\_All.csv

