1\) PD\_Rate\_All.csv



City Police Department (PD) rates replicated to all ZIPs covered by each PD.



Rows: One row per Month (MM/YYYY) × Zipcode.



Computation: For each PD and month, the PD’s monthly rate is computed using city-level annual population and then assigned to all ZIPs in that PD.



Columns:



Month — MM/YYYY (2018–2023)



Zipcode — 5-digit ZIP



Crime rate per 1000 — numeric or N/A (if month has no usable data)



Sorting: ZIP ascending, then by year/month ascending.



2\) SO\_Rate\_All.csv



County Sheriff’s Office (SO) rates replicated to all ZIPs within each county.



Rows: One row per Month (MM/YYYY) × Zipcode.



Computation: For each county and month, the SO’s monthly rate is computed using county-level annual population and then assigned to all ZIPs in that county.



Columns:



Month — MM/YYYY (2018–2023)



Zipcode — 5-digit ZIP



Crime rate per 1000 — numeric or N/A



Sorting: ZIP ascending, then by year/month ascending.



3\) Merged\_PD\_SO\_Rate\_All.csv



Union of PD and SO outputs with per-key summation.



Key: (Month, Zipcode)



Merge logic:



Sum PD and SO rates for the same key; treat N/A as 0.



If both sides are N/A for a key, the merged value is N/A.



Internal duplicates within each file (same key) are resolved first using the same rule before cross-file merging.



Columns:



Month — MM/YYYY (2018–2023)



Zipcode — 5-digit ZIP



Crime rate per 1000 — numeric or N/A



Sorting: ZIP ascending, then by year/month ascending.



De-duplication: No duplicate (Month, Zipcode) rows.

