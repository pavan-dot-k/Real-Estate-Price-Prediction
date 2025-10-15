# make_az_sheriff_rate.py
# v1.8-SO — Arizona Sheriff’s Offices (County) monthly crime rate (/1,000) for 2018–2023
# - Mirrors your PD v1.8 logic (no ZIP weighting). County-level population used as denominator.
# - Assign each year's rate to ALL ZIPs in the county (Standard/P.O. Box/Unique all included).
# - Robust monthly table parser; skips "Summary Date" line (Mesa/Phoenix/Tucson-style exports).
# - Auto-detect county name from header; typo/alias fixes (e.g., "Sherrif"→"Sheriff").
# - Fetch county ZIPs via zip-codes.com; fetch county population via Census API (ACS).

import os, io, re, sys, math, time, traceback, unicodedata
import requests, pandas as pd
from typing import List, Dict, Optional, Tuple

# ================== CONFIG ==================
INPUT_DIR  = r"C:\Users\16377\Downloads\AZSherrifList"
OUTPUT_DIR = r"C:\Users\16377\Downloads\RATE"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "SO_Rate_All.csv")

STATE_ABBR = "AZ"
STATE_FIPS = "04"
YEARS = list(range(2018, 2024))
MONTH_COLS = ["January","February","March","April","May","June",
              "July","August","September","October","November","December"]
HTTP_HDRS = {"User-Agent": "Mozilla/5.0 (compatible; SO-Rates/1.8; +local)"}

DEBUG_PRINT = True  # set False when stable

# ================== Small helpers ==================
def _debug(msg: str):
    if DEBUG_PRINT:
        print(f"[DEBUG] {msg}")

def _norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"[\(\)\.\,]+"," ",s)
    s = re.sub(r"\s+"," ",s).strip()
    return s

# common typo & alias fixes for county labels
TYPO_CORR = {
    "sherrif": "sheriff",
    "sheriffs": "sheriff's",
    "sheriff s": "sheriff's",
}

def normalize_county_label(raw: str) -> str:
    n = _norm_text(raw)
    for k,v in TYPO_CORR.items():
        n = n.replace(k, v)
    # remove suffixes like "county sheriff's office", "co so", "sheriff office", etc.
    n = re.sub(r"\bcounty\s+sheriff(?:'s)?\s+office\b", "", n)
    n = re.sub(r"\bco\s*so\b", "", n)
    n = re.sub(r"\bsheriff(?:'s)?\s+office\b", "", n)
    n = re.sub(r"\bso\b", "", n)
    n = re.sub(r"\bdept\b", "", n)
    # remove trailing "county" duplication after stripping
    n = re.sub(r"\bcounty\b", "", n)
    n = re.sub(r"\s+"," ",n).strip()
    # put back "County" in title-case for matching
    if n and not n.endswith("county"):
        n = n + " county"
    return n

def read_header_lines(csv_path: str, max_lines: int = 100) -> list[str]:
    with open(csv_path, "r", encoding="latin1") as f:
        return f.read().splitlines()[:max_lines]

def parse_office_and_county(lines: list[str]) -> Tuple[str, str]:
    """
    Return (office_name, county_name_plain). If header has 'Jurisdiction by Geography: X',
    we try to extract 'X County' as the county name.
    """
    office_name, county = None, None
    # 1) Jurisdiction by Geography: ...
    for ln in lines:
        m = re.match(r'^\s*"?Jurisdiction by Geography\s*:\s*(.+?)"?\s*$', ln, flags=re.I)
        if m:
            office_name = m.group(1).strip().strip('"')
            county = office_name
            break
    # 2) Fallback: Jurisdiction: ...
    if not county:
        for ln in lines:
            m = re.match(r'^\s*"?Jurisdiction\s*:\s*(.+?)"?\s*$', ln, flags=re.I)
            if m:
                office_name = m.group(1).strip().strip('"')
                county = office_name
                break
    if not county:
        raise ValueError("Cannot parse county/office from header.")
    # normalize to "... County"
    norm = normalize_county_label(county)
    # title-case for pretty output
    nice = " ".join([w.capitalize() for w in norm.split()])
    return office_name or nice, nice

# ================== County FIPS lookup & Population ==================
def _get_json(url: str, params: dict) -> list:
    r = requests.get(url, params=params, timeout=25, headers=HTTP_HDRS)
    r.raise_for_status()
    return r.json()

def fetch_az_counties_list(year_for_lookup: int = 2023) -> list[list[str]]:
    """
    Returns rows like: ["Apache County, Arizona", "04", "001"]
    """
    params = {"get": "NAME", "for": "county:*", "in": f"state:{STATE_FIPS}"}
    url = f"https://api.census.gov/data/{year_for_lookup}/acs/acs5"
    try:
        return _get_json(url, params)
    except Exception:
        # fallback to acs1
        url2 = f"https://api.census.gov/data/{year_for_lookup}/acs/acs1"
        return _get_json(url2, params)

def _base_county_name(name: str) -> str:
    left = name.split(",")[0]  # "Apache County"
    left = _norm_text(left)
    left = re.sub(r"\bcounty\b","", left)
    return re.sub(r"\s+"," ", left).strip()  # "apache"

def resolve_county_code(county_label: str) -> str:
    """
    county_label is like "Apache County". Return county FIPS (3-digit string).
    """
    target = _norm_text(county_label)
    target = re.sub(r"\bcounty\b","", target).strip()  # "apache"
    data = fetch_az_counties_list(2023)
    # exact / startswith / token overlap
    cands = []
    for row in data[1:]:
        name, st, county = row[0], row[1], row[2]
        base = _base_county_name(name)  # "apache"
        if base == target:
            return county
        cands.append((county, base))
    # fuzzy
    for fips, base in cands:
        if base.startswith(target) or target.startswith(base) or target in base or base in target:
            return fips
    # token overlap
    def jacc(a,b):
        sa, sb = set(a.split()), set(b.split())
        return len(sa & sb) / max(1, len(sa|sb))
    best=None; bs=0.0
    for fips, base in cands:
        s=jacc(base, target)
        if s>bs: bs=s; best=fips
    if best and bs>=0.8:
        return best
    raise ValueError(f"Cannot resolve county FIPS for: {county_label}")

def fetch_county_population_series(county_fips: str) -> Dict[int,int]:
    """
    Return {year: population} using ACS (prefer acs1; fallback acs5/decennial 2020).
    """
    def acs1(y):
        url=f"https://api.census.gov/data/{y}/acs/acs1"
        params={"get":"NAME,B01003_001E","for":f"county:{county_fips}","in":f"state:{STATE_FIPS}"}
        return _get_json(url, params)
    def acs5(y):
        url=f"https://api.census.gov/data/{y}/acs/acs5"
        params={"get":"NAME,B01003_001E","for":f"county:{county_fips}","in":f"state:{STATE_FIPS}"}
        return _get_json(url, params)
    def dec2020():
        url="https://api.census.gov/data/2020/dec/pl"
        params={"get":"NAME,P1_001N","for":f"county:{county_fips}","in":f"state:{STATE_FIPS}"}
        return _get_json(url, params)

    out={}
    for y in YEARS:
        try:
            arr=acs1(y); val=int(arr[1][1])
        except Exception:
            if y==2020:
                try:
                    arr=acs5(y); val=int(arr[1][1])
                except Exception:
                    arr=dec2020(); val=int(arr[1][1])
            else:
                arr=acs5(y); val=int(arr[1][1])
        out[y]=val
        time.sleep(0.12)
    return out

# ================== County ZIPs ==================
def slug(s:str)->str: return re.sub(r"[^a-z0-9]+","-",s.strip().lower()).strip("-")

def get_county_zipcodes_zipcodes_com(county_label: str) -> List[str]:
    """
    Scrape https://www.zip-codes.com/county/az-<county>.asp and collect all 5-digit ZIPs.
    Includes Standard / Unique / P.O. Box.
    """
    url=f"https://www.zip-codes.com/county/{STATE_ABBR.lower()}-{slug(county_label.replace('County','').strip())}.asp"
    r=requests.get(url, timeout=25, headers=HTTP_HDRS)
    r.raise_for_status()
    html=r.text
    # ZIPs appear in several tables; capture all 5-digit tokens in ZIP columns:
    zips = set(re.findall(r">\s*(\d{5})\s*<", html))
    if not zips:
        raise ValueError(f"No ZIPs found at {url}")
    return sorted(zips)

# ================== Monthly counts parser (robust) ==================
def _find_month_table_start(lines: List[str]) -> Optional[int]:
    for i, ln in enumerate(lines):
        if re.search(r'\bSummary\s*Month\b', ln, flags=re.I):
            return i
    months = ["january","february","march","april","may","june",
              "july","august","september","october","november","december"]
    for i, ln in enumerate(lines):
        low = ln.lower()
        if all(m in low for m in months):
            return i
    for i, ln in enumerate(lines):
        if re.search(r'\byear\b', ln, flags=re.I):
            return i
    return None

def read_monthly_counts(csv_path: str) -> pd.DataFrame:
    """
    Returns rows: year, month_index, count (only 2018–2023)
    """
    with open(csv_path,"r",encoding="latin1") as f:
        lines = f.read().splitlines()

    start = _find_month_table_start(lines)
    if start is None:
        raise ValueError("Monthly table header not found.")

    cleaned = [lines[start]]
    i = start + 1
    if i < len(lines) and re.search(r'^\s*"?Summary Date"?\s*$', lines[i], flags=re.I):
        _debug("Found 'Summary Date' after header → skip")
        i += 1
    cleaned.extend(lines[i:])

    df = pd.read_csv(io.StringIO("\n".join(cleaned)))
    df = df.rename(columns=lambda c: str(c).strip())

    def _num(s: pd.Series) -> pd.Series:
        return pd.to_numeric(s.astype(str).str.replace(",", ""), errors="coerce")

    if "Summary Month" in df.columns:
        year_series = _num(df["Summary Month"])
    elif "Year" in df.columns:
        year_series = _num(df["Year"])
    else:
        year_series = None
        for c in df.columns:
            if re.fullmatch(r"\D*year\D*", c, flags=re.I):
                year_series = _num(df[c]); break
        if year_series is None:
            for c in df.columns:
                s = _num(df[c])
                if s.notna().all() and s.astype(int).astype(str).str.fullmatch(r"\d{4}").all():
                    year_series = s; break
        if year_series is None:
            raise ValueError("Cannot identify year column.")
    df["year"] = year_series

    df = df[df["year"].notna()].copy()
    df["year"] = df["year"].astype(int)
    df = df[df["year"].isin(YEARS)].copy()

    for c in MONTH_COLS:
        if c not in df.columns:
            df[c] = math.nan
        df[c] = _num(df[c])

    rows = []
    for _, r in df.iterrows():
        y = int(r["year"])
        for idx, mname in enumerate(MONTH_COLS, start=1):
            v = r[mname]
            if pd.notna(v):
                rows.append({"year": y, "month_index": idx, "count": float(v)})
    out = pd.DataFrame(rows).sort_values(["year","month_index"]).reset_index(drop=True)
    return out

# ================== PROCESS ONE COUNTY SO ==================
def process_single_so_csv(csv_path: str) -> pd.DataFrame:
    header = read_header_lines(csv_path)
    office_name, county_name = parse_office_and_county(header)  # e.g., "Maricopa County Sheriff's Office", "Maricopa County"
    _debug(f"Office='{office_name}', County='{county_name}'")

    # County FIPS & population series
    county_fips = resolve_county_code(county_name)
    pop_by_year = fetch_county_population_series(county_fips)  # {year: pop}

    # County ZIPs (all)
    zip_list = get_county_zipcodes_zipcodes_com(county_name)
    _debug(f"ZIPs for {county_name}: {len(zip_list)}")

    # Monthly counts (2018–2023)
    monthly = read_monthly_counts(csv_path)
    full_grid = pd.DataFrame([{"year":y,"month_index":m} for y in YEARS for m in range(1,13)])
    monthly = full_grid.merge(monthly, on=["year","month_index"], how="left")

    # Compute PD-level (SO-level) monthly rate and assign to all ZIPs
    rows=[]
    for _, r in monthly.iterrows():
        y=int(r["year"]); m=int(r["month_index"])
        month_str = f"{m:02d}/{y}"
        count = r["count"] if pd.notna(r["count"]) else None
        pop = float(pop_by_year.get(y, 0))
        if count is None or pop<=0:
            rate_out = "N/A"
        else:
            rate_out = round((float(count)/pop)*1000.0, 2)
        for z in zip_list:
            rows.append({"Month": month_str, "Zipcode": z, "Crime rate per 1000": rate_out})

    out = pd.DataFrame(rows, columns=["Month","Zipcode","Crime rate per 1000"])
    return out

# ================== MAIN (batch) ==================
def main():
    print(">>> START: AZ Sheriff’s Offices monthly crime-rate batch (v1.8-SO)")
    print(f"[INFO] INPUT_DIR = {INPUT_DIR}")
    print(f"[INFO] OUTPUT    = {OUTPUT_FILE}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files=[os.path.join(INPUT_DIR,f) for f in os.listdir(INPUT_DIR) if f.lower().endswith(".csv")]
    print(f"[INFO] CSV detected = {len(files)}")
    if not files:
        raise FileNotFoundError(f"No CSV files under: {INPUT_DIR}")

    combined=[]; failed=[]
    for fp in files:
        try:
            print(f"[INFO] Processing: {fp}")
            df=process_single_so_csv(fp)
            combined.append(df)
            print(f"[OK ] Rows: {len(df)}")
        except Exception as e:
            print(f"[ERR] Failed: {fp} -> {e}", file=sys.stderr)
            traceback.print_exc()
            failed.append((fp, str(e)))

    if not combined:
        raise RuntimeError("All files failed; no output produced.")
    out_df=pd.concat(combined, ignore_index=True)
    out_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig", lineterminator="\n")

    print("\n========== SUMMARY ==========")
    print(f"Processed files : {len(combined)}")
    print(f"Failed files    : {len(failed)}")
    if failed:
        for f,msg in failed:
            print(f"  - {f}: {msg}")
    print(f"Total rows      : {len(out_df)}")
    print(f"Output file     : {OUTPUT_FILE}")
    print(">>> DONE")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(">>> ERROR occurred:")
        traceback.print_exc()
        sys.exit(1)
