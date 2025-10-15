# make_azpd_monthly_rate_batch.py
# v1.8 — AZ PD batch: monthly crime rate (/1,000) for 2018–2023
# - Robust monthly table parser; skips stray "Summary Date" line (Mesa/Phoenix/Tucson)
# - Alias map for joint/special agencies (ASU/NAU/UA/airport/college/Round Valley/Snowflake–Taylor)
# - ZIP fetch with City/Town & hyphen/space variants; Zillow overrides
# - Common TYPO corrections (e.g., "snowflake-tayor" -> "snowflake-taylor", "pheonix" -> "phoenix")
# - Missing months → rate "N/A"; full 72-month grid per PD

import os, io, re, sys, math, time, json, traceback, unicodedata
import requests, pandas as pd
from typing import List, Dict, Optional

# ================== CONFIG ==================
INPUT_DIR  = r"C:\Users\16377\Downloads\AZPDlist"
OUTPUT_DIR = r"C:\Users\16377\Downloads\RATE"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "PD_Rate_All.csv")

STATE_ABBR = "AZ"
STATE_FIPS = "04"
YEARS = list(range(2018, 2024))
MONTH_COLS = ["January","February","March","April","May","June",
              "July","August","September","October","November","December"]
HTTP_HDRS = {"User-Agent": "Mozilla/5.0 (compatible; PD-Rates/1.8; +local)"}

ZIP_MODE = "zillow"  # "zillow" or "city_all"
DEBUG_PRINT = True   

# Zillow-aligned overrides (extend as needed)
ZILLOW_CITY_ZIPS_AZ: Dict[str, List[str]] = {
    "flagstaff": ["86001","86002","86003","86004","86005","86011"],
}

# jurisdiction aliases → component cities (for joint PDs / special agencies)
JURISDICTION_ALIASES: Dict[str, List[str]] = {
    "round valley": ["eagar", "springerville"],
    "snowflake-taylor": ["snowflake", "taylor"],
    "az state university": ["tempe"],
    "arizona state university": ["tempe"],
    "northern az university": ["flagstaff"],
    "northern arizona university": ["flagstaff"],
    "university of arizona": ["tucson"],
    "tucson airport authority": ["tucson"],
    "central arizona college": ["coolidge"],  # main campus
}


TYPO_CORRECTIONS: Dict[str, str] = {
    "pheonix": "phoenix",
    "snowflake-tayor": "snowflake-taylor",
   
    # "huachua": "huachuca",
    # "st john": "st johns",
}

# ================== UTILS ==================
def _debug(msg: str):
    if DEBUG_PRINT:
        print(f"[DEBUG] {msg}")

def _norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"[\(\)\.\,]+"," ",s)
    s = re.sub(r"\s+"," ",s).strip()
    return s

def _get_json_or_raise(url: str, params: dict, tries: int = 3, sleep_sec: float = 0.5) -> list:
    last_exc = None
    for _ in range(tries):
        try:
            r = requests.get(url, params=params, timeout=20, headers=HTTP_HDRS)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_exc = e
            time.sleep(sleep_sec)
    raise requests.HTTPError(f"Bad/Non-JSON from {url} params={params}. Last: {last_exc}")

# ================== HEADER → PD & CITY ==================
def read_header_lines(csv_path: str, max_lines: int = 100) -> list[str]:
    with open(csv_path, "r", encoding="latin1") as f:
        return f.read().splitlines()[:max_lines]

def parse_pd_and_city(lines: list[str]) -> dict:
    pd_name, city = None, None
    for ln in lines:
        m = re.match(r'^\s*"?Jurisdiction by Geography\s*:\s*(.+?)"?\s*$', ln, flags=re.I)
        if m:
            pd_name = m.group(1).strip().strip('"')
            city = re.sub(r"\b(Police Department|PD|Sheriff(?:'s)? Office|Sheriff(?:'s)? Dept\.?)\b","",pd_name,flags=re.I).strip(", -")
            break
    if not city:
        for ln in lines:
            m = re.match(r'^\s*"?Jurisdiction\s*:\s*(.+?)"?\s*$', ln, flags=re.I)
            if m:
                pd_name = m.group(1).strip().strip('"')
                city = re.sub(r"\b(Police Department|PD|Sheriff(?:'s)? Office|Sheriff(?:'s)? Dept\.?)\b","",pd_name,flags=re.I).strip(", -")
                break
    if not city:
        raise ValueError("Cannot parse city/PD from header.")
    return {"pd_name": pd_name or f"{city} PD", "city": city}

def normalize_city(name: str) -> str:
    n = _norm_text(name)
    n = re.sub(r"\b(police department|pd|sheriff(?:'s)? office|sheriff(?:'s)? dept)\b","",n)
    n = re.sub(r"\b(city|town|village)\s+of\s+","",n)
    n = re.sub(r"\b(city|town|village)\b","",n)
    n = re.sub(r"\s+"," ",n).strip()
    
    if n in TYPO_CORRECTIONS:
        n = TYPO_CORRECTIONS[n]
    return n

# ================== CENSUS PLACE DISCOVERY ==================
def _base_place_name(name: str) -> str:
    left = name.split(",")[0]
    left = _norm_text(left)
    left = re.sub(r"\b(city|town|village|cdp)\b","",left)
    left = re.sub(r"\bbalance\b","",left)
    left = re.sub(r"\s+"," ",left).strip()
    return left

def _token_jaccard(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb: return 0.0
    return len(sa & sb) / len(sa | sb)

def _fetch_places_list(year_for_lookup: int = 2023) -> list[list[str]]:
    params = {"get":"NAME","for":"place:*","in":f"state:{STATE_FIPS}"}
    url5 = f"https://api.census.gov/data/{year_for_lookup}/acs/acs5"
    try:
        return _get_json_or_raise(url5, params)
    except requests.HTTPError:
        url1 = f"https://api.census.gov/data/{year_for_lookup}/acs/acs1"
        return _get_json_or_raise(url1, params)

def find_place_code_for_city_AZ(city: str, year_for_lookup: int = 2023, strict: bool=False) -> Optional[str]:
    target = normalize_city(city)
    if not target: return None
    def _search(year:int)->Optional[str]:
        data = _fetch_places_list(year)
        cands=[]
        for row in data[1:]:
            name, st, place = row[0], row[1], row[2]
            base = _base_place_name(name)
            if not base: continue
            if base == target: return place
            cands.append((place, base))
        for place, base in cands:
            if base.startswith(target) or target.startswith(base) or target in base or base in target:
                return place
        best_pc, best_score = None, 0.0
        for place, base in cands:
            s = _token_jaccard(base, target)
            if s > best_score: best_score, best_pc = s, place
        return best_pc if best_score >= 0.8 else None
    pc=None
    for yr in (year_for_lookup, 2022):
        try:
            pc=_search(yr)
            if pc: break
        except requests.RequestException:
            continue
    if not pc and strict: raise ValueError(f"Cannot find Census 'place' for {city}, AZ.")
    return pc

def resolve_component_place_codes(norm_label: str) -> List[str]:
    if norm_label in JURISDICTION_ALIASES:
        pcs=[]
        for comp_city in JURISDICTION_ALIASES[norm_label]:
            pc = find_place_code_for_city_AZ(comp_city, 2023, strict=True)
            pcs.append(pc)
        return pcs
    pc = find_place_code_for_city_AZ(norm_label, 2023, strict=False)
    if not pc: raise ValueError(f"Cannot find Census 'place' for {norm_label}, AZ.")
    return [pc]

# ================== POPULATION ==================
def _fetch_acs1_population(year:int, place_code:str, api_key:Optional[str])->int:
    url=f"https://api.census.gov/data/{year}/acs/acs1"
    params={"get":"NAME,B01003_001E","for":f"place:{place_code}","in":f"state:{STATE_FIPS}"}
    if api_key: params["key"]=api_key
    arr=_get_json_or_raise(url, params)
    return int(arr[1][1])

def _fetch_acs5_population(year:int, place_code:str, api_key:Optional[str])->int:
    url=f"https://api.census.gov/data/{year}/acs/acs5"
    params={"get":"NAME,B01003_001E","for":f"place:{place_code}","in":f"state:{STATE_FIPS}"}
    if api_key: params["key"]=api_key
    arr=_get_json_or_raise(url, params)
    return int(arr[1][1])

def _fetch_decennial_2020_population(place_code:str, api_key:Optional[str])->int:
    url="https://api.census.gov/data/2020/dec/pl"
    params={"get":"NAME,P1_001N","for":f"place:{place_code}","in":f"state:{STATE_FIPS}"}
    if api_key: params["key"]=api_key
    arr=_get_json_or_raise(url, params)
    return int(arr[1][1])

def get_population_series_for_places(place_codes: List[str]) -> Dict[int,int]:
    key=os.environ.get("CENSUS_API_KEY")
    summed: Dict[int,int] = {y:0 for y in YEARS}
    for pc in place_codes:
        for y in YEARS:
            try:
                val=_fetch_acs1_population(y, pc, key)
            except (requests.HTTPError, requests.RequestException, ValueError):
                if y==2020:
                    try:
                        val=_fetch_acs5_population(y, pc, key)
                    except (requests.HTTPError, requests.RequestException, ValueError):
                        val=_fetch_decennial_2020_population(pc, key)
                else:
                    val=_fetch_acs5_population(y, pc, key)
            summed[y] += int(val)
            time.sleep(0.15)
    return summed

# ================== ZIP CODES ==================
def slug_city(s:str)->str: return re.sub(r"[^a-z0-9]+","-",s.strip().lower()).strip("-")

def _zip_variants_for_query(city_label: str) -> List[str]:
    base = city_label.strip()
    
    base = TYPO_CORRECTIONS.get(_norm_text(base), base)

    variants = [base]
    if not re.search(r"\b(city|town)\b", base, flags=re.I):
        variants += [f"{base} city", f"{base} town"]
    if "-" in base:
        variants.append(base.replace("-", " "))
    elif " " in base:
        variants.append(base.replace(" ", "-"))
    variants += [base.title(), base.lower()]
    seen=set(); out=[]
    for v in variants:
        v=v.strip()
        if v and v.lower() not in seen:
            seen.add(v.lower()); out.append(v)
    return out

def get_zipcodes_zipcodes_com(city:str)->List[str]:
    url=f"https://www.zip-codes.com/city/{STATE_ABBR.lower()}-{slug_city(city)}.asp"
    r=requests.get(url, timeout=20, headers=HTTP_HDRS); r.raise_for_status()
    html=r.text
    rows=re.findall(r">(\d{5})<.*?>\s*(Standard|Unique|P\.O\. Box)\s*<", html, flags=re.I|re.S)
    return sorted({z for z,_ in rows})

def get_zipcodes_zippopotam(city:str)->List[str]:
    url=f"http://api.zippopotam.us/us/{STATE_ABBR.lower()}/{slug_city(city)}"
    r=requests.get(url, timeout=15, headers=HTTP_HDRS)
    if r.status_code!=200: return []
    data=r.json()
    return sorted({p.get("post code") for p in data.get("places",[]) if p.get("post code")})

def get_city_zipcodes_all(city_label:str)->List[str]:
    z=set()
    for variant in _zip_variants_for_query(city_label):
        try:
            z.update(get_zipcodes_zipcodes_com(variant))
        except Exception:
            pass
        try:
            z.update(get_zipcodes_zippopotam(variant))
        except Exception:
            pass
        if z: break
    if not z:
        raise ValueError(f"Cannot fetch ZIP codes for {city_label}, {STATE_ABBR}.")
    return sorted(z)

def get_zip_union_for_components(components: List[str]) -> List[str]:
    zset=set()
    for comp in components:
        key = normalize_city(comp)
        if ZIP_MODE=="zillow" and key in ZILLOW_CITY_ZIPS_AZ:
            zset.update(ZILLOW_CITY_ZIPS_AZ[key])
        else:
            zset.update(get_city_zipcodes_all(comp))
        time.sleep(0.1)
    return sorted(zset)

# ================== MONTHLY COUNTS (ROBUST + MESA/PHX/TUS FIX) ==================
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
    Reads monthly table:
      - finds header row (Summary Month / Year / line listing 12 months)
      - skips 'Summary Date' line right under header (Mesa/Phoenix/Tucson)
      - numbers with commas supported
      - returns rows: year, month_index, count
    """
    with open(csv_path,"r",encoding="latin1") as f:
        lines = f.read().splitlines()

    start = _find_month_table_start(lines)
    if start is None:
        raise ValueError("Monthly table header not found (no 'Summary Month'/'Year' or month headers).")

    # Build cleaned slice: header + data; skip 'Summary Date' if present
    cleaned = [lines[start]]
    i = start + 1
    if i < len(lines) and re.search(r'^\s*"?Summary Date"?\s*$', lines[i], flags=re.I):
        _debug("Found 'Summary Date' line after header → skipping it")
        i += 1
    nxt = lines[i][:60] if i < len(lines) else "<EOF>"
    _debug(f"header index = {start}, next line starts with: {nxt}")
    cleaned.extend(lines[i:])

    df = pd.read_csv(io.StringIO("\n".join(cleaned)))
    df = df.rename(columns=lambda c: str(c).strip())

    # year column normalize (with commas handling)
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
            raise ValueError("Cannot identify year column in monthly table.")
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

# ================== PROCESS ONE PD ==================
def process_single_pd_csv(csv_path:str)->pd.DataFrame:
    header=read_header_lines(csv_path)
    meta=parse_pd_and_city(header)
    norm_label=normalize_city(meta["city"])
    _debug(f"Parsed city: '{norm_label}'")

    components = JURISDICTION_ALIASES.get(norm_label, [norm_label])
    if components == [norm_label]:
        place_codes = resolve_component_place_codes(norm_label)
    else:
        place_codes = [find_place_code_for_city_AZ(c, 2023, strict=True) for c in components]

    pop_by_year = get_population_series_for_places(place_codes)
    zip_list = get_zip_union_for_components(components)

    monthly=read_monthly_counts(csv_path)
    full_grid=pd.DataFrame([{"year":y,"month_index":m} for y in YEARS for m in range(1,13)])
    monthly=full_grid.merge(monthly, on=["year","month_index"], how="left")

    monthly["population"]=monthly["year"].map(pop_by_year).astype(float)
    monthly["crime_rate_per_1000"]=(monthly["count"] / monthly["population"]) * 1000.0
    monthly["crime_rate_per_1000"]=monthly["crime_rate_per_1000"].round(2)
    monthly["Month"]=monthly.apply(lambda r: f"{int(r['month_index']):02d}/{int(r['year'])}", axis=1)

    rows=[]
    for _,r in monthly.iterrows():
        rate = r["crime_rate_per_1000"]
        rate_out = "N/A" if pd.isna(rate) else float(rate)
        for z in zip_list:
            rows.append({"Month": r["Month"], "Zipcode": z, "Crime rate per 1000": rate_out})
    out_df=pd.DataFrame(rows, columns=["Month","Zipcode","Crime rate per 1000"])

    _debug(f"months_present_after_fill = {monthly.shape[0]} (72 expected)")
    _debug(f"zip_count = {len(zip_list)}  → expected rows = {72*len(zip_list)}")
    return out_df

# ================== BATCH MAIN ==================
def main():
    print(">>> START: AZ PD monthly crime-rate batch")
    print(f"[INFO] INPUT_DIR = {INPUT_DIR}")
    print(f"[INFO] OUTPUT    = {OUTPUT_FILE}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files=[os.path.join(INPUT_DIR,f) for f in os.listdir(INPUT_DIR) if f.lower().endswith(".csv")]
    print(f"[INFO] CSV detected = {len(files)}")
    if not files: raise FileNotFoundError(f"No CSV files under: {INPUT_DIR}")

    combined=[]; failed=[]
    for fp in files:
        try:
            print(f"[INFO] Processing: {fp}")
            df=process_single_pd_csv(fp)
            combined.append(df)
            print(f"[OK ] Rows: {len(df)}")
        except Exception as e:
            print(f"[ERR] Failed: {fp} -> {e}", file=sys.stderr)
            traceback.print_exc()
            failed.append((fp, str(e)))

    if not combined: raise RuntimeError("All files failed; no output produced.")
    out_df=pd.concat(combined, ignore_index=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig", lineterminator="\n")

    print("\n========== SUMMARY ==========")
    print(f"Processed files : {len(combined)}")
    print(f"Failed files    : {len(failed)}")
    if failed:
        for f,msg in failed: print(f"  - {f}: {msg}")
    print(f"Total rows      : {len(out_df)}")
    print(f"Output file     : {OUTPUT_FILE}")
    print(">>> DONE")

# ================== ENTRY ==================
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(">>> ERROR occurred:")
        traceback.print_exc()
        sys.exit(1)
