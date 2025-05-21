#!/usr/bin/env python3
# ingest/ufo_ingest.py — merge NUFORC+Kaggle, clean, geocode, index “ufo_sightings”

import os, json, sys
import pandas as pd, pytz
from elasticsearch import Elasticsearch, helpers
from elasticsearch.helpers import BulkIndexError
from kaggle.api.kaggle_api_extended import KaggleApi

DATA_DIR = "data"
ES_HOST  = os.getenv("ES_HOST", "http://elasticsearch:9200")
INDEX    = "ufo_sightings"

# 1️⃣ Download Kaggle CSV if missing
csv_path = os.path.join(DATA_DIR, "ufo_sightings.csv")
if not os.path.exists(csv_path):
    api = KaggleApi(); api.authenticate()
    api.dataset_download_files("andrewmvd/ufo-sightings", path=DATA_DIR, unzip=True)

# 2️⃣ Load NUFORC JSON + Kaggle CSV (don’t auto-parse dates)
df1 = pd.DataFrame(json.load(open(os.path.join(DATA_DIR,"nuforc.json"))))
df2 = pd.read_csv(csv_path, parse_dates=False, engine="python", on_bad_lines="skip")
df2.rename(columns={"datetime":"Occurred"}, inplace=True)
df = pd.concat([df1, df2], ignore_index=True, sort=False)
print(f"🔗 Combined: {len(df)} rows")

# 3️⃣ Clean the Occurred strings
df["Occurred"] = df["Occurred"].astype(str).str.replace(r"\s*Local$","",regex=True)

# 4️⃣ Two‐pass date parsing → Occurred_utc
eastern = pytz.timezone("US/Eastern")

# pass 1: let pandas parse as UTC
df["dt1"] = pd.to_datetime(df["Occurred"], errors="coerce", utc=True)

# pass 2: for those still NaT, parse naïve then localize→UTC
mask = df["dt1"].isna()
naive = pd.to_datetime(df.loc[mask,"Occurred"], errors="coerce", utc=False)
def localize_then_utc(ts):
    if pd.isna(ts): return pd.NaT
    try:
        return eastern.localize(ts, is_dst=None).astimezone(pytz.utc)
    except (pytz.AmbiguousTimeError,pytz.NonExistentTimeError):
        return eastern.localize(ts + pd.Timedelta(hours=1), is_dst=False).astimezone(pytz.utc)
df.loc[mask, "dt2"] = naive.apply(localize_then_utc)

# final Occurred_utc
df["Occurred_utc"] = df["dt1"].fillna(df["dt2"])
n_bad = df["Occurred_utc"].isna().sum()
print(f"📅 {len(df)} total, {n_bad} unparsed dates → dropping those")

# 5️⃣ Split & clean location into City/State/Country
df["Location"] = (
    df.get("city",pd.Series(dtype=str)).fillna("") + ", " +
    df.get("state",pd.Series(dtype=str)).fillna("") + ", " +
    df.get("country",pd.Series(dtype=str)).fillna("")
).str.strip(", ")
def split_loc(s):
    parts = [p.strip() for p in s.split(",")]
    parts += [None]*(3-len(parts))
    return pd.Series(parts[:3], index=["City","State","Country"])
df[["City","State","Country"]] = df["Location"].apply(split_loc)

# 6️⃣ Pull lat/lon from Kaggle CSV
df["lat"] = pd.to_numeric(df.get("latitude"),   errors="coerce") \
             .fillna(pd.to_numeric(df.get("lat_kaggle"), errors="coerce"))
df["lon"] = pd.to_numeric(df.get("longitude"),  errors="coerce") \
             .fillna(pd.to_numeric(df.get("lon_kaggle"), errors="coerce"))
n_loc_bad = df[["lat","lon"]].isna().any(axis=1).sum()
print(f"📍 {n_loc_bad} bad coords → dropping those")

# 7️⃣ Filter to only good rows
df = df[df["Occurred_utc"].notna() & df["lat"].notna() & df["lon"].notna()]
print(f"✅ {len(df)} rows remain after filtering")

# ─── FIXES ───────────────────────────────────────────────────────────────────

# 8️⃣ Coalesce shape from JSON (`Shape`) and CSV (`shape`)
df["shape"] = df.get("shape").fillna(df.get("Shape"))

# 9️⃣ Build a unified 'duration' text field from CSV’s two columns
df["duration"] = (
    df.get("duration (hours/min)", pd.Series(dtype=str))
      .fillna("")                      # use hours/min if present
      .replace("", pd.NA)
)
# fallback to seconds if hours/min is missing
sec = df.get("duration (seconds)")
if sec is not None:
    df["duration"] = df["duration"].fillna(
        sec.astype(str).str.strip() + " sec"
    )

# 🔟 Coalesce comments: CSV uses 'comments', JSON uses 'Text'
df["comments"] = (
    df.get("comments")
      .fillna(df.get("Text"))
)

# ─── 1️⃣1️⃣ Create ES index with proper mapping ───────────────────────────────
es = Elasticsearch([ES_HOST], verify_certs=False)
es.indices.delete(index=INDEX, ignore=[400,404])
es.indices.create(index=INDEX, body={
  "mappings": {
    "properties": {
      "Occurred_utc":      {"type":"date"},
      "City":              {"type":"keyword"},
      "State":             {"type":"keyword"},
      "Country":           {"type":"keyword"},
      "shape":             {"type":"keyword"},
      "duration":          {"type":"text"},
      "comments":          {"type":"text"},
      "location":          {"type":"geo_point"}
    }
  }
})

# ─── 1️⃣2️⃣ Bulk‐index ────────────────────────────────────────────────────────
def gen():
    for i,row in df.iterrows():
        yield {
          "_index": INDEX, "_id": i,
          "_source": {
            "Occurred_utc": row["Occurred_utc"].isoformat(),
            "City":         row["City"],
            "State":        row["State"],
            "Country":      row["Country"],
            "shape":        row["shape"] if pd.notna(row["shape"]) else None,
            "duration":     row["duration"] if pd.notna(row["duration"]) else None,
            "comments":     row["comments"] if pd.notna(row["comments"]) else None,
            "location":     {"lat": row["lat"], "lon": row["lon"]}
          }
        }

try:
    count, errors = helpers.bulk(es, gen(), stats_only=True, raise_on_error=False)
    print(f"✅ Indexed {count} docs")
    if errors:
        print("⚠️  First bulk error:", errors[0])
except BulkIndexError as e:
    print("🔴 Bulk‐index failed:", e)
    sys.exit(1)

# ─── 1️⃣3️⃣ Quick verify ─────────────────────────────────────────────────────
resp = es.count(index=INDEX, body={"query":{"exists":{"field":"Occurred_utc"}}})
print(f"🔍 ES shows {resp['count']} docs with Occurred_utc")