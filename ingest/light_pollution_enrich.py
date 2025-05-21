#!/usr/bin/env python3
#
# 1) Download Globe-at-Night CSVs into a single DataFrame
# 2) Scroll through ufo_sightings and update each doc with its nearest light-pollution measurement

import os, sys, math
import requests
import pandas as pd
from io import StringIO
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from elasticsearch import Elasticsearch, helpers

ES_HOST          = os.getenv("ES_HOST", "http://elasticsearch:9200")
UFO_INDEX        = "ufo_sightings"
PAGE_URL         = "https://globeatnight.org/maps-data/"
TIME_WINDOW_DAYS = 45
MAX_DISTANCE_KM  = 500

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δφ = math.radians(lat2 - lat1)
    Δλ = math.radians(lon2 - lon1)
    a = math.sin(Δφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(Δλ/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def fetch_csv_urls():
    resp = requests.get(PAGE_URL, timeout=30); resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    return sorted({
        requests.compat.urljoin(PAGE_URL, a["href"])
        for a in soup.find_all("a", href=True)
        if a["href"].lower().endswith(".csv") and "gan" in a["href"].lower()
    })

def download_and_merge(urls):
    dfs = []
    for url in urls:
        year = os.path.basename(url).lstrip("GaN").split(".")[0]
        try:
            r = requests.get(url, timeout=30); r.raise_for_status()
        except Exception as e:
            print(f"⚠️ failed to fetch {url}: {e}", file=sys.stderr)
            continue
        df = pd.read_csv(StringIO(r.text))
        df["__year"] = int(year)
        dfs.append(df)
    if not dfs:
        print("🔴 No CSVs downloaded; aborting", file=sys.stderr)
        sys.exit(1)
    return pd.concat(dfs, ignore_index=True)

def normalize_and_clean(df):
    # pick & rename exactly one of each: date, lat, lon, brightness, __year→year
    mapping = {}
    for c in df.columns:
        lc = c.lower()
        if "date" in lc and "time" not in lc:
            mapping[c] = "date"; break
    for c in df.columns:
        if any(k in c.lower() for k in ("lat","latitude")):
            mapping[c] = "lat"; break
    for c in df.columns:
        if any(k in c.lower() for k in ("lon","longitude")):
            mapping[c] = "lon"; break
    for c in df.columns:
        if any(k in c.lower() for k in ("sqm","bright","radiance")):
            mapping[c] = "brightness"; break
    mapping["__year"] = "year"

    df = df.rename(columns=mapping)
    req = ["date","lat","lon","brightness","year"]
    missing = set(req) - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    df = df[req]
    df["date"]       = pd.to_datetime(df["date"], errors="coerce")
    df["lat"]        = pd.to_numeric(df["lat"],   errors="coerce")
    df["lon"]        = pd.to_numeric(df["lon"],   errors="coerce")
    df["brightness"] = pd.to_numeric(df["brightness"], errors="coerce")

    before = len(df)
    df = df.dropna(subset=req)
    after  = len(df)
    print(f"✅ Cleaned Globe-at-Night: dropped {before-after} bad rows, {after} remain")
    return df

def enrich_ufo_with_light(globe_df):
    es     = Elasticsearch(ES_HOST, verify_certs=False)
    scroll = es.search(
        index=UFO_INDEX,
        scroll="5m",
        size=1000,
        _source=["Occurred_utc","location"]
    )
    sid, hits = scroll["_scroll_id"], scroll["hits"]["hits"]
    actions = []

    while hits:
        for doc in hits:
            src   = doc["_source"]
            loc   = src.get("location")
            dtstr = src.get("Occurred_utc")
            if not loc or not dtstr:
                continue

            # parse + drop timezone
            try:
                ufo_dt = datetime.fromisoformat(dtstr).replace(tzinfo=None)
            except:
                continue

            # narrow to ±TIME_WINDOW_DAYS
            start = ufo_dt - timedelta(days=TIME_WINDOW_DAYS)
            end   = ufo_dt + timedelta(days=TIME_WINDOW_DAYS)
            sub   = globe_df[(globe_df.date >= start) & (globe_df.date <= end)].copy()
            if sub.empty:
                continue

            # compute distance, filter & pick closest
            sub["dist"] = sub.apply(
                lambda r: haversine(loc["lat"], loc["lon"], r.lat, r.lon),
                axis=1
            )
            sub = sub[sub.dist <= MAX_DISTANCE_KM]
            if sub.empty:
                continue

            best    = sub.nsmallest(1, "dist").iloc[0]
            days_off = abs((ufo_dt.date() - best.date.date()).days)

            actions.append({
                "_op_type": "update",
                "_index":   UFO_INDEX,
                "_id":      doc["_id"],
                "doc": {
                    "light_pollution": {
                        "date":        best.date.isoformat(),
                        "brightness":  float(best.brightness),
                        "distance_km": float(best.dist),
                        "days_offset": int(days_off)
                    }
                }
            })

            if len(actions) >= 5000:
                helpers.bulk(es, actions)
                actions = []

        batch = es.scroll(scroll_id=sid, scroll="5m")
        sid, hits = batch["_scroll_id"], batch["hits"]["hits"]

    if actions:
        helpers.bulk(es, actions)

    print("✅ light_pollution_enrich complete — UFOs tagged with nearest light pollution.")

if __name__ == "__main__":
    print("→ Downloading Globe-at-Night CSVs…")
    urls = fetch_csv_urls()
    print(f"→ Found {len(urls)} CSVs; merging…")
    raw = download_and_merge(urls)

    print("→ Cleaning & normalizing…")
    globe_df = normalize_and_clean(raw)

    print("→ Enriching `ufo_sightings` with nearest light pollution…")
    enrich_ufo_with_light(globe_df)