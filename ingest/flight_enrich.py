#!/usr/bin/env python3
# ingest/flight_enrich.py — enrich UFO sightings with monthly air-traffic stats

import os, sys, pandas as pd
from datetime import datetime
from elasticsearch import Elasticsearch, helpers

ES_HOST   = os.getenv("ES_HOST", "http://elasticsearch:9200")
UFO_INDEX = "ufo_sightings"
CSV_PATH  = os.path.join("data", "air_traffic_data.csv")

def load_air_traffic(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)

    # month name → month number
    df["month_num"] = pd.to_datetime(df["Month"], format="%B", errors="coerce").dt.month
    # normalize the “GEO Region” column
    df["region"] = df["GEO Region"].str.strip().str.upper()

    # pick passenger count (fall back to Adjusted)
    df["passenger_count"] = pd.to_numeric(
        df["Passenger Count"].fillna(df["Adjusted Passenger Count"]),
        errors="coerce"
    )

    # drop bad rows
    df = df.dropna(subset=["region","Year","month_num","passenger_count"])

    # uppercase the activity code so it matches pivot keys
    df["act_type"] = df["Activity Type Code"].str.upper()

    # sum by region/year/month/activity
    grp = (df.groupby(
               ["region","Year","month_num","act_type"],
               as_index=False
           )["passenger_count"].sum())

    pivot = grp.pivot_table(
        index   = ["region","Year","month_num"],
        columns = "act_type",
        values  = "passenger_count",
        fill_value=0,
    )
    pivot["TOTAL"] = pivot.sum(axis=1)

    lookup = {}
    for (region, year, month), row in pivot.iterrows():
        lookup[(region, int(year), int(month))] = {
            "region"   : region,
            "enplaned" : int(row.get("ENPLANED", 0)),
            "deplaned" : int(row.get("DEPLANED", 0)),
            "transit"  : int(row.get("THRU / TRANSIT", 0)),
            "total"    : int(row["TOTAL"]),
        }
    return lookup

def map_country_to_region(country: str | None) -> str | None:
    """
    Map every Country (ISO or full name) into exactly the
    GEO REGION keys in the SFO CSV pivot, e.g. 'US', 'CANADA', etc.
    """
    if not country:
        return None
    c = country.strip().upper()

    mapping = {
        # ── North America ─────────────────────────────────────────────────────
        **dict.fromkeys(
            ["US","USA","UNITED STATES","UNITED STATES OF AMERICA"], 
            "US"
        ),
        **dict.fromkeys(["CA","CANADA"], "CANADA"),
        **dict.fromkeys(["MX","MEXICO"], "MEXICO"),

        # ── Central America ───────────────────────────────────────────────────
        **dict.fromkeys([
            "GT","GUATEMALA","BZ","BELIZE","SV","EL SALVADOR",
            "HN","HONDURAS","NI","NICARAGUA","CR","COSTA RICA","PA","PANAMA"
        ], "CENTRAL AMERICA"),

        # ── South America ─────────────────────────────────────────────────────
        **dict.fromkeys([
            "AR","ARGENTINA","BO","BOLIVIA","BR","BRAZIL",
            "CL","CHILE","CO","COLOMBIA","EC","ECUADOR",
            "PE","PERU","UY","URUGUAY","VE","VENEZUELA",
            "PY","PARAGUAY","SR","SURINAME","GY","GUYANA"
        ], "SOUTH AMERICA"),

        # ── Europe ────────────────────────────────────────────────────────────
        **dict.fromkeys([
            "UK","UNITED KINGDOM","GB","IE","IRELAND",
            "FR","FRANCE","DE","GERMANY","ES","SPAIN","PT","PORTUGAL",
            "IT","ITALY","NL","NETHERLANDS","BE","BELGIUM",
            "SE","SWEDEN","NO","NORWAY","DK","DENMARK",
            "FI","FINLAND","PL","POLAND","CZ","CZECH REPUBLIC",
            "GR","GREECE","HU","HUNGARY","RO","ROMANIA",
            "CH","SWITZERLAND","AT","AUSTRIA"
        ], "EUROPE"),

        # ── Asia ───────────────────────────────────────────────────────────────
        **dict.fromkeys([
            "CN","CHINA","JP","JAPAN","KR","SOUTH KOREA",
            "IN","INDIA","SG","SINGAPORE","MY","MALAYSIA",
            "ID","INDONESIA","TH","THAILAND",
            "PH","PHILIPPINES","VN","VIETNAM","HK","HONG KONG","TW","TAIWAN",
            "RU","RUSSIA"
        ], "ASIA"),

        # ── Middle East ───────────────────────────────────────────────────────
        **dict.fromkeys([
            "SAUDI ARABIA","AE","UNITED ARAB EMIRATES","IL","ISRAEL",
            "IRAN","IQ","IRAQ","JO","JORDAN","LB","LEBANON",
            "SY","SYRIA","KW","KUWAIT","QA","QATAR","BH","BAHRAIN","OM","OMAN"
        ], "MIDDLE EAST"),

        # ── Australasia ───────────────────────────────────────────────────────
        **dict.fromkeys([
            "AU","AUSTRALIA","NZ","NEW ZEALAND","PG","PAPUA NEW GUINEA",
            "FJ","FIJI"
        ], "AUSTRALIA / OCEANIA"),

        # ── Africa ────────────────────────────────────────────────────────────
        **dict.fromkeys([
            "ZA","SOUTH AFRICA","NG","NIGERIA","EG","EGYPT",
            "DZ","ALGERIA","MA","MOROCCO","KE","KENYA"
        ], "AFRICA"),
    }

    return mapping.get(c)

def main() -> None:
    lookup = load_air_traffic(CSV_PATH)
    if not lookup:
        print(f"🔴  Couldn’t build lookup from {CSV_PATH}")
        sys.exit(1)

    es = Elasticsearch([ES_HOST], verify_certs=False)
    scroll = es.search(
        index   = UFO_INDEX,
        scroll  = "5m",
        size    = 1000,
        _source = ["Occurred_utc", "Country"],
        query   = {"match_all": {}},
    )
    sid, hits = scroll["_scroll_id"], scroll["hits"]["hits"]
    actions   = []

    while hits:
        for doc in hits:
            src = doc["_source"]
            try:
                dt = datetime.fromisoformat(src["Occurred_utc"])
            except Exception:
                continue

            region = map_country_to_region(src.get("Country"))
            if not region:
                # still unmapped – you can log src["Country"] here to debug
                continue

            key = (region, dt.year, dt.month)
            metrics = lookup.get(key)
            if not metrics:
                continue

            actions.append({
                "_op_type": "update",
                "_index"  : UFO_INDEX,
                "_id"     : doc["_id"],
                "doc"     : {"air_traffic_monthly": metrics},
            })

            if len(actions) >= 5000:
                helpers.bulk(es, actions)
                actions.clear()

        batch     = es.scroll(scroll_id=sid, scroll="5m")
        sid, hits = batch["_scroll_id"], batch["hits"]["hits"]

    if actions:
        helpers.bulk(es, actions)

    print("✅ flight_enrich → added air_traffic_monthly to sightings")

if __name__ == "__main__":
    main()