#!/usr/bin/env python3
# ingest/meteor_enrich.py — tag each UFO with active meteor showers

import os
import pandas as pd
from datetime import datetime
from elasticsearch import Elasticsearch, helpers

ES_HOST       = os.getenv("ES_HOST", "http://elasticsearch:9200")
UFO_INDEX     = "ufo_sightings"
SHOWERS_FILE  = "data/meteor_showers.csv"

def load_showers():
    # read the pipe‐delimited table, skip comments
    df = pd.read_csv(
        SHOWERS_FILE,
        sep="|",
        comment="#",
        skipinitialspace=True,
        names=[
          "iau_no","code","name",
          "begin_sol","max_sol","end_sol",
          "ra","dra","dec","ddec","vg","dvg","ref"
        ],
        usecols=["code","name","begin_sol","max_sol","end_sol"]
    )

    # drop any showers where we can't determine a full window
    df = df.dropna(subset=["begin_sol","max_sol","end_sol"])

    # approximate: solar‐longitude (°) → day‐of‐year (DOY)
    # 0° ≃ March 20 → DOY 79
    base   = 79.0
    factor = 365.2422 / 360.0

    df["begin_doy"] = (base + df.begin_sol * factor).round().astype(int)
    df["peak_doy"]  = (base + df.max_sol   * factor).round().astype(int)
    df["end_doy"]   = (base + df.end_sol   * factor).round().astype(int)

    return df[["code","name","begin_doy","peak_doy","end_doy"]]

def main():
    showers = load_showers()
    es      = Elasticsearch([ES_HOST], verify_certs=False)

    # scroll through all UFO docs
    scroll = es.search(
      index=UFO_INDEX, scroll="2m", size=1000,
      _source=["Occurred_utc"]
    )
    sid  = scroll["_scroll_id"]
    hits = scroll["hits"]["hits"]
    actions = []

    while hits:
        for d in hits:
            # parse date, extract day‐of‐year
            dt  = datetime.fromisoformat(d["_source"]["Occurred_utc"][:10])
            doy = dt.timetuple().tm_yday

            # find any showers active on that day
            active = showers[
              (showers.begin_doy <= doy) &
              (showers.end_doy   >= doy)
            ]
            if active.empty:
                continue

            actions.append({
              "_op_type":"update",
              "_index":  UFO_INDEX,
              "_id":     d["_id"],
              "doc": {
                "meteor_shower_codes":   active.code.tolist(),
                "meteor_shower_names":   active.name.tolist(),
                "days_from_shower_peak": (doy - active.peak_doy).tolist()
              }
            })

        batch = es.scroll(scroll_id=sid, scroll="2m")
        sid   = batch["_scroll_id"]
        hits  = batch["hits"]["hits"]

    if actions:
        helpers.bulk(es, actions, chunk_size=5000)

    print("✅ meteor_enrich complete — UFOs tagged with active meteor showers.")

if __name__=="__main__":
    main()