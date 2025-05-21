# dashboard.py – Global UFO + Meteor + Air-Traffic explorer
# ============================================================================
# Streamlit 1.32 • Bokeh 3 • Folium 0.15 • Elasticsearch 8
# ============================================================================

import time
import streamlit as st
import pandas as pd
import numpy  as np
from elasticsearch import Elasticsearch, helpers
from bokeh.plotting import figure
from bokeh.models   import (
    ColumnDataSource, LinearAxis, Range1d,
    DatetimeTickFormatter, NumeralTickFormatter, HoverTool,
)
import folium
from folium import CircleMarker
from streamlit_folium import st_folium
from branca.colormap import linear
from bokeh.transform import dodge

# ─────────────────────────────────────────────────────────────────────────────
# Elasticsearch
# ─────────────────────────────────────────────────────────────────────────────
ES_HOST = "http://elasticsearch:9200"
INDEX   = "ufo_sightings"
ES      = Elasticsearch([ES_HOST], verify_certs=False)

FIELDS = [
    "Occurred_utc", "location", "shape", "State", "Country",
    "meteor_shower_codes", "days_from_shower_peak",
    "air_traffic_monthly", "light_pollution",
]

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def read_index(_cache_buster: float) -> pd.DataFrame:
    """
    Pull the index into a DataFrame and normalise nested fields.
    _cache_buster is a dummy parameter – when its value changes
    Streamlit invalidates the cache so we can force a reload.
    """
    hits = helpers.scan(
        ES,
        index=INDEX,
        query={"_source": FIELDS, "sort": ["_doc"]},
        size=1_500,
        scroll="3m",
    )
    df = pd.DataFrame(h["_source"] for h in hits)

    # ── normalise list / dict columns ────────────────────────────────────────
    df["meteor_shower_codes"] = df["meteor_shower_codes"].apply(
        lambda x: x if isinstance(x, list) else [])
    df["days_from_shower_peak"] = df["days_from_shower_peak"].apply(
        lambda x: x if isinstance(x, list) else [])
    df["air_traffic_monthly"] = df["air_traffic_monthly"].apply(
        lambda x: x if isinstance(x, dict) else {})
    
    # ── flatten light_pollution → brightness ────────────────────────────────
    df["light_pollution"] = df["light_pollution"].apply(
        lambda d: float(d.get("brightness", np.nan)) if isinstance(d, dict) else np.nan
    )

    # ── timestamps & conveniences ────────────────────────────────────────────
    df["ts"]        = (pd.to_datetime(df["Occurred_utc"], errors="coerce", utc=True)
                         .dt.tz_convert(None))
    df["month"]     = df["ts"].dt.to_period("M").dt.to_timestamp()
    df["n_showers"] = df["meteor_shower_codes"].str.len()

    df["traffic_total"]  = df["air_traffic_monthly"].apply(
        lambda d: d.get("total",  np.nan))
    df["traffic_region"] = df["air_traffic_monthly"].apply(
        lambda d: d.get("region"))

    return df

def prune_for_bokeh(tdf: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """Remove rows containing NaN/∞ in numeric cols and cast to plain float."""
    tdf = (tdf.replace([np.inf, -np.inf], np.nan)
              .dropna(subset=numeric_cols)
              .copy())
    for c in numeric_cols:
        tdf[c] = tdf[c].astype(float)
    return tdf

def add_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add meteor / flight / unknown boolean columns."""
    df = df.copy()

    # meteor-likely: at least one shower code within ±2 days of peak
    df["meteor_flag"] = (
        (df["n_showers"] > 0) &
        df["days_from_shower_peak"].apply(
            lambda lst: any(abs(d) <= 2 for d in lst or [])
        )
    )

    # flight-likely: traffic ≥ *median of its own region* (skip-na)
    reg_p90 = (df.groupby("traffic_region")["traffic_total"]
                 .transform(lambda s: np.nanpercentile(s, 90)))
    df["flight_flag"] = df["traffic_total"] >= 0.5 * reg_p90

    df["unknown_flag"] = ~(df["meteor_flag"] | df["flight_flag"])
    return df

def bokeh_hover():
    return HoverTool(
        tooltips=[
            ("Month",      "@month{%Y-%m}"),
            ("Sightings",  "@sightings{0,0}"),
            ("Avg showers","@avg_showers{0.0}"),
            ("Avg traffic","@avg_traffic{0,0}"),
        ],
        formatters={"@month": "datetime"},
        mode="vline",
    )

# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(layout="wide")
st.title("🌐 Global UFO Sightings Explorer")

# ↻  Reload-data button (cache buster) ----------------------------------------
if st.sidebar.button("↻ Reload data"):
    st.session_state["_cache_buster"] = time.time()

cache_buster = st.session_state.get("_cache_buster", 0.0)
df_all = read_index(cache_buster)

if df_all.empty:
    st.error("Index is empty – run ingest first."); st.stop()

# ── sidebar filters ──────────────────────────────────────────────────────────
min_date = pd.Timestamp("2006-01-01").date()
max_date = df_all["ts"].dt.date.max()

with st.sidebar:
    dr    = st.date_input("Date range", (min_date, max_date))
    shape = st.selectbox(
        "Shape", ["All"] + sorted(df_all["shape"].dropna().unique()))
    state = st.selectbox(
        "State", ["All"] + sorted(df_all["State"].dropna().unique()))
    n_map = st.slider("Points on map", 100, 5_000, 500, 100)

mask = (
    (df_all.ts >= pd.to_datetime(f"{dr[0]}T00:00:00")) &
    (df_all.ts <= pd.to_datetime(f"{dr[1]}T23:59:59"))
)
if shape != "All": mask &= df_all["shape"] == shape
if state != "All": mask &= df_all["State"] == state

df = add_flags(df_all[mask]).copy()
st.sidebar.markdown(f"**{len(df)} sightings** matched.")

# One common random-sample for both maps (latest first)
df_sample = df.sort_values("ts", ascending=False).head(n_map)

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Meteor-Shower Activity",
    "Air-Traffic Correlation",
    "Traffic × Sightings",
    "Light-Pollution Analysis",
])

# ════════════════════ Tab 1 – Meteor Showers ═════════════════════════════════
with tab1:
    st.subheader("Monthly UFO Counts vs. Average # Meteor Showers")
    ts1 = (df.groupby("month")
             .agg(sightings=("ts","count"), avg_showers=("n_showers","mean"))
             .reset_index())
    ts1 = prune_for_bokeh(ts1, ["sightings","avg_showers"])

    if ts1.empty:
        st.info("No data to plot.")
    else:
        src1             = ColumnDataSource(ts1)
        s_min,s_max      = ts1["sightings"].agg(["min","max"])
        sh_min,sh_max    = ts1["avg_showers"].agg(["min","max"])

        p = figure(x_axis_type="datetime", height=300, sizing_mode="stretch_width")
        p.y_range        = Range1d(s_min-1, s_max+1)
        p.extra_y_ranges = {"sh": Range1d(sh_min-0.2, sh_max+0.2)}
        p.add_layout(LinearAxis(y_range_name="sh",
                                axis_label="Avg # Showers"), "right")
        p.line("month","sightings",   source=src1, line_width=2,
               legend_label="Sightings")
        p.line("month","avg_showers", source=src1, line_width=2,
               color="orange", y_range_name="sh",
               legend_label="Avg # Showers")

        p.xaxis.formatter    = DatetimeTickFormatter(years="%Y")
        p.yaxis[0].formatter = NumeralTickFormatter(format="0")
        p.yaxis[1].formatter = NumeralTickFormatter(format="0.0")
        p.yaxis.axis_label   = "Sightings"
        p.legend.location    = "top_left"
        p.add_tools(bokeh_hover())
        st.bokeh_chart(p)

    # ── MAP (meteor vs unexplained) ─────────────────────────────────────────
    st.markdown("**Meteor-likely (orange) vs Unexplained (violet)**")

    fmap1 = folium.Map(location=[20,0], zoom_start=2,
                       tiles="CartoDB dark_matter", control_scale=True)

    layer_meteor  = folium.FeatureGroup(name="Meteor-likely",  show=True)
    layer_unknown = folium.FeatureGroup(name="Unexplained",    show=True)

    for _, r in df_sample.iterrows():
        lat, lon = r["location"]["lat"], r["location"]["lon"]

        if r["meteor_flag"]:
            folium.CircleMarker(
                [lat, lon], radius=5, color="orange",
                fill=True, fill_opacity=.7
            ).add_to(layer_meteor)

        elif r["unknown_flag"]:
            folium.CircleMarker(
                [lat, lon], radius=5, color="violet",
                fill=True, fill_opacity=.7
            ).add_to(layer_unknown)

    layer_meteor.add_to(fmap1)
    layer_unknown.add_to(fmap1)
    folium.LayerControl(collapsed=False).add_to(fmap1)
    st_folium(fmap1, width=900, height=500)

# ════════════════════ Tab 2 – Air Traffic ════════════════════════════════════
with tab2:
    st.subheader("Monthly UFO Counts vs. Average Monthly Air Traffic")

    ts2 = (df.groupby("month")
             .agg(sightings   = ("ts","count"),
                  avg_traffic = ("traffic_total","mean"),
                  region      = ("traffic_region","first"))
             .reset_index())
    ts2 = prune_for_bokeh(ts2, ["sightings","avg_traffic"])

    if ts2.empty:
        st.info("No air-traffic data for these filters.")
    else:
        src2             = ColumnDataSource(ts2)
        s_min,s_max      = ts2["sightings"].agg(["min","max"])
        tr_min,tr_max    = ts2["avg_traffic"].agg(["min","max"])

        q = figure(x_axis_type="datetime", height=300, sizing_mode="stretch_width")
        q.y_range        = Range1d(s_min-1, s_max+1)
        q.extra_y_ranges = {"tr": Range1d(tr_min-1000, tr_max+1000)}
        q.add_layout(LinearAxis(y_range_name="tr",
                                axis_label="Avg Traffic"), "right")
        q.line("month","sightings",   source=src2, line_width=2,
               legend_label="Sightings")
        q.line("month","avg_traffic", source=src2, line_width=2,
               color="green", y_range_name="tr",
               legend_label="Avg Traffic")

        q.xaxis.formatter    = DatetimeTickFormatter(years="%Y")
        q.yaxis[0].formatter = NumeralTickFormatter(format="0")
        q.yaxis[1].formatter = NumeralTickFormatter(format="0,0")
        q.yaxis.axis_label   = "Sightings"
        q.legend.location    = "top_left"
        q.add_tools(bokeh_hover())
        st.bokeh_chart(q)

    # ── MAP (flight vs unexplained) ────────────────────────────────────────
    st.markdown("**Flight-likely (blue) vs Unexplained (violet)**")

    fmap2 = folium.Map(location=[20,0], zoom_start=2,
                       tiles="CartoDB dark_matter", control_scale=True)

    layer_flight   = folium.FeatureGroup(name="Flight-likely", show=True)
    layer_unknown2 = folium.FeatureGroup(name="Unexplained",   show=True)

    for _, r in df_sample.iterrows():
        lat, lon = r["location"]["lat"], r["location"]["lon"]
        tot      = r["traffic_total"]

        if r["flight_flag"]:
            folium.CircleMarker(
                [lat, lon],
                radius= 5,
                color="blue", fill=True, fill_opacity=.7
            ).add_to(layer_flight)

        elif r["unknown_flag"]:
            folium.CircleMarker(
                [lat, lon],
                radius=5, color="violet",
                fill=True, fill_opacity=.6
            ).add_to(layer_unknown2)

    layer_flight.add_to(fmap2)
    layer_unknown2.add_to(fmap2)
    folium.LayerControl(collapsed=False).add_to(fmap2)
    st_folium(fmap2, width=900, height=500)

# ════════════════════ Tab 3 – Scatter ════════════════════════════════════════
with tab3:
    st.subheader("Is Monthly Air-Traffic Linked to UFO Sightings?")

    corr = (df.groupby("month")
              .agg(sightings=("ts","count"),
                   avg_traffic=("traffic_total","mean"))
              .dropna())
    corr = prune_for_bokeh(corr.reset_index(), ["sightings","avg_traffic"])

    if len(corr) < 3:
        st.info("Need at least three data-points.")
    else:
        src3 = ColumnDataSource(corr)
        scat = figure(height=350, sizing_mode="stretch_width",
                      x_axis_label="Avg Monthly Passengers",
                      y_axis_label="Sightings")
        scat.circle("avg_traffic", "sightings", size=8, source=src3,
                    fill_alpha=.6, line_width=0)

        # trend-line
        m, b = np.polyfit(corr["avg_traffic"], corr["sightings"], 1)
        xs   = np.linspace(corr["avg_traffic"].min(),
                           corr["avg_traffic"].max(), 2)
        scat.line(xs, m*xs+b, line_dash="dashed", color="black")

        scat.add_tools(HoverTool(
            tooltips=[
                ("Month",      "@month{%Y-%m}"),
                ("Passengers", "@avg_traffic{0,0}"),
                ("Sightings",  "@sightings{0,0}")
            ],
            formatters={"@month": "datetime"}
        ))
        scat.xaxis.formatter = NumeralTickFormatter(format="0,0")
        scat.yaxis.formatter = NumeralTickFormatter(format="0")
        st.bokeh_chart(scat)
# ════════════════════ Tab 4 – Light-Pollution Analysis ═════════════════════════════════
with tab4:
    st.subheader("Distribution of Light-Pollution by Sighting Type")

    # — 1) grab the three groups, filter out invalids
    groups = {
        "Meteor-likely": df.loc[df["meteor_flag"],   "light_pollution"],
        "Flight-likely": df.loc[df["flight_flag"],   "light_pollution"],
        "Unexplained":   df.loc[df["unknown_flag"],  "light_pollution"],
    }
    # drop NaN / inf
    for k, s in groups.items():
        groups[k] = s.replace([np.inf, -np.inf], np.nan).dropna()

    # — 2) merge all values to compute percentile-based cap
    all_vals = pd.concat(groups.values())
    if all_vals.empty:
        st.info("No light-pollution data available for these filters.")
    else:
        cap = np.percentile(all_vals, 99)               # 99th percentile cap
        floor = np.percentile(all_vals, 1)              # 1st percentile floor
        bins = np.linspace(floor, cap, 25)              # 24 bins between 1% and 99%

        p4 = figure(
            height=350, sizing_mode="stretch_width",
            x_axis_label="Radiance (VIIRS)", 
            y_axis_label="Count of Sightings"
        )

        colors = {
            "Meteor-likely": "orange",
            "Flight-likely": "blue",
            "Unexplained":   "gray"
        }

        # — 3) for each group, compute histogram on the capped range and draw a quad
        for label, vals in groups.items():
            # cap values > cap and < floor into the first/last bin
            clipped = np.clip(vals.values, floor, cap)
            counts, edges = np.histogram(clipped, bins=bins)
            # draw quads between edges[i] and edges[i+1]
            p4.quad(
                top=counts, bottom=0,
                left=edges[:-1], right=edges[1:],
                fill_color=colors[label], line_color=None,
                fill_alpha=0.6,
                legend_label=label
            )

        p4.legend.location = "top_right"
        p4.legend.click_policy = "hide"
        st.bokeh_chart(p4)

    st.markdown("**Map of Recent Sightings Colored by Light-Pollution**")
    df4 = df_sample.dropna(subset=["light_pollution"])
    if df4.empty:
        st.info("No light-pollution values on recent sightings.")
    else:
        cmap = linear.YlOrRd_09.scale(
            df4["light_pollution"].min(),
            df4["light_pollution"].quantile(0.99)  # same cap here
        )
        m4 = folium.Map(location=[20,0], zoom_start=2, tiles="CartoDB.Positron")
        for _, r in df4.iterrows():
            val = r["light_pollution"]
            shown = min(val, cap)
            CircleMarker(
                (r["location"]["lat"], r["location"]["lon"]),
                radius=5,
                fill=True, fill_opacity=0.8,
                color=cmap(shown),
                popup=f"Radiance: {val:.2f}"
            ).add_to(m4)

        # 🔧 ADD LEGEND HERE
        cmap.caption = "Radiance (VIIRS) - Light Pollution"
        cmap.add_to(m4)

        st_folium(m4, width=900, height=500)


with st.expander("ℹ️ About & Data Sources"):
    st.markdown("""
* **UFO reports**: NUFORC + Kaggle  
* **Meteor showers**: IAU Meteor Data Center  
* **Air-traffic**: SFO passenger totals (SF Open Data)  
* Built with **Streamlit**, **Bokeh**, **Folium** & **Elasticsearch**.
""")