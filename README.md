# Global UFO Sightings Explorer

An end-to-end platform to ingest, enrich, and visualize global UFO reports alongside meteor showers, air traffic, and light pollution.

---

## 📁 Project Structure

```text
.
├── dashboard/
│   ├── dashboard.py
│   ├── Dockerfile
│   └── requirements.txt
├── ingest/
│   ├── ufo_ingest.py
│   ├── meteor_enrich.py
│   ├── light_pollution_enrich.py
│   ├── flight_enrich.py
│   ├── check_enrich.py
│   ├── Dockerfile
│   └── requirements.txt
├── data/
│   ├── nuforc.json
│   ├── ufo_sightings.csv
│   ├── air_traffic_data.csv
│   └── meteor_showers.csv
├── docker-compose.yml
└── README.md   ← you are here
```
---

## ⚙️ Prerequisites

- **Docker** (Engine ≥ 19.03)  
- **Docker Compose** (v1.27+)  

---

## 🔑 Environment Variables

Create a `.env` file in the project root with your API keys:

```dotenv
# Elasticsearch (optional override)
ES_HOST=http://elasticsearch:9200

# Kaggle (for UFO CSV)
KAGGLE_USERNAME=<your_username>
KAGGLE_KEY=<your_key>

# Optional enrichment APIs
NOAA_TOKEN=<your_noaa_token>
OPENSKY_USER=<your_opensky_user>
OPENSKY_PASS=<your_opensky_pass>
WEATHERBIT_KEY=<your_weatherbit_key>
```
---

## 🚀 Running with Docker Compose

**1. Start Elasticsearch & Kibana**
```bash
docker-compose up -d elasticsearch kibana
```
Wait until the Elasticsearch healthcheck passes.

**2. Build & run the Ingest service**
```bash
docker-compose build ingest
docker-compose up ingest
```
This will:
	•	Merge NUFORC + Kaggle UFO data
	•	Parse and clean timestamps and locations
	•	Enrich with meteor showers, light pollution, and air traffic
	•	Index into the Elasticsearch ufo_sightings index

 **3. Build & run the Dashboard service**
 ```bash
docker-compose build dashboard
docker-compose up dashboard
```
•	Visit http://localhost:8000 to explore the Streamlit dashboard.

 **4. Tear down**
 ```bash
docker-compose down
```
