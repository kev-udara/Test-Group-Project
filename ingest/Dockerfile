FROM python:3.10-slim

# 1. Install OS-level build tools + GDAL
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      gdal-bin \
      libgdal-dev \
      python3-dev && \
    rm -rf /var/lib/apt/lists/*

# 2. Point pip at GDAL’s config
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal
ENV GDAL_CONFIG=/usr/bin/gdal-config

WORKDIR /app

# 3. Copy & install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy your ingestion scripts in
COPY *.py .

CMD ["python","ufo_ingest.py"]