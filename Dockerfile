FROM python:3.11-slim

LABEL maintainer="pm25-hyperlocal"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal-dev gdal-bin libgeos-dev libproj-dev \
    libhdf5-dev libhdf5-hld-0 libhdf5-openmpi-dev \
    libnetcdf-dev \
    && rm -rf /var/lib/apt/lists/*

ENV CPLUS_INCLUDE_PATH=/usr/include/gdal:/usr/include/hdf5/serial
ENV C_INCLUDE_PATH=/usr/include/gdal:/usr/include/hdf5/serial

WORKDIR /app

COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

COPY . .

ARG BUILD_VERSION=unknown
ENV BUILD_VERSION=${BUILD_VERSION}

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
