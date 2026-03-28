# drought-monitoring

A Python package for computing the **Composite Drought Index (CDI)** over
20–30 year monitoring periods using Google Earth Engine, with in-memory
dask-parallelised spatial computation, Cloud Optimized GeoTIFF export, and
publication-quality climate science visualisations.

[![PyPI version](https://img.shields.io/pypi/v/drought-monitoring)](https://pypi.org/project/drought-monitoring/)
[![Python](https://img.shields.io/pypi/pyversions/drought-monitoring)](https://pypi.org/project/drought-monitoring/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Installation

```bash
# pip
pip install drought-monitoring

# uv
uv add drought-monitoring

# with all optional dependencies (GEE + COG + plots)
pip install "drought-monitoring[all]"
```

Optional extras:

| Extra | Installs | Use for |
|-------|----------|---------|
| `gee` | `earthengine-api` | fetching GEE data |
| `cog` | `rasterio`, `rioxarray` | COG export / import |
| `plot` | `matplotlib` | visualisation |
| `all` | all of the above | full workflow |

---

## Package structure

```
drought_monitoring/
├── core.py      CDI mathematics on pd.Series
├── spatial.py   pixel-wise computation on xr.DataArray (dask-parallelised)
├── gee.py       GEE authentication + ERA5-Land / MODIS xee cube fetching
├── io.py        Cloud Optimized GeoTIFF (COG) export and import
└── plot.py      publication-quality climate science figures
tests/
└── test_core.py
pyproject.toml
README.md
```

---

## Typical Jupyter notebook workflow

### 1. Authenticate GEE

```python
from drought_monitoring.gee import authenticate
authenticate(project="my-gee-project")   # opens browser on first use
```

### 2. Generate yearly drought maps (full in-memory pipeline)

```python
from drought_monitoring.gee import yearly_drought_maps

aoi = [38.0, 3.5, 42.5, 7.0]   # [lon_min, lat_min, lon_max, lat_max]
                                  # Borena region, Southern Ethiopia

# Streams ERA5 + MODIS via xee, computes CDI pixel-wise with dask,
# resamples to annual means — nothing written to disk
ds = yearly_drought_maps(aoi, start_year=2000, end_year=2020)
# xr.Dataset with variables: PDI, TDI, VDI, CDI
# dims: (year, latitude, longitude)
```

### 3. Plot all years

```python
ds["CDI"].plot(col="time", col_wrap=4, cmap="RdBu", robust=True, figsize=(20, 14))
```

### 4. Export to Cloud Optimized GeoTIFFs

```python
from drought_monitoring.io import cdi_stack_to_cog

paths = cdi_stack_to_cog(ds, output_dir="outputs/", prefix="Borena_2000_2020")
# outputs/Borena_2000_2020_PDI.tif
# outputs/Borena_2000_2020_TDI.tif
# outputs/Borena_2000_2020_VDI.tif
# outputs/Borena_2000_2020_CDI.tif
```

### 5. Visualise COGs interactively

```python
import leafmap

m = leafmap.Map(center=[5.0, 40.0], zoom=6)
m.add_cog_layer("outputs/Borena_2000_2020_CDI.tif", name="CDI")
m
```

---

## Area-averaged time series workflow

For a single-pixel or spatially-averaged CDI time series:

```python
from drought_monitoring.gee import fetch_era5_precip, fetch_era5_temp, fetch_modis_ndvi
from drought_monitoring import compute_all
from drought_monitoring.plot import plot_timeseries, plot_anomaly_bars

precip = fetch_era5_precip(aoi, start_year=2000, end_year=2020)
temp   = fetch_era5_temp(aoi,   start_year=2000, end_year=2020)
ndvi   = fetch_modis_ndvi(aoi,  start_year=2000, end_year=2020)

df = compute_all(precip, temp, ndvi, window=3)
# pd.DataFrame with columns: PDI, TDI, VDI, CDI

fig = plot_timeseries(df, title="CDI — Borena Region",
                      subtitle="ERA5-Land + MODIS MOD13A3  |  2000–2020")
fig.savefig("CDI_timeseries.png", dpi=300, bbox_inches="tight")

fig2 = plot_anomaly_bars(df, title="Annual Mean CDI Anomaly")
fig2.savefig("CDI_annual.png", dpi=300, bbox_inches="tight")
```

---

## Data sources

| Variable | GEE collection | Band | Units |
|----------|---------------|------|-------|
| Precipitation | `ECMWF/ERA5_LAND/MONTHLY_AGGR` | `total_precipitation_sum` | mm month⁻¹ |
| Temperature | `ECMWF/ERA5_LAND/MONTHLY_AGGR` | `temperature_2m` | °C |
| NDVI | `MODIS/061/MOD13A3` | `NDVI` | [−1, 1] |

---

## CDI formula

Each sub-index follows Burchard-Levine et al. (2024):

```
DI = (μ_IP / μ_LTM) × sqrt(RL_IP / RL_LTM)

CDI = 0.50 × PDI + 0.25 × TDI + 0.25 × VDI
```

| Symbol | Meaning |
|--------|---------|
| `μ_IP` | Trailing rolling mean over the interest period (default 3 months) |
| `μ_LTM` | Long-term mean of `μ_IP` for that calendar month |
| `RL_IP` | Count of months inside the IP where the anomaly condition holds |
| `RL_LTM` | Long-term mean of `RL_IP` for that calendar month |

PDI & VDI use **deficit** streaks (`raw < monthly LTM`).
TDI uses **excess** streaks (`raw > monthly LTM`).

---

## CDI severity classes

| CDI value | Class |
|-----------|-------|
| < 0.50 | Extreme drought |
| 0.50 – 0.65 | Severe drought |
| 0.65 – 0.80 | Moderate drought |
| 0.80 – 0.90 | Mild drought |
| 0.90 – 1.10 | Near normal |
| 1.10 – 1.20 | Mild wet |
| 1.20 – 1.30 | Moderately wet |
| > 1.30 | Very wet |

Values < 1 indicate drought; values ≈ 1 are near-normal; values > 1 are wetter than normal.

---

## Monitoring period

| Default | Minimum | Maximum |
|---------|---------|---------|
| 20 years | 20 years | 30 years |

An error is raised if the requested period is outside this range.

---

## COG structure

Each output GeoTIFF contains **one band per timestep**.
Band descriptions are ISO-8601 date strings (`YYYY-MM` or `YYYY`), so every
file is self-documenting and can be range-requested from cloud storage
(S3, GCS, Azure Blob) by leafmap, QGIS, or any GDAL tool.

---

## Running tests

```bash
pytest tests/ -v
```

---

## Reference

Based on: Burchard-Levine, V. et al. (2024). *pyCDI: a Python implementation
of the composite drought index.* EO-Africa R&D, ESA.
<https://github.com/VicenteBurchard/pyCDI>
