# drought-monitoring v0.1.0

A **lightweight Python package** for computing the **Composite Drought Index (CDI)**
over **20–30 year monitoring periods** using Google Earth Engine data,
writing outputs as **Cloud Optimized GeoTIFFs**, and producing
**publication-quality climate science visualisations**.

---

## Package structure

```
drought_cdi/
├── core.py     CDI mathematics on pd.Series
├── spatial.py  pixel-wise computation on xr.DataArray
├── gee.py      GEE authentication + ERA5-Land / MODIS data fetching
├── io.py       Cloud Optimized GeoTIFF (COG) export and import
├── plot.py     publication-quality climate science figures
└── tests/
    └── test_core.py
pyproject.toml
README.md
```

---

## Installation

```bash
# Core only (numpy, pandas, xarray)
uv pip install -e .

# Full workflow (GEE + COG + plots)
uv pip install -e ".[all]"
```

---

## Monitoring period

| Default | Minimum | Maximum |
|---------|---------|---------|
| 20 years | 20 years | 30 years |

An error is raised if the requested period is outside 20–30 years.

---

## Typical Jupyter notebook workflow

```python
# ── 1. Authenticate GEE ──────────────────────────────────────────────────────
from drought_cdi.gee import authenticate
authenticate(project="my-gee-project")   # opens browser on first use

# ── 2. Define your study area ────────────────────────────────────────────────
aoi = [38.0, 3.5, 42.5, 7.0]   # [lon_min, lat_min, lon_max, lat_max]
                                 # Borena region, Southern Ethiopia

# ── 3. Fetch 20-year monthly time series ────────────────────────────────────
from drought_cdi.gee import fetch_era5_precip, fetch_era5_temp, fetch_modis_ndvi

precip = fetch_era5_precip(aoi, start_year=2000, end_year=2020)  # mm/month
temp   = fetch_era5_temp(aoi,   start_year=2000, end_year=2020)  # °C
ndvi   = fetch_modis_ndvi(aoi,  start_year=2000, end_year=2020)  # [-1,1]

# ── 4. Compute CDI ───────────────────────────────────────────────────────────
from drought_cdi import compute_all

df = compute_all(precip, temp, ndvi, window=3)
# Returns pd.DataFrame with columns: PDI, TDI, VDI, CDI

# ── 5. Visualise ─────────────────────────────────────────────────────────────
from drought_cdi.plot import plot_timeseries, plot_anomaly_bars, plot_seasonal_cycle

fig = plot_timeseries(
    df,
    title="Composite Drought Index — Borena Region",
    subtitle="ERA5-Land + MODIS MOD09GQ  |  2000–2020",
)
fig.savefig("CDI_Borena_timeseries.png", dpi=300, bbox_inches="tight")

fig2 = plot_anomaly_bars(df, freq="Y", title="Annual Mean CDI Anomaly")
fig2.savefig("CDI_Borena_annual.png", dpi=300, bbox_inches="tight")

fig3 = plot_seasonal_cycle(df)
fig3.savefig("CDI_Borena_seasonal.png", dpi=300, bbox_inches="tight")

# ── 6. Export Cloud Optimized GeoTIFFs ──────────────────────────────────────
# Option A: from spatial xr.Dataset (pixel-wise)
from drought_cdi.gee import fetch_era5_precip_cube, fetch_era5_temp_cube
from drought_cdi.spatial import spatial_cdi
from drought_cdi.io import cdi_stack_to_cog

precip_cube = fetch_era5_precip_cube(aoi, start_year=2000, end_year=2020)
temp_cube   = fetch_era5_temp_cube(aoi,   start_year=2000, end_year=2020)
# (fetch ndvi cube similarly, or resample your time series)

ds = spatial_cdi(precip_cube, temp_cube, ndvi_cube)
paths = cdi_stack_to_cog(ds, output_dir="outputs/", prefix="Borena_2000_2020")
# → outputs/Borena_2000_2020_PDI.tif
# → outputs/Borena_2000_2020_TDI.tif
# → outputs/Borena_2000_2020_VDI.tif
# → outputs/Borena_2000_2020_CDI.tif

# Option B: broadcast area-averaged time series onto a spatial template
from drought_cdi.io import series_to_cog

paths = series_to_cog(df, spatial_ds=precip_cube, output_dir="outputs/",
                      prefix="Borena_2000_2020")

# ── 7. Visualise COGs in leafmap ─────────────────────────────────────────────
import leafmap
m = leafmap.Map(center=[5.0, 40.0], zoom=6)
m.add_raster(str(paths["CDI"]), colormap="RdBu", vmin=-3, vmax=3,
             layer_name="CDI 2020-12")
m
```

---

## Data sources

| Variable | GEE collection | Band | Units |
|----------|---------------|------|-------|
| Precipitation | `ECMWF/ERA5_LAND/MONTHLY_AGGR` | `total_precipitation_sum` | mm month⁻¹ |
| Temperature | `ECMWF/ERA5_LAND/MONTHLY_AGGR` | `temperature_2m` | °C |
| NDVI | `MODIS/061/MOD09GQ` | derived from b01+b02 | − |

---

## CDI formula

```
DI = (IP − LTM) / LTM  +  run / LTM_run
CDI = (PDI + TDI + VDI) / 3   [default equal weights]
```

| Symbol | Meaning |
|--------|---------|
| IP | Interest-period value (3-month trailing rolling mean) |
| LTM | Long-term mean for that calendar month |
| run | Consecutive months where the anomaly condition is active |
| LTM_run | Long-term mean of run lengths |

PDI & VDI use **deficit** streaks (IP < LTM). TDI uses **excess** streaks (IP > LTM).

---

## CDI severity classes

| CDI value | Class |
|-----------|-------|
| < −2.0 | Extreme drought |
| −2.0 – −1.5 | Severe drought |
| −1.5 – −1.0 | Moderate drought |
| −1.0 – −0.5 | Mild drought |
| −0.5 – +0.5 | Near normal |
| +0.5 – +1.0 | Mild wet |
| +1.0 – +1.5 | Moderately wet |
| > +1.5 | Very wet |

---

## COG structure

Each output GeoTIFF contains **one band per month**.
Band descriptions are ISO-8601 date strings (`YYYY-MM`), so every file
is self-documenting and can be range-requested from cloud storage
(S3, GCS, Azure Blob) by leafmap, QGIS, or any GDAL tool.

---

## Running tests

```bash
pytest drought_cdi/tests/ -v
```

---

## Reference

Based on: Burchard-Levine, V. et al. (2024). *pyCDI: a Python implementation
of the composite drought index.* EO-Africa R&D, ESA.
<https://github.com/VicenteBurchard/pyCDI>