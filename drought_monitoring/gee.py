"""
drought_monitoring.gee
===============
Google Earth Engine (GEE) authentication and data-fetching helpers.

Workflow in a Jupyter notebook
-------------------------------
::

    import ee
    from drought_cdi.gee import authenticate, fetch_era5_precip, fetch_era5_temp, fetch_modis_ndvi

    # Step 1 – one-time interactive authentication
    authenticate(project="my-gee-project")

    # Step 2 – define your area of interest as a bounding box
    aoi = [38.0, 3.5, 42.5, 7.0]   # [lon_min, lat_min, lon_max, lat_max] – Borena, Ethiopia

    # Step 3 – fetch 20-year monthly time series (area-averaged)
    precip = fetch_era5_precip(aoi, start_year=2000, end_year=2020)
    temp   = fetch_era5_temp(aoi,   start_year=2000, end_year=2020)
    ndvi   = fetch_modis_ndvi(aoi,  start_year=2000, end_year=2020)

    # Step 4 – compute CDI
    from drought_cdi import compute_all
    df = compute_all(precip, temp, ndvi)

Data sources
------------
- Precipitation & Temperature : ERA5-Land Monthly Aggregated
  ``ECMWF/ERA5_LAND/MONTHLY_AGGR``
- NDVI                        : MODIS Terra Surface Reflectance 8-Day
  ``MODIS/061/MOD09GQ`` – bands sur_refl_b02 (NIR) & sur_refl_b01 (Red)

Monitoring period
-----------------
Default monitoring period is **20 years**; supported range is 20–30 years.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import xarray as xr

try:
    import ee
    _HAS_EE = True
except ImportError:
    _HAS_EE = False

try:
    import xee as _xee  # noqa: F401 – registers the "ee" xarray engine
    _HAS_XEE = True
except ImportError:
    _HAS_XEE = False


_MIN_YEARS = 20
_MAX_YEARS = 30
_ERA5_COLLECTION   = "ECMWF/ERA5_LAND/MONTHLY_AGGR"
_MODIS_COLLECTION  = "MODIS/061/MOD13A3"


# Authentication
def authenticate(project: Optional[str] = None, quiet: bool = False) -> None:
    """
    Authenticate and initialise the Google Earth Engine Python API.

    Call this **once** at the top of your Jupyter notebook.  On first ever
    use it opens a browser tab for OAuth consent; on subsequent calls in the
    same Python session it is a silent no-op if already initialised.

    Parameters
    ----------
    project : str, optional
        GEE Cloud project ID (required for most accounts since 2023).
        Example: ``'my-gee-project-123'``.
    quiet : bool
        Suppress the confirmation banner.

    Raises
    ------
    ImportError  – if ``earthengine-api`` is not installed.
    RuntimeError – if authentication or initialisation fails.

    Example
    -------
    >>> from drought_monitoring.gee import authenticate
    >>> authenticate(project="my-project")
    ✓ GEE authenticated and initialised successfully.
    """
    _require_ee()
    if _is_initialised():
        if not quiet:
            print("✓ GEE already initialised.")
        return
    try:
        ee.Authenticate()
        kwargs: dict = {}
        if project:
            kwargs["project"] = project
        ee.Initialize(**kwargs)
        if not quiet:
            print("✓ GEE authenticated and initialised successfully.")
    except Exception as exc:
        raise RuntimeError(
            "GEE authentication failed.\n"
            "  • Make sure you have run  earthengine authenticate  in your terminal at\n"
            "    least once to store credentials.\n"
            "  • Pass project='<your-cloud-project-id>' if required by your account.\n"
            f"Original error: {exc}"
        ) from exc

def _require_ee() -> None:
    if not _HAS_EE:
        raise ImportError(
            "earthengine-api is required for GEE data access.\n"
            "Install it with:  pip install earthengine-api"
        )


def _require_xee() -> None:
    if not _HAS_XEE:
        raise ImportError(
            "xee is required for lazy in-memory cube fetching.\n"
            "Install it with:  pip install xee"
        )


def _is_initialised() -> bool:
    try:
        ee.Number(1).getInfo()
        return True
    except Exception:
        return False


def _require_initialised() -> None:
    _require_ee()
    if not _is_initialised():
        raise RuntimeError(
            "GEE is not initialised.  "
            "Call  drought_monitoring.gee.authenticate()  first."
        )


def _parse_aoi(aoi) -> "ee.Geometry":
    """
    Coerce *aoi* to an ``ee.Geometry``.

    Accepted inputs
    ~~~~~~~~~~~~~~~
    * ``ee.Geometry`` / ``ee.Feature`` / ``ee.FeatureCollection``
    * GeoJSON ``dict``
    * Bounding-box list / tuple  ``[lon_min, lat_min, lon_max, lat_max]``
    * Polygon ring  ``[[lon, lat], ...]``
    """
    if isinstance(aoi, dict):
        return ee.Geometry(aoi)
    if isinstance(aoi, (list, tuple)):
        if len(aoi) == 4 and all(isinstance(v, (int, float)) for v in aoi):
            lon_min, lat_min, lon_max, lat_max = aoi
            return ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])
        return ee.Geometry.Polygon([list(c) for c in aoi])
    if hasattr(aoi, "geometry"):
        return aoi.geometry()
    return aoi


def _validate_period(start_year: int, end_year: int) -> None:
    n_years = end_year - start_year + 1
    if n_years < _MIN_YEARS:
        raise ValueError(
            f"Monitoring period must be at least {_MIN_YEARS} years "
            f"(got {n_years}: {start_year}–{end_year})."
        )
    if n_years > _MAX_YEARS:
        raise ValueError(
            f"Monitoring period must be at most {_MAX_YEARS} years "
            f"(got {n_years}: {start_year}–{end_year})."
        )


def _date_range(start_year: int, end_year: int) -> tuple[str, str]:
    return f"{start_year}-01-01", f"{end_year}-12-31"


def _collection_to_series(
    collection: "ee.ImageCollection",
    aoi: "ee.Geometry",
    band: str,
    scale: int,
) -> pd.Series:
    """
    Reduce every image in a monthly collection to an area-averaged scalar
    and return a ``pd.Series`` with a monthly ``DatetimeIndex``.
    """
    def _reduce(img):
        val = img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=scale,
            maxPixels=1e10,
            bestEffort=True,
        ).get(band, None)
        return ee.Feature(None, {
            "yyyymm": img.date().format("YYYY-MM"),
            "value":  val,
        })

    info = collection.map(_reduce).getInfo()
    if info is None:
        raise ValueError("GEE returned no data. Check your AOI, date range, and authentication.")
    feats = info["features"]
    records = [
        (f["properties"]["yyyymm"], f["properties"]["value"])
        for f in feats
        if f["properties"]["value"] is not None
    ]
    if not records:
        raise ValueError(
            f"No data returned for band '{band}'. "
            "Check your AOI, date range, and GEE authentication."
        )
    records.sort()
    dates, values = zip(*records)
    idx = pd.to_datetime([f"{d}-01" for d in dates])
    return pd.Series(list(values), index=idx, dtype=np.float64)

def fetch_era5_precip(
    aoi,
    start_year: int,
    end_year:   Optional[int] = None,
    scale:      int = 1000,
) -> pd.Series:
    """
    Fetch ERA5-Land monthly total precipitation (area-averaged).

    Parameters
    ----------
    aoi        : GEE geometry
        Bounding box ``[lon_min, lat_min, lon_max, lat_max]``,
        GeoJSON dict, polygon ring, or any ``ee.Geometry``.
    start_year : int
        First year of the monitoring period.
    end_year   : int, optional
        Last year (inclusive).  Defaults to ``start_year + 19`` (20 years).
    scale      : int
        Reduction scale in metres (default 1 000).

    Returns
    -------
    pd.Series
        Monthly precipitation in **mm month⁻¹** with a ``DatetimeIndex``.
    """
    _require_initialised()
    end_year = end_year or (start_year + _MIN_YEARS - 1)
    _validate_period(start_year, end_year)
    aoi_geom   = _parse_aoi(aoi)
    start, end = _date_range(start_year, end_year)

    col  = (
        ee.ImageCollection(_ERA5_COLLECTION)
        .filterDate(start, end)
        .filterBounds(aoi_geom)
        .select("total_precipitation_sum")
    )
    s = _collection_to_series(col, aoi_geom, "total_precipitation_sum", scale)
    s = s * 1000.0        # m → mm
    s.name = "precip_mm"
    return s


def fetch_era5_temp(
    aoi,
    start_year: int,
    end_year:   Optional[int] = None,
    scale:      int = 1000,
) -> pd.Series:
    """
    Fetch ERA5-Land monthly mean 2-m temperature (area-averaged).

    Parameters
    ----------
    aoi        : GEE geometry
    start_year : int
    end_year   : int, optional  (default: ``start_year + 19``)
    scale      : int  (default 1 000 m)

    Returns
    -------
    pd.Series
        Monthly mean temperature in **°C** with a ``DatetimeIndex``.
    """
    _require_initialised()
    end_year = end_year or (start_year + _MIN_YEARS - 1)
    _validate_period(start_year, end_year)
    aoi_geom   = _parse_aoi(aoi)
    start, end = _date_range(start_year, end_year)

    col  = (
        ee.ImageCollection(_ERA5_COLLECTION)
        .filterDate(start, end)
        .filterBounds(aoi_geom)
        .select("temperature_2m")
    )
    s = _collection_to_series(col, aoi_geom, "temperature_2m", scale)
    s = s - 273.15        # K → °C
    s.name = "temp_c"
    return s


def fetch_modis_ndvi(
    aoi,
    start_year: int,
    end_year:   Optional[int] = None,
    scale:      int = 1000,
) -> pd.Series:
    """
    Fetch MODIS MOD13A3 monthly NDVI (area-averaged).

    Uses the MODIS Terra Vegetation Indices Monthly L3 Global 1 km product
    (``MODIS/061/MOD13A3``), which provides pre-composited monthly NDVI.
    The raw band values are scaled by 0.0001 to give NDVI in [−1, 1].

    Parameters
    ----------
    aoi        : GEE geometry
    start_year : int
    end_year   : int, optional  (default: ``start_year + 19``)
    scale      : int  (default 1000 m — native MOD13A3 resolution)

    Returns
    -------
    pd.Series
        Monthly NDVI in **[−1, 1]** with a ``DatetimeIndex``.
    """
    _require_initialised()
    end_year = end_year or (start_year + _MIN_YEARS - 1)
    _validate_period(start_year, end_year)
    aoi_geom   = _parse_aoi(aoi)
    start, end = _date_range(start_year, end_year)

    col = (
        ee.ImageCollection(_MODIS_COLLECTION)
        .filterDate(start, end)
        .filterBounds(aoi_geom)
        .select("NDVI")
    )
    s = _collection_to_series(col, aoi_geom, "NDVI", scale)
    s = s * 0.0001        # scale factor → [−1, 1]
    s.name = "ndvi"
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Spatial cube fetchers  (returns xr.DataArray for COG export)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_era5_precip_cube(
    aoi,
    start_year: int,
    end_year:   Optional[int] = None,
    scale:      int = 1000,
) -> "xr.DataArray":
    """
    Fetch ERA5-Land monthly precipitation as a ``(time, lat, lon)`` cube.

    Returns
    -------
    xr.DataArray  in mm month⁻¹, ready for :func:`drought_cdi.io.to_cog`.
    """
    end_year = end_year or (start_year + _MIN_YEARS - 1)
    return _fetch_era5_cube(
        aoi, start_year, end_year,
        band="total_precipitation_sum",
        name="precip_mm",
        multiplier=1000.0,
        offset=0.0,
        scale=scale,
    )


def fetch_era5_temp_cube(
    aoi,
    start_year: int,
    end_year:   Optional[int] = None,
    scale:      int = 1000,
) -> "xr.DataArray":
    """
    Fetch ERA5-Land monthly 2-m temperature as a ``(time, lat, lon)`` cube.

    Returns
    -------
    xr.DataArray  in °C, ready for :func:`drought_cdi.io.to_cog`.
    """
    end_year = end_year or (start_year + _MIN_YEARS - 1)
    return _fetch_era5_cube(
        aoi, start_year, end_year,
        band="temperature_2m",
        name="temp_c",
        multiplier=1.0,
        offset=-273.15,
        scale=scale,
    )


def _fetch_era5_cube(
    aoi,
    start_year:  int,
    end_year:    int,
    band:        str,
    name:        str,
    multiplier:  float,
    offset:      float,
    scale:       int,
) -> "xr.DataArray":
    import xarray as xr
    _require_initialised()
    _validate_period(start_year, end_year)
    aoi_geom   = _parse_aoi(aoi)
    start, end = _date_range(start_year, end_year)

    col    = (
        ee.ImageCollection(_ERA5_COLLECTION)
        .filterDate(start, end)
        .filterBounds(aoi_geom)
        .select(band)
    )
    imgs   = col.toList(col.size())
    n      = col.size().getInfo()
    bounds_info = aoi_geom.bounds().getInfo()
    if bounds_info is None:
        raise ValueError("GEE could not resolve AOI bounds.")
    bounds = bounds_info["coordinates"][0]
    lon_min = min(c[0] for c in bounds)
    lat_min = min(c[1] for c in bounds)
    lon_max = max(c[0] for c in bounds)
    lat_max = max(c[1] for c in bounds)

    dates, slices = [], []
    for i in range(n):
        img  = ee.Image(imgs.get(i))
        date = img.date().format("YYYY-MM-dd").getInfo()
        if date is None:
            raise ValueError(f"GEE returned no date for image index {i}.")
        dates.append(pd.Timestamp(date))
        arr  = np.array(
            img.sampleRectangle(region=aoi_geom, defaultValue=-9999)
               .get(band).getInfo(),
            dtype=np.float64,
        )
        arr[arr == -9999] = np.nan
        slices.append(arr)

    data   = np.stack(slices, axis=0) * multiplier + offset
    n_lat  = data.shape[1]
    n_lon  = data.shape[2]
    lats   = np.linspace(lat_max, lat_min, n_lat)
    lons   = np.linspace(lon_min, lon_max, n_lon)

    da = xr.DataArray(
        data,
        dims=["time", "latitude", "longitude"],
        coords={"time": dates, "latitude": lats, "longitude": lons},
        name=name,
    )
    da.attrs.update({"crs": "EPSG:4326", "source": _ERA5_COLLECTION, "band": band})
    return da


# ─────────────────────────────────────────────────────────────────────────────
# xee-based lazy cube fetchers  (in-memory, dask-backed, no disk I/O)
# ─────────────────────────────────────────────────────────────────────────────

def open_era5_cube(
    aoi,
    start_year: int,
    end_year:   Optional[int] = None,
    scale:      float = 0.1,
    chunks:     Optional[dict] = None,
) -> "xr.Dataset":
    """
    Open ERA5-Land monthly precipitation and temperature as a lazy
    dask-backed ``xr.Dataset`` via xee — **no data is downloaded** until
    ``.compute()`` is called.

    Parameters
    ----------
    aoi        : ``[lon_min, lat_min, lon_max, lat_max]`` or any GEE geometry.
    start_year : int
    end_year   : int, optional  (default: ``start_year + 19``).
    scale      : float  spatial resolution in **degrees** (default 0.1 ≈ 11 km).
    chunks     : dict   dask chunk spec (default ``{'time': 12}``).

    Returns
    -------
    xr.Dataset
        Variables ``precip_mm`` (mm month⁻¹) and ``temp_c`` (°C).
        Dims: ``(time, latitude, longitude)``.
    """
    import xarray as xr
    _require_xee()
    _require_initialised()
    end_year = end_year or (start_year + _MIN_YEARS - 1)
    _validate_period(start_year, end_year)
    aoi_geom   = _parse_aoi(aoi)
    start, end = _date_range(start_year, end_year)
    chunks     = chunks or {"time": 12}

    col = (
        ee.ImageCollection(_ERA5_COLLECTION)
        .filterDate(start, end)
        .filterBounds(aoi_geom)
        .select(["total_precipitation_sum", "temperature_2m"])
    )

    ds = xr.open_dataset(
        col,
        engine="ee",
        crs="EPSG:4326",
        scale=scale,
        geometry=aoi_geom,
        chunks=chunks,
    )
    # Unit conversions are lazy — no data is fetched here
    ds["total_precipitation_sum"] = ds["total_precipitation_sum"] * 1_000.0  # m → mm
    ds["temperature_2m"]          = ds["temperature_2m"] - 273.15            # K → °C
    return ds.rename({"total_precipitation_sum": "precip_mm",
                      "temperature_2m": "temp_c"})


def open_modis_ndvi(
    aoi,
    start_year: int,
    end_year:   Optional[int] = None,
    scale:      float = 0.1,
    chunks:     Optional[dict] = None,
) -> "xr.DataArray":
    """
    Open MODIS MOD13A3 monthly NDVI as a lazy dask-backed ``xr.DataArray``
    via xee — **no data is downloaded** until ``.compute()`` is called.

    Parameters
    ----------
    aoi        : ``[lon_min, lat_min, lon_max, lat_max]`` or any GEE geometry.
    start_year : int
    end_year   : int, optional  (default: ``start_year + 19``).
    scale      : float  spatial resolution in degrees (default 0.1).
    chunks     : dict   dask chunk spec (default ``{'time': 12}``).

    Returns
    -------
    xr.DataArray  NDVI in [−1, 1].  Dims: ``(time, latitude, longitude)``.
    """
    import xarray as xr
    _require_xee()
    _require_initialised()
    end_year = end_year or (start_year + _MIN_YEARS - 1)
    _validate_period(start_year, end_year)
    aoi_geom   = _parse_aoi(aoi)
    start, end = _date_range(start_year, end_year)
    chunks     = chunks or {"time": 12}

    col = (
        ee.ImageCollection(_MODIS_COLLECTION)
        .filterDate(start, end)
        .filterBounds(aoi_geom)
        .select("NDVI")
    )

    ds = xr.open_dataset(
        col,
        engine="ee",
        crs="EPSG:4326",
        scale=scale,
        geometry=aoi_geom,
        chunks=chunks,
    )
    ndvi      = ds["NDVI"] * 0.0001   # raw DN → [−1, 1]
    ndvi.name = "ndvi"
    return ndvi


def yearly_drought_maps(
    aoi,
    start_year: int,
    end_year:   Optional[int] = None,
    window:     int   = 3,
    weights:    tuple = (0.50, 0.25, 0.25),
    scale:      float = 0.1,
    chunks:     Optional[dict] = None,
) -> "xr.Dataset":
    """
    Full in-memory pipeline: stream ERA5 + MODIS via xee, compute CDI
    pixel-wise in parallel with dask, and return **annual mean drought maps**.
    Nothing is written to disk at any stage.

    Steps
    -----
    1. Open ERA5 precip + temp as a lazy dask cube (xee).
    2. Open MODIS NDVI as a lazy dask cube (xee).
    3. Align the MODIS time axis and grid to ERA5 (bilinear interpolation).
    4. Compute PDI / TDI / VDI / CDI pixel-wise (dask-parallelised).
    5. Resample monthly maps to **yearly means**.
    6. Trigger ``.compute()`` — data are fetched from GEE and all dask
       tasks execute; the result fits in RAM.

    Parameters
    ----------
    aoi        : ``[lon_min, lat_min, lon_max, lat_max]`` or GEE geometry.
    start_year : int
    end_year   : int, optional  (default: ``start_year + 19``).
    window     : int   IP window in months (default 3).
    weights    : tuple ``(w_PDI, w_TDI, w_VDI)`` — must sum to 1.0.
    scale      : float  spatial resolution in degrees (default 0.1 ≈ 11 km).
    chunks     : dict   dask chunk spec (default ``{'time': 12}``).

    Returns
    -------
    xr.Dataset
        Variables ``PDI``, ``TDI``, ``VDI``, ``CDI`` on a
        ``(year, latitude, longitude)`` grid.  All data is in memory.

    Example
    -------
    >>> from drought_monitoring.gee import authenticate, yearly_drought_maps
    >>> authenticate(project="my-project")
    >>> aoi = [38.0, 3.5, 42.5, 7.0]   # Borena, Ethiopia
    >>> ds  = yearly_drought_maps(aoi, start_year=2000, end_year=2020)
    >>> ds["CDI"].sel(time="2015").plot()
    """
    import numpy as np
    from .spatial import yearly_drought_maps as _spatial_yearly

    if not np.isclose(sum(weights), 1.0, atol=1e-6):
        raise ValueError(f"weights must sum to 1.0, got {sum(weights):.6f}.")

    end_year = end_year or (start_year + _MIN_YEARS - 1)
    chunks   = chunks or {"time": 12}

    # 1 & 2 — lazy dask cubes (nothing downloaded yet)
    era5_ds = open_era5_cube(aoi, start_year, end_year, scale=scale, chunks=chunks)
    ndvi_da = open_modis_ndvi(aoi, start_year, end_year, scale=scale, chunks=chunks)

    precip = era5_ds["precip_mm"]
    temp   = era5_ds["temp_c"]

    # 3 — align MODIS grid and time axis to ERA5 (bilinear, still lazy)
    ndvi = ndvi_da.interp_like(precip, method="linear")

    # 4–6 — pixel-wise CDI → yearly means → materialise
    return _spatial_yearly(precip, temp, ndvi, window=window, weights=weights)