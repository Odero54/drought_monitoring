"""
drought_monitoring.io
==============
Input / output utilities for drought_cdi products.

Output format : **Cloud Optimized GeoTIFF (COG)**
---------------------------------------------------
All raster outputs are written as COGs so they can be:

* Streamed and visualised directly in **leafmap** without local download.
* Shared via HTTP (S3, GCS, Azure Blob) with range-request support.
* Ingested into any GDAL-compatible tool (QGIS, ArcGIS, rasterio, etc.).

COG structure
-------------
Each output file contains **one band per timestep** (months).  Band
descriptions are ISO-8601 date strings (``YYYY-MM``), so the temporal
dimension is self-documenting inside the file metadata.

Functions
---------
to_cog          – write an xr.DataArray (time, lat, lon) to a COG
cdi_stack_to_cog – write the full PDI/TDI/VDI/CDI stack, one COG per variable
read_cog        – read a COG back into an xr.DataArray
list_cog_dates  – extract the date metadata from a COG's band descriptions
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

try:
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import from_bounds
    from rasterio import MemoryFile
    from rasterio.enums import Resampling
    import rasterio.shutil as rio_shutil
    _HAS_RASTERIO = True
except ImportError:
    _HAS_RASTERIO = False

try:
    import xarray as xr
    _HAS_XR = True
except ImportError:
    _HAS_XR = False


def _require_rasterio() -> None:
    if not _HAS_RASTERIO:
        raise ImportError(
            "rasterio is required for COG I/O.\n"
            "Install it with:  pip install rasterio"
        )

def _require_xarray() -> None:
    if not _HAS_XR:
        raise ImportError(
            "xarray is required.\n"
            "Install it with:  pip install xarray"
        )

def _cog_profile(
    n_bands: int,
    height:  int,
    width:   int,
    transform,
    crs_str: str = "EPSG:4326",
    nodata:  float = np.nan,
    dtype:   str = "float32",
) -> dict:
    """Return a rasterio profile dict configured for COG output."""
    return {
        "driver":   "GTiff",
        "dtype":    dtype,
        "nodata":   nodata,
        "width":    width,
        "height":   height,
        "count":    n_bands,
        "crs":      CRS.from_user_input(crs_str),
        "transform": transform,
        "tiled":    True,
        "blockxsize": 512,
        "blockysize": 512,
        "compress": "deflate",
        "predictor": 3,          # floating-point delta predictor
        "zlevel":   6,
        "interleave": "band",
    }

def to_cog(
    da: "xr.DataArray",
    path: Union[str, Path],
    crs: str = "EPSG:4326",
    nodata: float = np.nan,
    dtype: str = "float32",
    overview_levels: Optional[list[int]] = None,
    lat_dim: str = "latitude",
    lon_dim: str = "longitude",
    time_dim: str = "time",
) -> Path:
    """
    Write an ``xr.DataArray`` with dims ``(time, lat, lon)`` to a
    **Cloud Optimized GeoTIFF**.

    Each time-step becomes one band.  Band descriptions are set to the
    ISO-8601 date string of each timestep so the file is self-documenting.

    Parameters
    ----------
    da       : xr.DataArray
        Input cube with dims ``(time, latitude, longitude)`` or equivalent.
        The ``time`` coordinate should be datetime-like.
    path     : str | Path
        Output file path (e.g. ``'output/CDI_2000_2020.tif'``).
    crs      : str
        EPSG code or WKT string for the output CRS (default ``'EPSG:4326'``).
    nodata   : float
        No-data fill value written into the GeoTIFF (default NaN).
    dtype    : str
        Output dtype (default ``'float32'``).
    overview_levels : list of int, optional
        Pyramid overview factors.  Defaults to ``[2, 4, 8, 16, 32]``.
    lat_dim  : str  – name of the latitude  dimension (default ``'latitude'``).
    lon_dim  : str  – name of the longitude dimension (default ``'longitude'``).
    time_dim : str  – name of the time dimension      (default ``'time'``).

    Returns
    -------
    Path  – resolved path of the written COG.

    Example
    -------
    >>> from drought_cdi.io import to_cog
    >>> to_cog(ds["CDI"], "outputs/CDI_Borena_2000_2020.tif")
    PosixPath('outputs/CDI_Borena_2000_2020.tif')
    """
    _require_rasterio()
    _require_xarray()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if overview_levels is None:
        overview_levels = [2, 4, 8, 16, 32]
    lat_dim = str(next((d for d in da.dims if d in (lat_dim, "lat", "y")), lat_dim))
    lon_dim = str(next((d for d in da.dims if d in (lon_dim, "lon", "x")), lon_dim))
    time_dim = str(next((d for d in da.dims if d in (time_dim, "t")), time_dim))
    lats = da[lat_dim].values
    lons = da[lon_dim].values
    times = da[time_dim].values

    height = len(lats)
    width  = len(lons)

    lat_min, lat_max = float(lats.min()), float(lats.max())
    lon_min, lon_max = float(lons.min()), float(lons.max())

    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, width, height)

    def _fmt_date(t) -> str:
        ts = pd.Timestamp(t)
        return ts.strftime("%Y-%m")

    band_descriptions = [_fmt_date(t) for t in times]
    n_bands = len(times)

    profile = _cog_profile(n_bands, height, width, transform, crs, nodata, dtype)
    data = da.values.astype(np.float32)
    if data.ndim == 2:                    # single time-step
        data = data[np.newaxis, :, :]
    if lats[0] < lats[-1]:               # bottom-up → flip to top-down
        data  = data[:, ::-1, :]
        transform = from_bounds(lon_min, lat_min, lon_max, lat_max, width, height)
    with MemoryFile() as mem:
        with mem.open(**profile) as tmp:
            tmp.write(data)
            for i, desc in enumerate(band_descriptions, start=1):
                tmp.update_tags(i, date=desc)
                tmp.set_band_description(i, desc)
            valid_levels = [level for level in overview_levels if min(height, width) // level >= 2]
            if valid_levels:
                tmp.build_overviews(valid_levels, Resampling.average)
            tmp.update_tags(ns="rio_overview", resampling="average")
        with mem.open() as tmp:
            rio_shutil.copy(tmp, str(path), driver="GTiff", copy_src_overviews=True, **{
                "tiled":       True,
                "blockxsize":  512,
                "blockysize":  512,
                "compress":    "deflate",
                "predictor":   3,
                "zlevel":      6,
            })

    file_mb = path.stat().st_size / 1_048_576
    print(
        f"✓ COG written → {path}  "
        f"({n_bands} bands, {height}×{width} px, {file_mb:.2f} MB)"
    )
    return path.resolve()


def cdi_stack_to_cog(
    ds: "xr.Dataset",
    output_dir: Union[str, Path],
    prefix: str = "CDI",
    crs: str = "EPSG:4326",
    **kwargs,
) -> dict[str, Path]:
    """
    Write PDI, TDI, VDI, and CDI from an ``xr.Dataset`` to separate COG files.

    Parameters
    ----------
    ds         : xr.Dataset
        Must contain variables ``'PDI'``, ``'TDI'``, ``'VDI'``, ``'CDI'``.
    output_dir : str | Path
        Directory where COGs will be written.
    prefix     : str
        Filename prefix (default ``'CDI'``).
        Files will be named ``{prefix}_PDI.tif``, ``{prefix}_TDI.tif``, etc.
    crs        : str  – output CRS (default ``'EPSG:4326'``).
    **kwargs   : forwarded to :func:`to_cog`.

    Returns
    -------
    dict  mapping variable name → resolved ``Path``.

    Example
    -------
    >>> paths = cdi_stack_to_cog(ds, "outputs/", prefix="Borena_2000_2020")
    >>> paths["CDI"]
    PosixPath('outputs/Borena_2000_2020_CDI.tif')
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    variables = [v for v in ("PDI", "TDI", "VDI", "CDI") if v in ds]
    if not variables:
        raise ValueError("Dataset contains none of PDI, TDI, VDI, CDI.")

    paths = {}
    for var in variables:
        out_path = output_dir / f"{prefix}_{var}.tif"
        paths[var] = to_cog(ds[var], out_path, crs=crs, **kwargs)
    return paths


def series_to_cog(
    df: pd.DataFrame,
    spatial_ds: "xr.Dataset",
    output_dir: Union[str, Path],
    prefix: str = "CDI",
    crs: str = "EPSG:4326",
    **kwargs,
) -> dict[str, Path]:
    """
    Broadcast area-averaged index ``pd.Series`` back onto a spatial template
    (uniform value per time step) and write COGs.

    Useful when CDI was computed on spatially-averaged inputs but you still
    want georeferenced raster output for leafmap visualisation.

    Parameters
    ----------
    df         : pd.DataFrame  – output of ``compute_all()``
    spatial_ds : xr.Dataset    – spatial template with lat/lon coords
    output_dir : str | Path
    prefix     : str

    Returns
    -------
    dict mapping variable name → Path.
    """
    import xarray as xr
    _require_xarray()

    lat_dim = next((d for d in spatial_ds.dims if d in ("latitude", "lat", "y")), None)
    lon_dim = next((d for d in spatial_ds.dims if d in ("longitude", "lon", "x")), None)
    if lat_dim is None or lon_dim is None:
        raise ValueError("spatial_ds must have recognisable lat/lon dimensions.")

    lats = spatial_ds[lat_dim].values
    lons = spatial_ds[lon_dim].values

    variables = [c for c in df.columns if c in ("PDI", "TDI", "VDI", "CDI")]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    for var in variables:
        series = df[var].dropna()
        # Broadcast scalar → (T, Y, X)
        data  = np.stack(
            [np.full((len(lats), len(lons)), val, dtype=np.float32) for val in series.values],
            axis=0,
        )
        da = xr.DataArray(
            data,
            dims=["time", lat_dim, lon_dim],
            coords={"time": series.index.values, lat_dim: lats, lon_dim: lons},
            name=var,
        )
        out_path = output_dir / f"{prefix}_{var}.tif"
        paths[var] = to_cog(da, out_path, crs=crs, lat_dim=lat_dim, lon_dim=lon_dim, **kwargs)

    return paths


def read_cog(path: Union[str, Path]) -> "xr.DataArray":
    """
    Read a COG written by :func:`to_cog` back into an ``xr.DataArray``.

    The band descriptions (ISO date strings) are parsed to reconstruct the
    ``time`` coordinate.

    Parameters
    ----------
    path : str | Path

    Returns
    -------
    xr.DataArray with dims ``(time, latitude, longitude)``.
    """
    _require_rasterio()
    _require_xarray()
    import xarray as xr

    path = Path(path)
    with rasterio.open(path) as src:
        data  = src.read().astype(np.float32)          # (bands, H, W)
        nodata = src.nodata
        if nodata is not None and not np.isnan(nodata):
            data[data == nodata] = np.nan
        transform = src.transform
        crs_str   = src.crs.to_string() if src.crs else "EPSG:4326"
        descs     = [src.descriptions[i] or f"band_{i+1}" for i in range(src.count)]
        tags_list = [src.tags(i + 1) for i in range(src.count)]

    # Try to recover dates from band descriptions
    dates = []
    for i, desc in enumerate(descs):
        d = desc or tags_list[i].get("date", "")
        try:
            dates.append(pd.Timestamp(d + "-01" if len(d) == 7 else d))
        except Exception:
            dates.append(pd.Timestamp(f"1900-01-{i+1:02d}"))

    height, width = data.shape[1], data.shape[2]
    left, top = transform * (0, 0)
    right, bottom = transform * (width, height)
    lats = np.linspace(top, bottom, height)
    lons = np.linspace(left, right, width)

    da = xr.DataArray(
        data,
        dims=["time", "latitude", "longitude"],
        coords={"time": dates, "latitude": lats, "longitude": lons},
        name=path.stem,
    )
    da.attrs["crs"] = crs_str
    da.attrs["source_file"] = str(path)
    return da


def list_cog_dates(path: Union[str, Path]) -> list[str]:
    """
    Return the list of date strings stored in a COG's band descriptions.

    Parameters
    ----------
    path : str | Path

    Returns
    -------
    list of str  (``'YYYY-MM'`` strings, one per band/timestep)
    """
    _require_rasterio()
    with rasterio.open(path) as src:
        return [src.descriptions[i] or f"band_{i+1}" for i in range(src.count)]