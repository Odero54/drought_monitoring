"""
drought_monitoring.spatial
===================
Pixel-wise CDI computation on ``xr.DataArray`` / ``xr.Dataset`` raster cubes.

All spatial functions are thin wrappers around the core 1-D functions in
:mod:`drought_monitoring.core`.  They iterate over spatial pixels, apply the
pandas-based index logic, and reassemble the result into a DataArray that
preserves the original coordinates and CRS (if present via rioxarray).

Typical input shape: ``(time, y, x)`` or ``(time, latitude, longitude)``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from .core import compute_pdi, compute_tdi, compute_vdi


# ── pixel-level helper (module-level so dask can pickle it) ──────────────────

def _pixel_wrapper(
    arr_1d:   np.ndarray,
    time_idx: pd.DatetimeIndex,
    index_fn,
    window:   int,
) -> np.ndarray:
    """Apply ``index_fn`` to a single pixel's time series (1-D numpy array)."""
    if np.all(np.isnan(arr_1d)):
        return np.full(len(arr_1d), np.nan, dtype=np.float64)
    s = pd.Series(arr_1d.astype(np.float64), index=time_idx)
    return index_fn(s, window).values.astype(np.float64)


# Internal pixel-wise dispatcher
def _apply_pixelwise(
    da: xr.DataArray,
    index_fn,          # callable(pd.Series, int) -> pd.Series
    window: int,
) -> xr.DataArray:
    """
    Apply a 1-D index function to every pixel in a (time, y, x) DataArray.

    Uses ``xr.apply_ufunc`` with ``vectorize=True`` and
    ``dask="parallelized"`` so that, when ``da`` is backed by a dask array,
    spatial chunks are computed in parallel.  No data is pulled from GEE /
    disk until ``.compute()`` is called.

    Parameters
    ----------
    da       : xr.DataArray  with dim ``time`` (required) plus spatial dims.
    index_fn : one of ``compute_pdi``, ``compute_tdi``, ``compute_vdi``.
    window   : rolling-window size forwarded to ``index_fn``.

    Returns
    -------
    xr.DataArray with the same shape / coords as ``da``.
    """
    if "time" not in da.dims:
        raise ValueError("DataArray must have a 'time' dimension.")

    time_idx     = pd.DatetimeIndex(da["time"].values)
    original_dims = da.dims

    # apply_ufunc moves the core dim (time) to the last axis inside the
    # vectorised function call, then appends it to the output axes.
    # We transpose back to restore the original dim order.
    result = xr.apply_ufunc(
        _pixel_wrapper,
        da,
        input_core_dims=[["time"]],
        output_core_dims=[["time"]],
        kwargs={"time_idx": time_idx, "index_fn": index_fn, "window": window},
        vectorize=True,           # loop over all non-core dims (pixels)
        dask="parallelized",      # keep as lazy dask array; parallelize over chunks
        output_dtypes=[np.float64],
    )
    return result.transpose(*original_dims)


# Public spatial-index functions
def spatial_pdi(precip: xr.DataArray, window: int = 3) -> xr.DataArray:
    """
    PDI computed pixel-wise on a ``(time, y, x)`` DataArray.

    Parameters
    ----------
    precip : xr.DataArray  – monthly precipitation cube.
    window : int           – interest-period window (default 3).

    Returns
    -------
    xr.DataArray named ``'PDI'``.
    """
    out = _apply_pixelwise(precip, compute_pdi, window)
    out.name = "PDI"
    return out


def spatial_tdi(temp: xr.DataArray, window: int = 3) -> xr.DataArray:
    """
    TDI computed pixel-wise on a ``(time, y, x)`` DataArray.

    Parameters
    ----------
    temp   : xr.DataArray  – monthly mean temperature cube.
    window : int           – interest-period window (default 3).

    Returns
    -------
    xr.DataArray named ``'TDI'``.
    """
    out = _apply_pixelwise(temp, compute_tdi, window)
    out.name = "TDI"
    return out


def spatial_vdi(ndvi: xr.DataArray, window: int = 3) -> xr.DataArray:
    """
    VDI computed pixel-wise on a ``(time, y, x)`` DataArray.

    Parameters
    ----------
    ndvi   : xr.DataArray  – monthly mean NDVI cube.
    window : int           – interest-period window (default 3).

    Returns
    -------
    xr.DataArray named ``'VDI'``.
    """
    out = _apply_pixelwise(ndvi, compute_vdi, window)
    out.name = "VDI"
    return out


def spatial_cdi(
    precip: xr.DataArray,
    temp:   xr.DataArray,
    ndvi:   xr.DataArray,
    window:  int = 3,
    weights: tuple[float, float, float] = (0.50, 0.25, 0.25),
) -> xr.Dataset:
    """
    Compute all four indices pixel-wise and return an ``xr.Dataset``.

    Parameters
    ----------
    precip, temp, ndvi : xr.DataArray
        Raster cubes with dims ``(time, y, x)``.
    window  : int
        Interest-period window in months.
    weights : tuple of three floats
        CDI component weights (PDI, TDI, VDI); must sum to 1.0.

    Returns
    -------
    xr.Dataset with variables ``PDI``, ``TDI``, ``VDI``, ``CDI``.

    Example
    -------
    >>> ds = spatial_cdi(era5_precip, era5_temp, modis_ndvi)
    >>> ds["CDI"].isel(time=-1).plot()
    """
    import numpy as np
    if not np.isclose(sum(weights), 1.0, atol=1e-6):
        raise ValueError(f"weights must sum to 1.0, got {sum(weights):.6f}.")

    pdi = spatial_pdi(precip, window)
    tdi = spatial_tdi(temp,   window)
    vdi = spatial_vdi(ndvi,   window)
    cdi = weights[0] * pdi + weights[1] * tdi + weights[2] * vdi
    cdi.name = "CDI"

    return xr.Dataset({"PDI": pdi, "TDI": tdi, "VDI": vdi, "CDI": cdi})


def yearly_drought_maps(
    precip: xr.DataArray,
    temp:   xr.DataArray,
    ndvi:   xr.DataArray,
    window:  int = 3,
    weights: tuple[float, float, float] = (0.50, 0.25, 0.25),
) -> xr.Dataset:
    """
    Compute annual mean drought maps from monthly raster cubes.

    The function computes monthly PDI / TDI / VDI / CDI pixel-wise
    (leveraging dask parallelism when the inputs are dask-backed), then
    resamples to **yearly means** and triggers ``.compute()`` so the result
    is a concrete in-memory ``xr.Dataset``.

    Parameters
    ----------
    precip, temp, ndvi : xr.DataArray
        Monthly raster cubes with dims ``(time, latitude, longitude)``.
        May be lazy dask arrays (e.g. from :func:`drought_monitoring.gee.open_era5_cube`).
    window  : int
        Interest-period window in months (default 3).
    weights : tuple of three floats
        CDI component weights ``(w_PDI, w_TDI, w_VDI)``; must sum to 1.0.

    Returns
    -------
    xr.Dataset
        Variables ``PDI``, ``TDI``, ``VDI``, ``CDI`` on a
        ``(year, latitude, longitude)`` grid.  All data is in memory.

    Example
    -------
    >>> from drought_monitoring.gee import authenticate, open_era5_cube, open_modis_ndvi
    >>> from drought_monitoring.spatial import yearly_drought_maps
    >>> authenticate(project="my-project")
    >>> aoi = [38.0, 3.5, 42.5, 7.0]
    >>> era5 = open_era5_cube(aoi, 2000, 2020)
    >>> ndvi = open_modis_ndvi(aoi, 2000, 2020).interp_like(era5["precip_mm"])
    >>> ds   = yearly_drought_maps(era5["precip_mm"], era5["temp_c"], ndvi)
    >>> ds["CDI"].sel(time="2015").plot()
    """
    # Rechunk so the full time axis is in one chunk per spatial tile.
    # This is required because the rolling statistics in the core functions
    # must see the complete time series for every pixel.
    spatial_chunk = 20   # pixels per spatial chunk (tune to available RAM)
    spatial_dims  = [d for d in precip.dims if d != "time"]
    chunk_spec    = {"time": -1, **{d: spatial_chunk for d in spatial_dims}}
    precip = precip.chunk(chunk_spec)
    temp   = temp.chunk(chunk_spec)
    ndvi   = ndvi.chunk(chunk_spec)

    # Monthly CDI (lazy dask graph)
    monthly_ds = spatial_cdi(precip, temp, ndvi, window=window, weights=weights)

    # Resample to yearly means and materialise
    return monthly_ds.resample(time="YE").mean().compute()