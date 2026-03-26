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


# Internal pixel-wise dispatcher
def _apply_pixelwise(
    da: xr.DataArray,
    index_fn,          # callable(pd.Series, int) -> pd.Series
    window: int,
) -> xr.DataArray:
    """
    Apply a 1-D index function to every pixel in a (time, y, x) DataArray.

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

    time_idx = pd.DatetimeIndex(da["time"].values)
    data     = da.values          # numpy array

    # 1-D input (no spatial dims
    if data.ndim == 1:
        s      = pd.Series(data, index=time_idx)
        result = index_fn(s, window).values
        return xr.DataArray(result, coords=da.coords, dims=da.dims)

    # 3-D input (time, y, x) 
    if data.ndim != 3:
        raise ValueError(
            f"Expected a 1-D or 3-D DataArray, got {data.ndim}-D. "
            "Reshape or select a 2-D spatial slice first."
        )

    T, Y, X = data.shape
    output  = np.full((T, Y, X), np.nan, dtype=np.float64)

    for yi in range(Y):
        for xi in range(X):
            pixel = data[:, yi, xi]
            if np.all(np.isnan(pixel)):          # mask / ocean pixels
                continue
            s             = pd.Series(pixel, index=time_idx)
            output[:, yi, xi] = index_fn(s, window).values

    return xr.DataArray(output, coords=da.coords, dims=da.dims)


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