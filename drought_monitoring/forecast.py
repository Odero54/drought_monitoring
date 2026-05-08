"""
drought_monitoring.forecast
===========================
Statistical 6-month drought forecast.

Pipeline
--------
1. STL decomposition (period=12, robust) of each raw input series.
2. ARIMA(1,0,0) fit to the STL remainder.
3. Point forecast = trend extrapolation + seasonal continuation + ARIMA remainder.
4. Bootstrap (n_bootstrap AR-innovation draws) → CI bands on CDI.

For VDI specifically, NDVI is forecast first and then converted back to VDI
using the full historical LTM, preserving correct calibration.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

from .core import compute_pdi, compute_tdi, compute_vdi


# ── helpers ──────────────────────────────────────────────────────────────────

def _next_dates(last: pd.Timestamp, n: int) -> pd.DatetimeIndex:
    return pd.date_range(last + pd.DateOffset(months=1), periods=n, freq="MS")


def _extend(history: pd.Series, forecast: pd.Series) -> pd.Series:
    """Concatenate observed history + forecast, dropping NaNs from history."""
    return pd.concat([history.dropna(), forecast])


def _stl_arima_forecast(
    series: pd.Series,
    n_months: int,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[pd.Series, np.ndarray, pd.DatetimeIndex]:
    """
    Forecast a monthly series n_months ahead.

    Steps
    -----
    1. STL decomposition → trend, seasonal, remainder.
    2. Linear extrapolation of the trend component.
    3. Seasonal continuation (repeat last 12-month cycle).
    4. ARIMA(1,0,0) on the remainder → remainder forecast.
    5. Bootstrap n_bootstrap AR-innovation paths for CI.

    Returns
    -------
    point    : pd.Series             — point forecast (n_months,)
    boot_mat : np.ndarray            — bootstrap draws (n_bootstrap, n_months)
    fc_dates : pd.DatetimeIndex
    """
    from statsmodels.tsa.seasonal import STL
    from statsmodels.tsa.arima.model import ARIMA

    s = series.dropna()
    n = len(s)

    # STL decomposition
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stl_res = STL(s, period=12, robust=True).fit()

    trend    = stl_res.trend.values
    seasonal = stl_res.seasonal.values
    remainder = pd.Series(stl_res.resid, index=s.index)

    # Trend extrapolation: linear fit on last 24 points
    w = min(24, n)
    slope, intercept = np.polyfit(np.arange(w), trend[-w:], 1)
    trend_fc = intercept + slope * np.arange(w, w + n_months)

    # Seasonal continuation: last full 12-month cycle, starting next month
    seasonal_fc = np.array([seasonal[-(12 - i % 12)] for i in range(n_months)])

    # ARIMA(1,0,0) on remainder
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        arima_fit = ARIMA(remainder, order=(1, 0, 0)).fit()

    ar_fc     = arima_fit.forecast(steps=n_months).values
    phi       = float(arima_fit.arparams[0]) if len(arima_fit.arparams) > 0 else 0.0
    residuals = arima_fit.resid.values

    # Point forecast
    point_fc = trend_fc + seasonal_fc + ar_fc

    # Bootstrap via AR-innovation resampling
    boot_mat   = np.empty((n_bootstrap, n_months))
    last_resid = float(remainder.iloc[-1])
    for b in range(n_bootstrap):
        innov = rng.choice(residuals, size=n_months, replace=True)
        r = last_resid
        for j in range(n_months):
            r = phi * r + innov[j]
            boot_mat[b, j] = trend_fc[j] + seasonal_fc[j] + r

    fc_dates = _next_dates(s.index[-1], n_months)
    return pd.Series(point_fc, index=fc_dates, name=series.name), boot_mat, fc_dates


# ── public API ────────────────────────────────────────────────────────────────

def forecast_all_statistical(
    precip: pd.Series,
    temp:   pd.Series,
    ndvi:   pd.Series,
    n_months:    int   = 6,
    window:      int   = 3,
    weights:     tuple[float, float, float] = (0.50, 0.25, 0.25),
    ci_level:    float = 0.90,
    n_bootstrap: int   = 300,
    random_seed: int   = 42,
) -> pd.DataFrame:
    """
    6-month CDI forecast using STL decomposition + ARIMA(1,0,0) on each raw series.

    Pipeline
    --------
    For each raw series (precip, temp, ndvi):

    1. STL-decompose → trend + seasonal + remainder.
    2. Fit ARIMA(1,0,0) to the remainder.
    3. Reconstruct: trend extrap + seasonal continuation + ARIMA remainder.
    4. Append forecast to history and compute sub-index on the extended series
       so the historical LTM is preserved (only 6 values dilute 300+).
    5. Bootstrap 300 AR-innovation draws → CI bands on CDI.

    Parameters
    ----------
    precip, temp, ndvi : pd.Series
        Monthly observed series with a DatetimeIndex.
    n_months    : int    — forecast horizon in months (default 6).
    window      : int    — IP window passed to sub-index functions (default 3).
    weights     : tuple  — (w_PDI, w_TDI, w_VDI), must sum to 1.0.
    ci_level    : float  — confidence level, e.g. 0.90 → 5th–95th pct (default 0.90).
    n_bootstrap : int    — bootstrap draws (default 300).
    random_seed : int    — for reproducibility (default 42).

    Returns
    -------
    pd.DataFrame  with DatetimeIndex covering the next n_months, columns:
        PDI, TDI, VDI, CDI   — point forecasts
        CDI_lower, CDI_upper — confidence interval bounds
        lead                 — lead time (1 … n_months)

    Example
    -------
    >>> fc = forecast_all_statistical(precip, temp, ndvi, n_months=6)
    >>> print(fc[["CDI", "CDI_lower", "CDI_upper"]].round(3))
    """
    if not np.isclose(sum(weights), 1.0, atol=1e-6):
        raise ValueError(f"weights must sum to 1.0, got {sum(weights):.6f}.")

    rng = np.random.default_rng(random_seed)

    # Forecast each raw series
    precip_fc, precip_boot, fc_dates = _stl_arima_forecast(precip, n_months, n_bootstrap, rng)
    temp_fc,   temp_boot,   _        = _stl_arima_forecast(temp,   n_months, n_bootstrap, rng)
    ndvi_fc,   ndvi_boot,   _        = _stl_arima_forecast(ndvi,   n_months, n_bootstrap, rng)

    # Point-forecast sub-indices on the extended series
    pdi_fc = compute_pdi(_extend(precip, precip_fc), window).iloc[-n_months:].values
    tdi_fc = compute_tdi(_extend(temp,   temp_fc),   window).iloc[-n_months:].values
    vdi_fc = compute_vdi(_extend(ndvi,   ndvi_fc),   window).iloc[-n_months:].values
    cdi_fc = weights[0] * pdi_fc + weights[1] * tdi_fc + weights[2] * vdi_fc

    # Bootstrap CI
    alpha     = (1.0 - ci_level) / 2.0
    cdi_boot  = np.empty((n_bootstrap, n_months))
    for b in range(n_bootstrap):
        pb = pd.Series(precip_boot[b], index=fc_dates)
        tb = pd.Series(temp_boot[b],   index=fc_dates)
        nb = pd.Series(ndvi_boot[b],   index=fc_dates)
        pdi_b = compute_pdi(_extend(precip, pb), window).iloc[-n_months:].values
        tdi_b = compute_tdi(_extend(temp,   tb), window).iloc[-n_months:].values
        vdi_b = compute_vdi(_extend(ndvi,   nb), window).iloc[-n_months:].values
        cdi_boot[b] = weights[0]*pdi_b + weights[1]*tdi_b + weights[2]*vdi_b

    cdi_lower = np.nanquantile(cdi_boot, alpha,       axis=0)
    cdi_upper = np.nanquantile(cdi_boot, 1.0 - alpha, axis=0)

    return pd.DataFrame({
        "PDI":       pdi_fc,
        "TDI":       tdi_fc,
        "VDI":       vdi_fc,
        "CDI":       cdi_fc,
        "CDI_lower": cdi_lower,
        "CDI_upper": cdi_upper,
        "lead":      np.arange(1, n_months + 1),
    }, index=fc_dates)


def forecast_vdi_from_ndvi(
    ndvi:        pd.Series,
    n_months:    int   = 6,
    window:      int   = 3,
    ci_level:    float = 0.90,
    n_bootstrap: int   = 300,
    random_seed: int   = 42,
) -> pd.DataFrame:
    """
    Forecast VDI directly from a raw NDVI series.

    Pipeline: NDVI → STL+ARIMA forecast → extended series → VDI
    with the historical LTM intact (only n_months new values are appended).

    Parameters
    ----------
    ndvi        : pd.Series  monthly NDVI [−1, 1] with DatetimeIndex.
    n_months    : int        forecast horizon (default 6).
    window      : int        IP window (default 3).
    ci_level    : float      confidence level (default 0.90).
    n_bootstrap : int        bootstrap draws (default 300).
    random_seed : int        for reproducibility (default 42).

    Returns
    -------
    pd.DataFrame with columns: VDI, VDI_lower, VDI_upper, lead.

    Example
    -------
    >>> vdi_fc = forecast_vdi_from_ndvi(ndvi, n_months=6)
    >>> print(vdi_fc.round(3))
    """
    rng = np.random.default_rng(random_seed)
    ndvi_fc, ndvi_boot, fc_dates = _stl_arima_forecast(ndvi, n_months, n_bootstrap, rng)

    vdi_fc = compute_vdi(_extend(ndvi, ndvi_fc), window).iloc[-n_months:].values

    alpha    = (1.0 - ci_level) / 2.0
    vdi_boot = np.empty((n_bootstrap, n_months))
    for b in range(n_bootstrap):
        nb = pd.Series(ndvi_boot[b], index=fc_dates)
        vdi_boot[b] = compute_vdi(_extend(ndvi, nb), window).iloc[-n_months:].values

    return pd.DataFrame({
        "VDI":       vdi_fc,
        "VDI_lower": np.nanquantile(vdi_boot, alpha,       axis=0),
        "VDI_upper": np.nanquantile(vdi_boot, 1.0 - alpha, axis=0),
        "lead":      np.arange(1, n_months + 1),
    }, index=fc_dates)
