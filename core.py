"""
drought_monitoring.core
================
Core computation of the three drought sub-indices and the Composite
Drought Index (CDI), following Burchard-Levine et al. (2024) / pyCDI.
 
Correct formula (from the paper's equation image)
--------------------------------------------------
Each sub-index is a **product**::
 
    DI = (μ_IP / μ_LTM) * sqrt(RL_IP / RL_LTM)
 
where:
 
    μ_IP     mean of the variable over the interest period (IP)
             — a trailing rolling mean of `window` months
    μ_LTM    long-term mean (LTM) for that same calendar month,
             computed from the full multi-year record of μ_IP values
    RL_IP    run length *within the current IP window* — the number of
             months inside the window where the condition holds:
               PDI / VDI  →  months where raw value < monthly LTM  (deficit)
               TDI        →  months where raw value > monthly LTM  (excess)
    RL_LTM   long-term mean of RL_IP for that calendar month
 
The ratio μ_IP / μ_LTM captures the magnitude of the anomaly.
The ratio RL_IP / RL_LTM captures the persistence of the condition.
Taking the square root balances the two factors so neither dominates.
 
For drought conditions (PDI, VDI: μ_IP < μ_LTM)  DI < 1.
For normal conditions                              DI ≈ 1.
Values > 1 indicate above-normal / wet conditions (PDI, VDI)
or cooling (TDI).
 
Default CDI weights (from the paper)
--------------------------------------
    w_PDI = 0.50   (precipitation dominates drought signal)
    w_TDI = 0.25
    w_VDI = 0.25
 
    CDI = 0.50 * PDI + 0.25 * TDI + 0.25 * VDI
"""

from __future__ import annotations
import numpy as np
import pandas as pd


# Correct default weights  (PDI=50 %, TDI=25 %, VDI=25 %)
DEFAULT_WEIGHTS: tuple[float, float, float] = (0.50, 0.25, 0.25)


# Private helpers
def _validate_series(s: pd.Series, name:str) -> None:
    """Basic sanity checks on an input series."""
    if not isinstance(s, pd.Series):
        raise TypeError(f"'{name}' must be a pandas Series, got {type(s).__name__}.")
    if not isinstance(s.index, pd.DatetimeIndex):
        raise ValueError(f"'{name}' must have a DatetimeIndex")
    if s.empty:
        raise ValueError(f"'{name}' is empty.")
    

def _monthly_ltm(series: pd.Series) -> pd.Series:
    """
    Long-term mean (LTM) for the **raw** monthly series, grouped by calendar
    month, aligned back to the original index.
 
    Used to determine whether each raw monthly value is above or below normal
    when counting the run length inside each IP window.
    """
    idx = pd.DatetimeIndex(series.index)
    return series.groupby(idx.month).transform("mean")


def _ip_mean(series: pd.Series, window: int) -> pd.Series:
    """
    μ_IP  — trailing rolling mean over `window` months.
 
    Uses min_periods=window so the first (window-1) values are NaN,
    preventing spurious results at the start of the record.
    """
    return series.rolling(window=window, min_periods=window).mean()


def _ip_ltm(mu_ip: pd.Series) -> pd.Series:
    """
    μ_LTM  — long-term mean of μ_IP for each calendar month.
 
    After the rolling mean is applied, this groups μ_IP by month and
    returns the multi-year average, aligned to the original index.
    """
    idx = pd.DatetimeIndex(mu_ip.index)
    return mu_ip.groupby(idx.month).transform("mean")


def _rl_ip(series: pd.Series, ltm_raw: pd.Series, window: int, *, deficit: bool) -> pd.Series:
    """
    RL_IP  — run length *within* the current IP window.
 
    For each timestep t, look back at the `window` raw monthly values
    ending at t and count how many of them satisfy the condition relative
    to their own monthly LTM:
 
        deficit (PDI / VDI) : raw value < monthly LTM   (below normal)
        excess  (TDI)       : raw value > monthly LTM   (above normal)
 
    This correctly captures the persistence of the adverse condition
    *inside* the interest period, as described in the paper.
 
    Parameters
    ----------
    series   : raw monthly time series (precip / temp / ndvi)
    ltm_raw  : monthly LTM of the raw series (same length, same index)
    window   : IP window size in months
    deficit  : True for PDI/VDI, False for TDI
 
    Returns
    -------
    pd.Series of int counts (0 … window), NaN where window incomplete.
    """
    if deficit:
        condition = (series < ltm_raw).astype(float)
    else:
        condition = (series > ltm_raw).astype(float)
 
    # Replace NaN positions with NaN so rolling sum propagates them correctly
    condition[series.isna() | ltm_raw.isna()] = np.nan
 
    rl = condition.rolling(window=window, min_periods=window).sum()
    return rl


def _rl_ltm(rl_ip: pd.Series) -> pd.Series:
    """
    RL_LTM  — long-term mean of RL_IP for each calendar month.
 
    Zeros are floored at a small epsilon to avoid division by zero when a
    particular calendar month has never seen a drought / excess run.
    """
    idx = pd.DatetimeIndex(rl_ip.index)
    ltm = rl_ip.groupby(idx.month).transform("mean")
    ltm = ltm.clip(lower=1e-6)    # guard against /0
    return ltm
 
 
def _drought_index(
    mu_ip:  pd.Series,
    mu_ltm: pd.Series,
    rl_ip:  pd.Series,
    rl_ltm: pd.Series,
) -> pd.Series:
    """
    DI = (μ_IP / μ_LTM) * sqrt(RL_IP / RL_LTM)
 
    Both μ_LTM and RL_LTM zero-cells are guarded upstream; any remaining
    zeros are converted to NaN here to avoid silent divide-by-zero.
    """
    mu_ratio = mu_ip / mu_ltm.replace(0, np.nan)
    rl_ratio = rl_ip / rl_ltm.replace(0, np.nan)
    return mu_ratio * np.sqrt(rl_ratio)


# Public API - per-index functions
def compute_pdi(precip: pd.Series, window: int = 3) -> pd.Series:
    """
    Precipitation Drought Index (PDI).
 
    Formula::
 
        PDI = (μ_IP / μ_LTM) * sqrt(RL_IP / RL_LTM)
 
    where RL_IP counts months within the IP window where precipitation
    falls **below** its monthly long-term mean (deficit condition).
 
    Parameters
    ----------
    precip : pd.Series
        Monthly precipitation totals (any unit, e.g. mm month⁻¹)
        with a DatetimeIndex.
    window : int
        IP window size in months (default 3).
 
    Returns
    -------
    pd.Series named ``'PDI'``.
        Values < 1  → drought  (below-normal precip + persistent deficit)
        Values ≈ 1  → near-normal
        Values > 1  → wetter than normal
    """
    _validate_series(precip, "precip")
    ltm_raw = _monthly_ltm(precip)
    mu_ip   = _ip_mean(precip, window)
    mu_ltm  = _ip_ltm(mu_ip)
    rl_ip   = _rl_ip(precip, ltm_raw, window, deficit=True)
    rl_ltm  = _rl_ltm(rl_ip)
    pdi     = _drought_index(mu_ip, mu_ltm, rl_ip, rl_ltm)
    pdi.name = "PDI"
    return pdi


def compute_tdi(temp: pd.Series, window: int = 3) -> pd.Series:
    """
    Temperature Drought Index (TDI).
 
    Formula::
 
        TDI = (μ_IP / μ_LTM) * sqrt(RL_IP / RL_LTM)
 
    where RL_IP counts months within the IP window where temperature
    is **above** its monthly long-term mean (excess / warm condition).
 
    Warm anomalies intensify drought by increasing evapotranspiration.
    Therefore TDI > 1 indicates drought-aggravating warmth.
 
    Parameters
    ----------
    temp : pd.Series
        Monthly mean temperature (°C) with a DatetimeIndex.
    window : int
        IP window size in months (default 3).
 
    Returns
    -------
    pd.Series named ``'TDI'``.
        Values > 1  → warmer than normal (drought-aggravating)
        Values ≈ 1  → near-normal temperature
        Values < 1  → cooler than normal
    """
    _validate_series(temp, "temp")
    ltm_raw = _monthly_ltm(temp)
    mu_ip   = _ip_mean(temp, window)
    mu_ltm  = _ip_ltm(mu_ip)
    rl_ip   = _rl_ip(temp, ltm_raw, window, deficit=False)   # excess = above LTM
    rl_ltm  = _rl_ltm(rl_ip)
    tdi     = _drought_index(mu_ip, mu_ltm, rl_ip, rl_ltm)
    tdi.name = "TDI"
    return tdi


def compute_vdi(ndvi: pd.Series, window: int = 3) -> pd.Series:
    """
    Vegetation Drought Index (VDI).
 
    Formula::
 
        VDI = (μ_IP / μ_LTM) * sqrt(RL_IP / RL_LTM)
 
    where RL_IP counts months within the IP window where NDVI
    falls **below** its monthly long-term mean (vegetation deficit).
 
    Parameters
    ----------
    ndvi : pd.Series
        Monthly mean NDVI [−1, 1] with a DatetimeIndex.
        Typically derived from MODIS MOD09GQ 8-day composites composited
        to monthly means.
    window : int
        IP window size in months (default 3).
 
    Returns
    -------
    pd.Series named ``'VDI'``.
        Values < 1  → vegetation stress / drought signal
        Values ≈ 1  → near-normal vegetation
        Values > 1  → greener than normal
    """
    _validate_series(ndvi, "ndvi")
    ltm_raw = _monthly_ltm(ndvi)
    mu_ip   = _ip_mean(ndvi, window)
    mu_ltm  = _ip_ltm(mu_ip)
    rl_ip   = _rl_ip(ndvi, ltm_raw, window, deficit=True)
    rl_ltm  = _rl_ltm(rl_ip)
    vdi     = _drought_index(mu_ip, mu_ltm, rl_ip, rl_ltm)
    vdi.name = "VDI"
    return vdi


def compute_cdi(
    pdi: pd.Series,
    tdi: pd.Series,
    vdi: pd.Series,
    weights: tuple[float, float, float] = DEFAULT_WEIGHTS,
) -> pd.Series:
    """
    Composite Drought Index (CDI).
 
    ::
 
        CDI = w_PDI * PDI + w_TDI * TDI + w_VDI * VDI
 
    Default weights follow the paper: PDI=0.50, TDI=0.25, VDI=0.25.
 
    Parameters
    ----------
    pdi, tdi, vdi : pd.Series
        The three sub-indices on a common DatetimeIndex.
    weights : tuple of three floats (w_PDI, w_TDI, w_VDI)
        Must sum to 1.0.
 
    Returns
    -------
    pd.Series named ``'CDI'``.
        Values < 1  → composite drought signal
        Values ≈ 1  → near-normal
        Values > 1  → above-normal / wet
 
    Raises
    ------
    ValueError
        If weights do not sum to 1.0.
    """
    if not np.isclose(sum(weights), 1.0, atol=1e-6):
        raise ValueError(
            f"weights must sum to 1.0, got {sum(weights):.6f}. "
            f"Received: {weights}"
        )
    df = pd.DataFrame({"PDI": pdi, "TDI": tdi, "VDI": vdi})
    cdi = df["PDI"] * weights[0] + df["TDI"] * weights[1] + df["VDI"] * weights[2]
    cdi.name = "CDI"
    return cdi


# Convenience wrapper
def compute_all(
    precip: pd.Series,
    temp:   pd.Series,
    ndvi:   pd.Series,
    window:  int = 3,
    weights: tuple[float, float, float] = DEFAULT_WEIGHTS,
) -> pd.DataFrame:
    """
    Compute PDI, TDI, VDI, and CDI in a single call.
 
    Parameters
    ----------
    precip, temp, ndvi : pd.Series
        Monthly time series with a shared DatetimeIndex.
    window : int
        IP window size in months (default 3).
    weights : tuple of three floats (w_PDI, w_TDI, w_VDI)
        Must sum to 1.0.  Default: (0.50, 0.25, 0.25).
 
    Returns
    -------
    pd.DataFrame with columns ``['PDI', 'TDI', 'VDI', 'CDI']``.
 
    Example
    -------
    >>> df = compute_all(precip, temp, ndvi)
    >>> df.tail()
    """
    pdi = compute_pdi(precip, window)
    tdi = compute_tdi(temp,   window)
    vdi = compute_vdi(ndvi,   window)
    cdi = compute_cdi(pdi, tdi, vdi, weights)
    return pd.DataFrame({"PDI": pdi, "TDI": tdi, "VDI": vdi, "CDI": cdi})