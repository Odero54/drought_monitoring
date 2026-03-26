"""
drought_cdi/tests/test_core.py
Tests for core CDI logic, IO helpers, and plot utilities.
Run with:  pytest drought_cdi/tests/ -v
"""
from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
import pytest
# import tempfile
# from pathlib import Path

from drought_monitoring.core import (
    compute_pdi, compute_tdi, compute_vdi, compute_cdi, compute_all,
    _rl_ip, _ip_mean, _monthly_ltm, DEFAULT_WEIGHTS,
)


def monthly(values, start="2010-01"):
    idx = pd.date_range(start, periods=len(values), freq="MS")
    return pd.Series(values, index=idx)

@pytest.fixture
def synth():
    rng = np.random.default_rng(42)
    n   = 240   # 20 years
    idx = pd.date_range("2000-01", periods=n, freq="MS")
    return (
        pd.Series(rng.uniform(5, 200, n),     index=idx, name="precip"),
        pd.Series(rng.uniform(15, 40, n),     index=idx, name="temp"),
        pd.Series(rng.uniform(0.05, 0.85, n), index=idx, name="ndvi"),
    )

def test_default_weights():
    """Paper specifies PDI=50%, TDI=25%, VDI=25%."""
    assert DEFAULT_WEIGHTS == (0.50, 0.25, 0.25)
    assert abs(sum(DEFAULT_WEIGHTS) - 1.0) < 1e-9

def test_rl_ip_deficit_all_below():
    """If all months in the window are below LTM, RL_IP = window."""
    idx = pd.date_range("2010-01", periods=12, freq="MS")
    # Make raw values always 1, LTM always 10 → condition always True
    raw = pd.Series(np.ones(12), index=idx)
    ltm = pd.Series(np.full(12, 10.0), index=idx)
    rl  = _rl_ip(raw, ltm, window=3, deficit=True)
    # After warmup: every window of 3 has 3 below-LTM months
    assert (rl.dropna() == 3).all()


def test_rl_ip_deficit_none_below():
    """If no month in the window is below LTM, RL_IP = 0."""
    idx = pd.date_range("2010-01", periods=12, freq="MS")
    raw = pd.Series(np.full(12, 10.0), index=idx)
    ltm = pd.Series(np.ones(12), index=idx)   # LTM much lower → never deficit
    rl  = _rl_ip(raw, ltm, window=3, deficit=True)
    assert (rl.dropna() == 0).all()


def test_rl_ip_excess_all_above():
    """If all months are above LTM, RL_IP (excess) = window."""
    idx = pd.date_range("2010-01", periods=12, freq="MS")
    raw = pd.Series(np.full(12, 10.0), index=idx)
    ltm = pd.Series(np.ones(12), index=idx)   # raw > ltm always
    rl  = _rl_ip(raw, ltm, window=3, deficit=False)
    assert (rl.dropna() == 3).all()


def test_rl_ip_partial():
    """Mixed signal: RL_IP counts only months satisfying condition."""
    # Pattern: below, above, below → deficit count in window of 3 = 2
    idx = pd.date_range("2010-01", periods=6, freq="MS")
    raw = pd.Series([1, 10, 1, 10, 1, 10], index=idx)
    ltm = pd.Series([5,  5, 5,  5, 5,  5], index=idx)
    rl  = _rl_ip(raw, ltm, window=3, deficit=True)
    # Window ending at month 3 (idx 2): [1,10,1] → 2 below-5
    assert rl.iloc[2] == 2.0
    # Window ending at month 4 (idx 3): [10,1,10] → 1 below-5
    assert rl.iloc[3] == 1.0


def test_rl_ip_nan_warmup():
    """First (window-1) values are NaN — no partial windows."""
    idx = pd.date_range("2010-01", periods=6, freq="MS")
    raw = pd.Series([1, 1, 1, 1, 1, 1], index=idx)
    ltm = pd.Series([5, 5, 5, 5, 5, 5], index=idx)
    rl  = _rl_ip(raw, ltm, window=3, deficit=True)
    assert np.isnan(rl.iloc[0])
    assert np.isnan(rl.iloc[1])
    assert not np.isnan(rl.iloc[2])


# ── Formula: product not sum ─────────────────────────────────────────────────

def test_formula_is_product(synth):
    """DI = (mu_IP / mu_LTM) * sqrt(RL_IP / RL_LTM) — verify manually."""
    p, _, _ = synth
    window  = 3

    ltm_raw = _monthly_ltm(p)
    mu_ip   = _ip_mean(p, window)
    from drought_monitoring.core import _ip_ltm, _rl_ltm
    mu_ltm  = _ip_ltm(mu_ip)
    rl_ip   = _rl_ip(p, ltm_raw, window, deficit=True)
    rl_ltm  = _rl_ltm(rl_ip)

    pdi_manual = (mu_ip / mu_ltm) * np.sqrt(rl_ip / rl_ltm)
    pdi_func   = compute_pdi(p, window)

    diff = (pdi_manual - pdi_func).abs().dropna().max()
    assert diff < 1e-9, f"Formula mismatch: max diff = {diff:.2e}"


def test_pdi_shape_and_name(synth):
    p, _, _ = synth
    pdi = compute_pdi(p)
    assert pdi.name == "PDI"
    assert len(pdi) == len(p)

def test_pdi_nan_warmup(synth):
    p, _, _ = synth
    pdi = compute_pdi(p, window=3)
    assert np.isnan(pdi.iloc[0]) and np.isnan(pdi.iloc[1])
    assert not np.isnan(pdi.iloc[2])

def test_pdi_requires_datetime():
    with pytest.raises(ValueError, match="DatetimeIndex"):
        compute_pdi(pd.Series([1, 2, 3]))

def test_pdi_requires_series():
    with pytest.raises(TypeError):
        compute_pdi([1, 2, 3])  # type: ignore[arg-type]

def test_tdi_name(synth):
    _, t, _ = synth
    assert compute_tdi(t).name == "TDI"

def test_vdi_name(synth):
    _, _, n = synth
    assert compute_vdi(n).name == "VDI"


def test_cdi_default_weights(synth):
    """Default weights must be 0.50 / 0.25 / 0.25."""
    p, t, n = synth
    pdi, tdi, vdi = compute_pdi(p), compute_tdi(t), compute_vdi(n)
    cdi = compute_cdi(pdi, tdi, vdi)
    exp = 0.50 * pdi + 0.25 * tdi + 0.25 * vdi
    pd.testing.assert_series_equal(cdi.dropna(), exp.dropna(),
                                   check_names=False, rtol=1e-9)

def test_cdi_custom_weights(synth):
    p, t, n = synth
    pdi, tdi, vdi = compute_pdi(p), compute_tdi(t), compute_vdi(n)
    w = (0.6, 0.2, 0.2)
    cdi = compute_cdi(pdi, tdi, vdi, weights=w)
    exp = 0.6 * pdi + 0.2 * tdi + 0.2 * vdi
    pd.testing.assert_series_equal(cdi.dropna(), exp.dropna(),
                                   check_names=False, rtol=1e-9)

def test_cdi_bad_weights_raises(synth):
    p, t, n = synth
    with pytest.raises(ValueError, match="sum to 1.0"):
        compute_cdi(compute_pdi(p), compute_tdi(t), compute_vdi(n),
                    weights=(0.5, 0.5, 0.5))
        

def test_all_columns(synth):
    p, t, n = synth
    df = compute_all(p, t, n)
    assert set(df.columns) == {"PDI", "TDI", "VDI", "CDI"}
    assert len(df) == len(p)

def test_all_interior_not_nan(synth):
    p, t, n = synth
    df = compute_all(p, t, n, window=3)
    assert df["CDI"].iloc[24:-24].notna().mean() > 0.95

def test_all_uses_default_weights(synth):
    """compute_all must apply 50/25/25 by default."""
    p, t, n = synth
    df  = compute_all(p, t, n)
    exp = 0.50 * df["PDI"] + 0.25 * df["TDI"] + 0.25 * df["VDI"]
    diff = (df["CDI"] - exp).abs().dropna().max()
    assert diff < 1e-9


# def test_gee_period_too_short():
#     from drought_cdi.gee import _validate_period
#     with pytest.raises(ValueError, match="at least 20"):
#         _validate_period(2010, 2015)

# def test_gee_period_too_long():
#     from drought_monitoring.gee import _validate_period
#     with pytest.raises(ValueError, match="at most 30"):
#         _validate_period(1990, 2025)

# def test_gee_period_valid():
#     from drought_cdi.gee import _validate_period
#     _validate_period(2000, 2020)
#     _validate_period(2000, 2029)


# def test_cog_roundtrip(synth):
#     try:
#         import xarray as xr
#     except ImportError:
#         pytest.skip("rasterio / xarray not installed")

#     from drought_monitoring.io import to_cog, read_cog

#     p, t, n = synth
#     df = compute_all(p, t, n)
#     times = df["CDI"].dropna().index[:20]
#     lats  = np.linspace(7.0, 3.5, 4)
#     lons  = np.linspace(38.0, 42.5, 5)
#     data  = np.random.default_rng(0).standard_normal((20, 4, 5)).astype(np.float32)

#     da = xr.DataArray(
#         data,
#         dims=["time", "latitude", "longitude"],
#         coords={"time": times, "latitude": lats, "longitude": lons},
#         name="CDI",
#     )

#     with tempfile.TemporaryDirectory() as tmpdir:
#         out_path = Path(tmpdir) / "test_CDI.tif"
#         written  = to_cog(da, out_path)
#         assert written.exists()
#         assert written.stat().st_size > 0

#         da_back = read_cog(written)
#         assert da_back.dims == ("time", "latitude", "longitude")
#         assert len(da_back["time"]) == 20


# @pytest.mark.skipif(not __import__("importlib").util.find_spec("matplotlib"),
#                     reason="matplotlib not installed")
# def test_plot_timeseries_smoke(synth):
#     import matplotlib; matplotlib.use("Agg")
#     import matplotlib.pyplot as plt
#     from drought_cdi.plot import plot_timeseries
#     p, t, n = synth
#     fig = plot_timeseries(compute_all(p, t, n), title="Test")
#     assert fig is not None
#     plt.close(fig)

# @pytest.mark.skipif(not __import__("importlib").util.find_spec("matplotlib"),
#                     reason="matplotlib not installed")
# def test_plot_anomaly_bars_smoke(synth):
#     import matplotlib; matplotlib.use("Agg")
#     import matplotlib.pyplot as plt
#     from drought_cdi.plot import plot_anomaly_bars
#     p, t, n = synth
#     fig = plot_anomaly_bars(compute_all(p, t, n))
#     assert fig is not None
#     plt.close(fig)

# @pytest.mark.skipif(not __import__("importlib").util.find_spec("matplotlib"),
#                     reason="matplotlib not installed")
# def test_plot_seasonal_cycle_smoke(synth):
#     import matplotlib; matplotlib.use("Agg")
#     import matplotlib.pyplot as plt
#     from drought_cdi.plot import plot_seasonal_cycle
#     p, t, n = synth
#     fig = plot_seasonal_cycle(compute_all(p, t, n))
#     assert fig is not None
#     plt.close(fig)