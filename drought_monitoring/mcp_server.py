"""
MCP server for drought-monitoring.

Exposes the full CDI pipeline as tools that Claude (or any MCP-compatible
model) can call:  authenticate GEE → fetch ERA5/MODIS → compute CDI →
classify severity → forecast → spatial maps → COG export.

Run:
    drought-mcp          # stdio transport (Claude Desktop / claude-code)
    python -m drought_monitoring.mcp_server
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "drought-monitoring",
    instructions=(
        "Tools for computing and forecasting the Composite Drought Index (CDI) "
        "from Google Earth Engine ERA5-Land and MODIS data. "
        "Always call authenticate_gee first, then fetch data implicitly via the "
        "other tools by providing an AOI as [lon_min, lat_min, lon_max, lat_max]."
    ),
)

# ---------------------------------------------------------------------------
# In-memory cache: avoid re-fetching the same AOI/year range twice per session
# ---------------------------------------------------------------------------
_DATA_CACHE: dict[tuple, tuple] = {}  # key → (precip, temp, ndvi)
_MAP_CACHE: dict[tuple, object] = {}  # key → xr.Dataset of annual maps


def _cache_key(aoi: list[float], start_year: int, end_year: int) -> tuple:
    return (tuple(aoi), start_year, end_year)


def _get_or_fetch(aoi: list[float], start_year: int, end_year: int):
    """Return cached (precip, temp, ndvi) or fetch from GEE and cache."""
    key = _cache_key(aoi, start_year, end_year)
    if key not in _DATA_CACHE:
        from drought_monitoring.gee import (
            fetch_era5_precip,
            fetch_era5_temp,
            fetch_modis_ndvi,
        )
        precip = fetch_era5_precip(aoi, start_year=start_year, end_year=end_year)
        temp   = fetch_era5_temp(aoi,   start_year=start_year, end_year=end_year)
        ndvi   = fetch_modis_ndvi(aoi,  start_year=start_year, end_year=end_year)
        _DATA_CACHE[key] = (precip, temp, ndvi)
    return _DATA_CACHE[key]


def _get_or_build_maps(aoi: list[float], start_year: int, end_year: int):
    """Return cached annual CDI xr.Dataset or compute it and cache."""
    key = _cache_key(aoi, start_year, end_year)
    if key not in _MAP_CACHE:
        from drought_monitoring.gee import yearly_drought_maps
        _MAP_CACHE[key] = yearly_drought_maps(
            aoi, start_year=start_year, end_year=end_year
        )
    return _MAP_CACHE[key]


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def authenticate_gee(project: str) -> str:
    """
    Authenticate with Google Earth Engine.

    Must be called once per session before any data-fetching tool.
    Credentials are stored locally by earthengine-api after the first
    browser-based login, so subsequent calls are silent.

    Args:
        project: Your GEE cloud project ID (e.g. 'my-gee-project').
    """
    from drought_monitoring.gee import authenticate
    authenticate(project=project)
    return f"GEE authenticated successfully with project '{project}'."


@mcp.tool()
def compute_drought_indices(
    aoi: list[float],
    start_year: int,
    end_year: int,
    window: int = 3,
    weights: list[float] | None = None,
) -> str:
    """
    Fetch ERA5-Land + MODIS data for an area and compute PDI, TDI, VDI, and CDI.

    Returns summary statistics, the most recent 12 months of index values,
    and the overall severity distribution.

    Args:
        aoi: Bounding box [lon_min, lat_min, lon_max, lat_max].
        start_year: First year of the monitoring period (minimum 20-year span required).
        end_year: Last year of the monitoring period.
        window: Trailing rolling window in months used for the interest period (default 3).
        weights: CDI component weights [pdi, tdi, vdi] summing to 1.0
                 (default [0.50, 0.25, 0.25]).
    """
    from drought_monitoring import compute_all
    from drought_monitoring.plot import classify_cdi

    w = tuple(weights) if weights else (0.50, 0.25, 0.25)
    precip, temp, ndvi = _get_or_fetch(aoi, start_year, end_year)
    df = compute_all(precip, temp, ndvi, window=window, weights=w)
    df["severity"] = classify_cdi(df["CDI"])
    valid = df.dropna(subset=["CDI"])

    lines = [
        f"CDI computed: {len(valid)} months  "
        f"({valid.index[0].date()} → {valid.index[-1].date()})",
        f"Weights — PDI: {w[0]}  TDI: {w[1]}  VDI: {w[2]}",
        "",
        "Summary statistics:",
        valid[["PDI", "TDI", "VDI", "CDI"]].describe().round(3).to_string(),
        "",
        "Most recent 12 months:",
        valid[["PDI", "TDI", "VDI", "CDI", "severity"]].tail(12).round(3).to_string(),
        "",
        "Severity distribution (% of months):",
        valid["severity"]
        .value_counts(normalize=True)
        .mul(100)
        .round(1)
        .to_string(),
    ]
    return "\n".join(lines)


@mcp.tool()
def get_worst_drought_periods(
    aoi: list[float],
    start_year: int,
    end_year: int,
    n: int = 10,
) -> str:
    """
    Return the n worst drought months and the worst calendar year for an area.

    Args:
        aoi: Bounding box [lon_min, lat_min, lon_max, lat_max].
        start_year: First year of the monitoring period.
        end_year: Last year of the monitoring period.
        n: Number of worst months to return (default 10).
    """
    from drought_monitoring import compute_all
    from drought_monitoring.plot import classify_cdi

    precip, temp, ndvi = _get_or_fetch(aoi, start_year, end_year)
    df = compute_all(precip, temp, ndvi)
    df["severity"] = classify_cdi(df["CDI"])
    valid = df.dropna(subset=["CDI"])

    worst_months = (
        valid[["CDI", "PDI", "TDI", "VDI", "severity"]]
        .nsmallest(n, "CDI")
        .round(3)
    )
    annual = valid["CDI"].resample("YE").mean()
    worst_year = annual.idxmin()

    lines = [
        f"Worst {n} drought months (ranked by CDI ascending):",
        worst_months.to_string(),
        "",
        f"Worst drought year: {worst_year.year}  "
        f"(annual mean CDI = {annual.min():.3f})",
        "",
        "Annual mean CDI (all years):",
        annual.round(3).to_string(),
    ]
    return "\n".join(lines)


@mcp.tool()
def forecast_drought(
    aoi: list[float],
    start_year: int,
    end_year: int,
    n_months: int = 6,
    ci_level: float = 0.90,
) -> str:
    """
    Generate a statistical drought forecast using STL decomposition + SARIMA.

    The model decomposes each index into trend, seasonal, and remainder components,
    fits SARIMA(1,0,0)(1,0,0)[12] on the remainder, and uses 300-draw bootstrap
    resampling to build confidence bands.

    Args:
        aoi: Bounding box [lon_min, lat_min, lon_max, lat_max].
        start_year: First year of the historical training period.
        end_year: Last year of the historical period (forecast begins immediately after).
        n_months: Number of months ahead to forecast (default 6).
        ci_level: Confidence interval level, e.g. 0.90 for 90% bands (default 0.90).
    """
    from drought_monitoring.forecast import forecast_all_statistical
    from drought_monitoring.plot import classify_cdi

    precip, temp, ndvi = _get_or_fetch(aoi, start_year, end_year)
    fc = forecast_all_statistical(
        precip, temp, ndvi,
        n_months=n_months,
        ci_level=ci_level,
    )
    fc["severity"] = classify_cdi(fc["CDI"])

    lines = [
        f"{n_months}-month drought forecast  "
        f"({ci_level * 100:.0f}% confidence band, STL + SARIMA):",
        "",
        fc[["CDI", "CDI_lower", "CDI_upper", "PDI", "TDI", "VDI", "severity", "lead"]]
        .round(3)
        .to_string(),
    ]
    return "\n".join(lines)


@mcp.tool()
def get_annual_spatial_summary(
    aoi: list[float],
    start_year: int,
    end_year: int,
) -> str:
    """
    Compute annual spatial CDI maps (dask-parallelised, fully in-memory) and
    return area-averaged annual CDI values for each year.

    This is the same pipeline as yearly_drought_maps() — streams ERA5 + MODIS via
    xee, computes CDI pixel-wise with dask, and resamples to annual means.

    Args:
        aoi: Bounding box [lon_min, lat_min, lon_max, lat_max].
        start_year: First year.
        end_year: Last year.
    """
    ds = _get_or_build_maps(aoi, start_year, end_year)

    spatial_dims = [d for d in ds["CDI"].dims if d != "time"]
    annual_cdi   = ds["CDI"].mean(dim=spatial_dims)
    annual_pdi   = ds["PDI"].mean(dim=spatial_dims)
    annual_tdi   = ds["TDI"].mean(dim=spatial_dims)
    annual_vdi   = ds["VDI"].mean(dim=spatial_dims)

    import pandas as pd
    summary = pd.DataFrame({
        "CDI": annual_cdi.values,
        "PDI": annual_pdi.values,
        "TDI": annual_tdi.values,
        "VDI": annual_vdi.values,
    }, index=annual_cdi["time"].values)
    summary.index = pd.to_datetime(summary.index).year
    summary.index.name = "year"

    lines = [
        f"Spatial dataset dims: {dict(ds.sizes)}",
        f"Variables: {list(ds.data_vars)}",
        "",
        "Area-averaged annual indices:",
        summary.round(3).to_string(),
        "",
        f"Driest year (lowest CDI): {summary['CDI'].idxmin()}  "
        f"(CDI = {summary['CDI'].min():.3f})",
        f"Wettest year (highest CDI): {summary['CDI'].idxmax()}  "
        f"(CDI = {summary['CDI'].max():.3f})",
    ]
    return "\n".join(lines)


@mcp.tool()
def export_cog(
    aoi: list[float],
    start_year: int,
    end_year: int,
    output_dir: str = "outputs/",
    prefix: str = "drought",
) -> str:
    """
    Export annual CDI maps to Cloud Optimized GeoTIFFs — one file per variable
    (PDI, TDI, VDI, CDI), one band per year, with ISO-date band descriptions.

    Args:
        aoi: Bounding box [lon_min, lat_min, lon_max, lat_max].
        start_year: First year.
        end_year: Last year.
        output_dir: Directory to write files into (created if absent).
        prefix: Filename prefix, e.g. 'Turkana_2000_2025'.
    """
    import os
    from drought_monitoring.io import cdi_stack_to_cog

    ds = _get_or_build_maps(aoi, start_year, end_year)
    os.makedirs(output_dir, exist_ok=True)
    paths = cdi_stack_to_cog(ds, output_dir=output_dir, prefix=prefix)

    lines = ["COG files written:"]
    for var, path in paths.items():
        size_mb = os.path.getsize(path) / 1_048_576
        lines.append(f"  {var}: {path}  ({size_mb:.2f} MB)")
    return "\n".join(lines)


@mcp.tool()
def forecast_vdi_from_ndvi(
    aoi: list[float],
    start_year: int,
    end_year: int,
    n_months: int = 6,
    ci_level: float = 0.90,
) -> str:
    """
    Forecast VDI directly from NDVI when precipitation and temperature are unavailable.

    Args:
        aoi: Bounding box [lon_min, lat_min, lon_max, lat_max].
        start_year: First year of the historical period.
        end_year: Last year of the historical period.
        n_months: Number of months to forecast (default 6).
        ci_level: Confidence interval level (default 0.90).
    """
    from drought_monitoring.forecast import forecast_vdi_from_ndvi as _fvdi

    _, _, ndvi = _get_or_fetch(aoi, start_year, end_year)
    fc = _fvdi(ndvi, n_months=n_months, ci_level=ci_level)

    lines = [
        f"VDI-only forecast — {n_months} months ahead "
        f"({ci_level * 100:.0f}% CI, STL + SARIMA on raw NDVI):",
        "",
        fc.round(3).to_string(),
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run() -> None:
    mcp.run()


if __name__ == "__main__":
    run()
