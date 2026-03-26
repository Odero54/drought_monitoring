"""
drought_cdi.plot
================
Publication-quality visualisation for CDI outputs.

Design follows conventions used in atmospheric and climate science journals
(Nature Climate Change, J. Hydrometeorology, IPCC reports):

  * Diverging brown-white-blue colourmap centred at zero.
  * Anomaly shading + 12-month running mean on CDI panel.
  * Severity-classification stripe above the main time series.
  * Minimal chrome: no top/right spines, neutral grey grid, 7-8 pt fonts.
  * Hovmöller latitude-time diagram for spatial output.
  * Annual anomaly bar chart (climate-report staple).
  * Single-date 2-D spatial map snapshot.

All figures return a ``matplotlib.figure.Figure`` for saving or display.
"""

from __future__ import annotations

import calendar


import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches
    import matplotlib.ticker as mticker
    import matplotlib.gridspec as gridspec
    import matplotlib.dates as mdates
    from matplotlib.lines import Line2D
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


CDI_CLASSES = [
    (-np.inf, -2.0, "#6B0000", "Extreme drought"),
    (-2.0,    -1.5, "#C0392B", "Severe drought"),
    (-1.5,    -1.0, "#E67E22", "Moderate drought"),
    (-1.0,    -0.5, "#F1C40F", "Mild drought"),
    (-0.5,     0.5, "#F0F0F0", "Near normal"),
    ( 0.5,     1.0, "#AED6F1", "Mild wet"),
    ( 1.0,     1.5, "#2E86C1", "Moderately wet"),
    ( 1.5,  np.inf, "#1A5276", "Very wet"),
]


def classify_cdi(series: pd.Series) -> pd.Series:
    """Map numeric CDI values to severity label strings."""
    out = pd.Series("", index=series.index, dtype=str)
    for lo, hi, _, label in CDI_CLASSES:
        out[(series >= lo) & (series < hi)] = label
    return out

def _build_cdi_cmap(n=512):
    colours = [
        "#6B0000", "#C0392B", "#E67E22", "#F1C40F",
        "#FAFAFA",
        "#AED6F1", "#2E86C1", "#1A5276",
    ]
    return mcolors.LinearSegmentedColormap.from_list("CDI", colours, N=n)

CDI_CMAP = _build_cdi_cmap()

def _cdi_norm(vmin=-3.0, vmax=3.0):
    return mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

def _require_mpl():
    if not _HAS_MPL:
        raise ImportError(
            "matplotlib is required for plotting.\n"
            "Install it with:  pip install drought-cdi[plot]"
        )

def _climate_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#AAAAAA")
    ax.spines["bottom"].set_color("#AAAAAA")
    ax.tick_params(colors="#444444", labelsize=7)
    ax.grid(axis="y", color="#E0E0E0", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

def _severity_handles():
    return [
        mpatches.Patch(facecolor=c, edgecolor="none", label=label)
        for _, _, c, label in CDI_CLASSES
    ]

def plot_timeseries(
    df,
    title="Composite Drought Index",
    subtitle="",
    figsize=(14, 10),
    show_components=True,
    show_severity_bar=True,
    vmin=-3.5,
    vmax=3.5,
):
    """
    Publication-quality CDI time-series figure.

    Parameters
    ----------
    df               : pd.DataFrame – must contain 'CDI'; optionally 'PDI','TDI','VDI'.
    title            : str – main figure title.
    subtitle         : str – optional second line (e.g. 'Borena Region  |  2000–2020').
    figsize          : tuple.
    show_components  : bool – add PDI / TDI / VDI sub-panels below CDI.
    show_severity_bar: bool – add colour-coded severity stripe at top.
    vmin / vmax      : float – y-axis / colour limits.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _require_mpl()
    has_comp = show_components and all(c in df.columns for c in ("PDI","TDI","VDI"))

    heights = (([0.28] if show_severity_bar else []) +
               [3.5] +
               ([1.4, 1.4, 1.4] if has_comp else []))
    n_rows = len(heights)
    fig = plt.figure(figsize=figsize, facecolor="white")
    gs  = gridspec.GridSpec(n_rows, 1, height_ratios=heights, hspace=0.55, figure=fig)
    row = 0
    if show_severity_bar:
        ax_bar = fig.add_subplot(gs[row])
        row += 1
        _draw_severity_bar(ax_bar, df["CDI"])

    ax_cdi = fig.add_subplot(gs[row])
    row += 1
    _draw_cdi_panel(ax_cdi, df["CDI"], vmin, vmax)
    full_title = title + (f"\n{subtitle}" if subtitle else "")
    ax_cdi.set_title(full_title, fontsize=10, fontweight="bold",
                     pad=10, loc="left", color="#111111")
    if show_severity_bar:
        ax_bar.sharex(ax_cdi)
    if has_comp:
        cfgs = [
            ("PDI", "#1A5276", "Precipitation Drought Index (PDI)"),
            ("TDI", "#922B21", "Temperature Drought Index (TDI)"),
            ("VDI", "#1E8449", "Vegetation Drought Index (VDI)"),
        ]
        for col, colour, label in cfgs:
            ax = fig.add_subplot(gs[row], sharex=ax_cdi)
            row += 1
            _draw_component_panel(ax, df[col], colour, label, vmin, vmax)
    fig.legend(
        handles=_severity_handles(),
        loc="center right",
        bbox_to_anchor=(1.18, 0.5),
        fontsize=6.5,
        title="Severity class",
        title_fontsize=7,
        frameon=True,
        framealpha=0.9,
        edgecolor="#CCCCCC",
    )
    return fig

def _draw_severity_bar(ax, cdi):
    cdi = cdi.dropna()
    for i in range(len(cdi) - 1):
        val = cdi.iloc[i]
        colour = "#F0F0F0"
        for lo, hi, c, _ in CDI_CLASSES:
            if lo <= val < hi:
                colour = c
                break
        ax.axvspan(cdi.index[i], cdi.index[i+1], color=colour, alpha=0.92, linewidth=0)
    ax.set_yticks([])
    ax.set_ylabel("Severity", fontsize=6, rotation=0, labelpad=38, va="center", color="#555")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(bottom=False, labelbottom=False)

def _draw_cdi_panel(ax, cdi, vmin, vmax):
    cdi = cdi.dropna()
    _climate_style(ax)
    ax.axhline(0, color="#666", linewidth=0.9, zorder=2)
    for v, ls in [(-1.0,":"),(-2.0,"--"),(1.0,":"),(2.0,"--")]:
        col = "#B03A2E" if v < 0 else "#1F618D"
        ax.axhline(v, color=col, linewidth=0.75, linestyle=ls, alpha=0.55, zorder=2)
    ax.fill_between(cdi.index, cdi, 0, where=(cdi<0),
                    color="#C0392B", alpha=0.45, linewidth=0, zorder=3)
    ax.fill_between(cdi.index, cdi, 0, where=(cdi>=0),
                    color="#2980B9", alpha=0.35, linewidth=0, zorder=3)
    ax.plot(cdi.index, cdi, color="#1A1A1A", linewidth=0.8, zorder=4)
    rm = cdi.rolling(12, center=True, min_periods=6).mean()
    ax.plot(rm.index, rm, color="#E67E22", linewidth=1.6, zorder=5, label="12-month mean")
    ax.set_ylim(vmin, vmax)
    ax.set_ylabel("CDI", fontsize=8)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(1))
    handles = [
        Line2D([0],[0], color="#C0392B", alpha=0.8, linewidth=5, label="Drought"),
        Line2D([0],[0], color="#2980B9", alpha=0.6, linewidth=5, label="Wet"),
        Line2D([0],[0], color="#E67E22", linewidth=1.6, label="12-month mean"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=6.5,
              frameon=True, framealpha=0.85, edgecolor="#CCC")

def _draw_component_panel(ax, s, colour, label, vmin, vmax):
    s = s.dropna()
    _climate_style(ax)
    ax.axhline(0, color="#999", linewidth=0.7)
    ax.fill_between(s.index, s, 0, where=(s<0),  color=colour, alpha=0.30, linewidth=0)
    ax.fill_between(s.index, s, 0, where=(s>=0), color=colour, alpha=0.12, linewidth=0)
    ax.plot(s.index, s, color=colour, linewidth=0.8)
    ax.set_ylim(vmin, vmax)
    ax.set_ylabel(label.split(" ")[0], fontsize=7.5)
    ax.set_title(label, fontsize=7.5, loc="left", pad=3, color="#333")
    ax.yaxis.set_major_locator(mticker.MultipleLocator(1))

def plot_seasonal_cycle(df, title="Seasonal Cycle of Drought Indices", figsize=(12, 4)):
    """Monthly climatology (mean ± 1σ and ± 2σ) for each index."""
    _require_mpl()
    month_labels = [calendar.month_abbr[m] for m in range(1,13)]
    cfgs = [("PDI","#1A5276"),("TDI","#922B21"),("VDI","#1E8449"),("CDI","#6C3483")]
    present = [(c,d) for c,d in cfgs if c in df.columns]
    fig, axes = plt.subplots(1, len(present), figsize=figsize, sharey=False, facecolor="white")
    if len(present) == 1:
        axes = [axes]
    for ax, (col, colour) in zip(axes, present):
        mu = df[col].groupby(df.index.month).mean()
        sd = df[col].groupby(df.index.month).std()
        ax.fill_between(mu.index, mu-2*sd, mu+2*sd, color=colour, alpha=0.10, linewidth=0)
        ax.fill_between(mu.index, mu-sd,   mu+sd,   color=colour, alpha=0.22, linewidth=0)
        ax.plot(mu.index, mu, color=colour, linewidth=1.6, marker="o", markersize=3, zorder=4)
        ax.axhline(0, color="#999", linewidth=0.7, linestyle="--")
        ax.set_xticks(range(1,13))
        ax.set_xticklabels(month_labels, fontsize=7, rotation=45)
        ax.set_title(col, fontsize=9, fontweight="bold", color=colour)
        ax.set_ylabel("Index value", fontsize=7.5)
        _climate_style(ax)
        # ±1σ and ±2σ annotation
        ax.text(0.02, 0.97, "Shading: ±1σ / ±2σ", transform=ax.transAxes,
                fontsize=6, va="top", color="#666")
    fig.suptitle(title, fontsize=10, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig

def plot_anomaly_bars(df, column="CDI", freq="YE",
                     title="Annual Mean CDI Anomaly", figsize=(12, 4)):
    """
    Coloured anomaly bar chart — standard in IPCC and WMO drought reports.

    Parameters
    ----------
    df     : pd.DataFrame with DatetimeIndex.
    column : column to plot (default 'CDI').
    freq   : 'YE' annual | 'QE' seasonal | 'ME' monthly.
    """
    _require_mpl()
    rs = df[column].resample(freq).mean().dropna()
    colours = ["#C0392B" if v < 0 else "#2980B9" for v in rs.values]
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    _climate_style(ax)
    width = {"YE": 300, "QE": 60, "ME": 20}.get(freq, 50)
    ax.bar(rs.index, rs.values, width=width, color=colours, edgecolor="none", zorder=3)
    ax.axhline(0,    color="#333", linewidth=0.9, zorder=4)
    ax.axhline(-1.0, color="#E74C3C", linewidth=0.7, linestyle=":", alpha=0.7)
    ax.axhline(-2.0, color="#922B21", linewidth=0.7, linestyle="--", alpha=0.8)
    ax.axhline( 1.0, color="#2980B9", linewidth=0.7, linestyle=":", alpha=0.7)
    ax.axhline( 2.0, color="#1A5276", linewidth=0.7, linestyle="--", alpha=0.8)
    ax.set_ylabel(f"Mean {column}", fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold", loc="left", pad=8, color="#111")
    handles = [mpatches.Patch(color="#C0392B", label="Drought (<0)"),
               mpatches.Patch(color="#2980B9", label="Wet (≥0)")]
    ax.legend(handles=handles, fontsize=7, loc="upper right",
              frameon=True, framealpha=0.9, edgecolor="#CCC")
    fig.tight_layout()
    return fig

def plot_hovmoller(da, lat_dim="latitude",
                  title="CDI Hovmöller Diagram (latitude–time)",
                  figsize=(12, 6), vmin=-3.0, vmax=3.0):
    """
    Latitude–time Hovmöller diagram — standard in atmospheric science.

    Parameters
    ----------
    da      : xr.DataArray  (time, latitude, longitude) – longitude is averaged.
    lat_dim : str.
    """
    _require_mpl()
    lon_dim = next((d for d in da.dims if d not in (lat_dim, "time")), None)
    data_2d = da.mean(dim=lon_dim) if lon_dim else da
    time_vals = pd.DatetimeIndex(data_2d["time"].values)
    lat_vals  = data_2d[lat_dim].values
    Z = data_2d.values.T   # (lat, time)

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    im = ax.pcolormesh(time_vals, lat_vals, Z,
                       cmap=CDI_CMAP, norm=_cdi_norm(vmin, vmax), shading="auto")
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02, aspect=25)
    cb.set_label("CDI", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    try:
        t_num = mdates.date2num(time_vals)
        TG, LG = np.meshgrid(t_num, lat_vals)
        ax.contour(TG, LG, Z, levels=[-2,-1,1,2],
                   colors=["#6B0000","#E67E22","#2E86C1","#1A5276"],
                   linewidths=0.6, alpha=0.65)
    except Exception:
        pass
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Latitude (°)", fontsize=8)
    ax.set_xlabel("Year", fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold", loc="left", pad=8, color="#111")
    _climate_style(ax)
    fig.tight_layout()
    return fig

def plot_map_snapshot(da, date=None, title="", figsize=(8, 6),
                      vmin=-3.0, vmax=3.0, add_colorbar=True, add_legend=True):
    """
    2-D spatial map of a single CDI time-step.

    Parameters
    ----------
    da   : xr.DataArray (time, lat, lon) or (lat, lon).
    date : str  ISO date e.g. '2015-07'.  If None, uses last time-step.
    """
    _require_mpl()
    if "time" in da.dims:
        da = da.sel(time=date, method="nearest") if date else da.isel(time=-1)
        auto_date = str(pd.Timestamp(da["time"].values))[:7]
    else:
        auto_date = ""

    if not title:
        name = getattr(da, "name", "CDI") or "CDI"
        title = f"{name}  –  {auto_date}"

    lat_dim = next((d for d in da.dims if d in ("latitude","lat","y")), da.dims[0])
    lon_dim = next((d for d in da.dims if d in ("longitude","lon","x")), da.dims[1])
    lats = da[lat_dim].values
    lons = da[lon_dim].values
    arr  = np.asarray(da, dtype=float)

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    im = ax.pcolormesh(lons, lats, arr,
                       cmap=CDI_CMAP, norm=_cdi_norm(vmin, vmax), shading="auto")
    if add_colorbar:
        cb = fig.colorbar(im, ax=ax, shrink=0.80, pad=0.03, aspect=20)
        cb.set_label("Index value", fontsize=8)
        cb.ax.tick_params(labelsize=7)
        for v in [-2, -1, 1, 2]:
            cb.ax.axhline(v, color="#444", linewidth=0.6, linestyle="--")
    if add_legend:
        ax.legend(handles=_severity_handles(), loc="lower left",
                  fontsize=6.5, frameon=True, framealpha=0.9,
                  edgecolor="#CCC", title="Severity", title_fontsize=7)
    ax.set_xlabel("Longitude (°)", fontsize=8)
    ax.set_ylabel("Latitude (°)",  fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold", loc="left", pad=8, color="#111")
    _climate_style(ax)
    fig.tight_layout()
    return fig