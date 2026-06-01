#!/usr/bin/env python3
"""
Visualizador interativo dos plots PREDEP gerados.

Uso:
  python scripts/viewer.py
  python scripts/viewer.py --port 8050
  python scripts/viewer.py --results-dir /caminho/para/results
"""

import argparse
import os
import re
from pathlib import Path

import flask
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import xarray as xr
from dash import Dash, Input, Output, dash_table, dcc, html

ROOT = Path(__file__).parent
PLOTS_DIR = ROOT / "plots"
RESULTS_DIR = ROOT / "results"
CLUSTERS_DIR = ROOT / "data" / "clusters"

_stats_cache: dict = {}
# (exp, mov) pairs where Brasil-wide data covers ≥2 basins; populated in main()
_valid_brasil: set = set()


# ── scan plots ───────────────────────────────────────────────────────────────


def _viz_label(filename: str) -> str:
    if "optimal" in filename or "winner" in filename:
        return "Lag ótimo"
    return "Todos os lags"


def scan_plots(plots_dir: Path) -> dict:
    """
    Retorna:
      {mov: {area: {tipo: {viz: {exp: Path}}}}}
    area  = "Brasil" | nome da bacia
    tipo  = "PREDEP" | "REGRESSAO" | "COMPARACAO"
    viz   = "Lag ótimo" | "Todos os lags"
    exp   = "exp21" | ...
    """
    index: dict = {}
    for exp_dir in sorted(plots_dir.iterdir()):
        if not exp_dir.is_dir() or not re.match(r"^exp\d+$", exp_dir.name):
            continue
        exp = exp_dir.name
        for mov_dir in sorted(exp_dir.iterdir()):
            if not mov_dir.is_dir() or mov_dir.name == "best_mov":
                continue
            mov = mov_dir.name
            for tipo in ("PREDEP", "REGRESSAO", "COMPARACAO"):
                tipo_dir = mov_dir / tipo
                if not tipo_dir.exists():
                    continue
                brazil_dir = tipo_dir / "BRAZIL"
                if brazil_dir.is_dir():
                    # Skip Brasil-wide plots when only 1 basin was computed
                    if _valid_brasil and (exp, mov) not in _valid_brasil:
                        pass
                    else:
                        for png in sorted(brazil_dir.glob("*.png")):
                            viz = _viz_label(png.name)
                            (index
                             .setdefault(mov, {})
                             .setdefault("Brasil", {})
                             .setdefault(tipo, {})
                             .setdefault(viz, {}))[exp] = png
                bacias_dir = tipo_dir / "BACIAS"
                if bacias_dir.is_dir():
                    for basin_dir in sorted(bacias_dir.iterdir()):
                        if not basin_dir.is_dir():
                            continue
                        for png in sorted(basin_dir.glob("*.png")):
                            viz = _viz_label(png.name)
                            (index
                             .setdefault(mov, {})
                             .setdefault(basin_dir.name, {})
                             .setdefault(tipo, {})
                             .setdefault(viz, {}))[exp] = png
    return index


def scan_best_mov(plots_dir: Path) -> dict:
    """
    Retorna:
      {lag_label: {exp: Path}}
    lag_label = "Geral" | "Lag 0" | "Lag 1" | ...
    """
    index: dict = {}
    for exp_dir in sorted(plots_dir.iterdir()):
        if not exp_dir.is_dir() or not re.match(
            r"^exp\d+$", exp_dir.name
        ):
            continue
        best_dir = exp_dir / "best_mov"
        if not best_dir.is_dir():
            continue
        # "Melhor MoV" only makes sense when ≥2 MoVs were computed
        n_movs = sum(
            1 for d in exp_dir.iterdir()
            if d.is_dir() and d.name != "best_mov"
        )
        if n_movs < 2:
            continue
        exp = exp_dir.name
        for p in sorted(best_dir.glob("best_mov_*.png")):
            label = p.stem.replace("best_mov_", "").replace("_", " ")
            index.setdefault(label, {})[exp] = p
    return index


def compute_valid_brasil(results_dir: Path) -> set:
    """
    Returns {(exp, mov)} pairs where the Brasil-wide NC file exists AND
    at least 2 basins were computed. Used to suppress partial Brasil plots.
    """
    valid: set = set()
    gran_dir = results_dir / "predep_granular_brazil"
    if not gran_dir.exists():
        return valid
    _suffix = "_predep_granular_seasonal"
    for exp_dir in gran_dir.iterdir():
        if not exp_dir.is_dir() or not re.match(r"^exp\d+$", exp_dir.name):
            continue
        exp = exp_dir.name
        for mov_dir in exp_dir.iterdir():
            if not mov_dir.is_dir() or mov_dir.name == "plots":
                continue
            mov = mov_dir.name
            brasil_nc = mov_dir / f"{mov}{_suffix}.nc"
            if not brasil_nc.exists():
                continue
            basin_ncs = [
                f for f in mov_dir.glob(f"*{_suffix}.nc")
                if f != brasil_nc
            ]
            if len(basin_ncs) >= 2:
                valid.add((exp, mov))
    return valid


# ── scan results ─────────────────────────────────────────────────────────────


def scan_results(results_dir: Path) -> dict:
    """
    Retorna {exp: {mov: {basin: Path}}} onde Path é:
      • arquivo .nc  — dados completos por lag
      • diretório    — modo Parquet ({season}.parquet dentro)
    Tenta NC primeiro; se não encontrar, usa Parquet.
    """
    index: dict = {}
    gran_dir = results_dir / "predep_granular_brazil"
    if not gran_dir.exists():
        return index
    _suffix = "_predep_granular_seasonal"
    for exp_dir in sorted(gran_dir.iterdir()):
        if not exp_dir.is_dir() or not re.match(r"^exp\d+$", exp_dir.name):
            continue
        exp = exp_dir.name
        for mov_dir in sorted(exp_dir.iterdir()):
            if not mov_dir.is_dir() or mov_dir.name == "plots":
                continue
            mov = mov_dir.name
            # 1) Try NC files
            nc_found = False
            for nc_file in mov_dir.glob(f"*{_suffix}.nc"):
                stem = nc_file.stem
                after_mov = stem[len(mov) + 1:]
                if not after_mov.endswith(_suffix):
                    continue
                basin = after_mov[: -len(_suffix)]
                if not basin:
                    continue
                (index
                 .setdefault(exp, {})
                 .setdefault(mov, {}))[basin] = nc_file
                nc_found = True
            if nc_found:
                continue
            # 2) Fallback: Parquet files (dir = parquet mode)
            pq_files = list(mov_dir.glob("*.parquet"))
            if not pq_files:
                continue
            df_sample = pd.read_parquet(pq_files[0], columns=["basin"])
            for basin in df_sample["basin"].unique():
                (index
                 .setdefault(exp, {})
                 .setdefault(mov, {}))[str(basin)] = mov_dir
    return index


def compute_mov_stats(
    results_dir: Path, exp: str, basin: str, threshold: float
) -> pd.DataFrame:
    """
    Para cada MoV disponível em (exp, basin), carrega o NetCDF e retorna
    um DataFrame com estatísticas por (MoV, Season):
      Max_R2, Media_R2, N_pixels_R2 (> threshold), Pct_pixels_R2,
      Melhor_lag, Max_alpha, N_pixels_alpha (> threshold), Pct_pixels_alpha,
      N_validos
    """
    key = (str(results_dir), exp, basin, threshold)
    if key in _stats_cache:
        return _stats_cache[key]

    index = scan_results(results_dir)
    seasons_order = ["DJF", "MAM", "JJA", "SON"]
    rows = []
    for mov, basin_map in sorted(index.get(exp, {}).items()):
        src = basin_map.get(basin)
        if src is None:
            continue

        if src.is_dir():
            # ── Parquet mode ──────────────────────────────────────────────
            mov_dir = src
            for season in seasons_order:
                pq = mov_dir / f"{season}.parquet"
                if not pq.exists():
                    continue
                df_s = pd.read_parquet(pq)
                df_b = df_s[df_s["basin"] == basin]
                if df_b.empty:
                    continue
                n_valid = len(df_b)
                r2_max = float(df_b["r2_max"].max())
                r2_mean = float(df_b["r2_mean"].mean())
                n_above_r2 = int((df_b["r2_max"] > threshold).sum())
                pct_r2 = round(100.0 * n_above_r2 / n_valid, 1)
                best_row = df_b.loc[df_b["r2_max"].idxmax()]
                best_lag = int(best_row["best_lag"])
                alpha_max = float(df_b["alpha_max"].max())
                n_above_alpha = int(
                    (df_b["alpha_max"] > threshold).sum()
                )
                pct_alpha = round(100.0 * n_above_alpha / n_valid, 1)
                rows.append({
                    "MoV": mov, "Season": season,
                    "Max_R2": round(r2_max, 4),
                    "Media_R2": round(r2_mean, 4),
                    "N_pixels_R2": n_above_r2,
                    "Pct_pixels_R2": pct_r2,
                    "Melhor_lag": best_lag,
                    "Max_alpha": round(alpha_max, 4),
                    "N_pixels_alpha": n_above_alpha,
                    "Pct_pixels_alpha": pct_alpha,
                    "N_validos": n_valid,
                })
            continue

        # ── NC mode ───────────────────────────────────────────────────────
        nc_path = src
        ds = xr.open_dataset(nc_path)
        r2 = ds["r2"].values           # (season, lag, lat, lon)
        alpha = ds["alpha_core"].values
        ds_seasons = list(ds.coords["season"].values)
        lags = list(ds.coords["lag"].values)
        ds.close()

        for season in seasons_order:
            if season not in ds_seasons:
                continue
            si = ds_seasons.index(season)
            r2_s = r2[si]       # (lag, lat, lon)
            alpha_s = alpha[si]

            valid_mask = ~np.all(np.isnan(r2_s), axis=0)  # (lat, lon)
            n_valid = int(valid_mask.sum())
            if n_valid == 0:
                continue

            with np.errstate(all="ignore"):
                r2_max_pixel = np.nanmax(r2_s, axis=0)
                alpha_max_pixel = np.nanmax(alpha_s, axis=0)

            r2_flat = r2_s[:, valid_mask]               # (lag, n_valid)
            r2_max = float(np.nanmax(r2_flat))
            finite = r2_flat[~np.isnan(r2_flat)]
            r2_mean = (
                float(np.nanmean(finite)) if finite.size > 0
                else float("nan")
            )
            n_above_r2 = int((r2_max_pixel[valid_mask] > threshold).sum())
            pct_r2 = round(100.0 * n_above_r2 / n_valid, 1)

            lag_means = [
                float(np.nanmean(r2_s[li][valid_mask]))
                for li in range(len(lags))
            ]
            best_lag = int(lags[int(np.argmax(lag_means))])

            alpha_flat = alpha_s[:, valid_mask]
            alpha_max = float(np.nanmax(alpha_flat))
            above_alpha = alpha_max_pixel[valid_mask] > threshold
            n_above_alpha = int(above_alpha.sum())
            pct_alpha = round(100.0 * n_above_alpha / n_valid, 1)

            rows.append({
                "MoV": mov,
                "Season": season,
                "Max_R2": round(r2_max, 4),
                "Media_R2": round(r2_mean, 4),
                "N_pixels_R2": n_above_r2,
                "Pct_pixels_R2": pct_r2,
                "Melhor_lag": best_lag,
                "Max_alpha": round(alpha_max, 4),
                "N_pixels_alpha": n_above_alpha,
                "Pct_pixels_alpha": pct_alpha,
                "N_validos": n_valid,
            })

    df = pd.DataFrame(rows)
    _stats_cache[key] = df
    return df


# ── app ──────────────────────────────────────────────────────────────────────

app = Dash(__name__, title="PREDEP Viewer")
server = app.server


@server.route("/plots/<path:filepath>")
def serve_plot(filepath: str):
    return flask.send_from_directory(str(PLOTS_DIR), filepath)


def _opts(items: list) -> list:
    return [{"label": i, "value": i} for i in items]


_ROW = {
    "display": "flex", "gap": "16px", "flexWrap": "wrap",
    "marginBottom": "16px", "alignItems": "flex-end",
}
_HIDDEN = {"display": "none"}
_DD = {"flex": "1", "minWidth": "130px"}
_EXP_LABEL = {
    "fontWeight": "600", "fontSize": "13px", "color": "#555",
    "marginBottom": "4px", "marginTop": "8px",
}
_RADIO_DIV = {
    "flex": "2", "minWidth": "200px",
    "alignSelf": "flex-end", "paddingBottom": "6px",
}
_CARD = {
    "background": "#f0f4f8", "borderRadius": "6px",
    "padding": "10px 16px", "marginBottom": "20px",
    "fontSize": "14px", "color": "#333",
}


def _load_basin_rings(clusters_dir: Path, basin: str) -> list:
    """Returns list of (lons, lats) rings for the basin shapefile."""
    try:
        import shapefile as pyshp
    except ImportError:
        return []
    basin_dir = clusters_dir / basin
    shp_files = list(basin_dir.glob("*.shp"))
    if not shp_files:
        return []
    sf = pyshp.Reader(str(shp_files[0]))
    rings = []
    for shape in sf.shapes():
        pts = shape.points
        rings.append(([p[0] for p in pts], [p[1] for p in pts]))
    return rings


def _map_grid_from_nc(nc_path: Path, season: str):
    """Retorna (lons, lats, best_r2, best_alpha, season_used) do NetCDF."""
    ds = xr.open_dataset(nc_path)
    lats = ds.coords["latitude"].values
    lons = ds.coords["longitude"].values
    seasons_nc = list(ds.coords["season"].values)
    r2_arr = ds["r2"].values        # (season, lag, lat, lon)
    alpha_arr = ds["alpha_core"].values
    ds.close()

    if season != "Todas" and season in seasons_nc:
        si = seasons_nc.index(season)
        season_used = season
        r2_3d = r2_arr[si]
        alpha_3d = alpha_arr[si]
    else:
        r2_3d = np.nanmax(r2_arr, axis=0)
        alpha_3d = np.nanmax(alpha_arr, axis=0)
        season_used = "Todas"

    with np.errstate(all="ignore"):
        best_r2 = np.nanmax(r2_3d, axis=0)
        best_alpha = np.nanmax(alpha_3d, axis=0)
    return lons, lats, best_r2, best_alpha, season_used


def _map_grid_from_parquet(mov_dir: Path, basin: str, season: str):
    """
    Retorna (lons, lats, best_r2, best_alpha, season_used) a partir dos
    Parquet (formato longo, max-por-pixel já pré-calculado).
    """
    seasons_all = ["DJF", "MAM", "JJA", "SON"]
    if season != "Todas":
        targets = [season]
        season_used = season
    else:
        targets = seasons_all
        season_used = "Todas"

    frames = []
    for s in targets:
        pq = mov_dir / f"{s}.parquet"
        if pq.exists():
            df = pd.read_parquet(pq)
            frames.append(df[df["basin"] == basin])
    if not frames:
        return None
    df_all = pd.concat(frames, ignore_index=True)
    if df_all.empty:
        return None

    # Max sobre seasons (quando "Todas") por pixel (lat, lon)
    agg = df_all.groupby(["latitude", "longitude"], as_index=False).agg(
        r2_max=("r2_max", "max"),
        alpha_max=("alpha_max", "max"),
    )
    piv_r2 = agg.pivot(
        index="latitude", columns="longitude", values="r2_max"
    ).sort_index()
    piv_al = agg.pivot(
        index="latitude", columns="longitude", values="alpha_max"
    ).sort_index()
    lats = piv_r2.index.values
    lons = piv_r2.columns.values
    best_r2 = piv_r2.values
    best_alpha = piv_al.values
    return lons, lats, best_r2, best_alpha, season_used


def _movs_for_basin(plots_dir: Path, exp: str, basin: str) -> list:
    """Sorted MoVs that have BACIAS plots for this (exp, basin)."""
    movs = []
    exp_dir = plots_dir / exp
    if not exp_dir.is_dir():
        return movs
    for mov_dir in sorted(exp_dir.iterdir()):
        if not mov_dir.is_dir():
            continue
        mov = mov_dir.name
        for tipo in ("PREDEP", "REGRESSAO", "COMPARACAO"):
            basin_dir = mov_dir / tipo / "BACIAS" / basin
            if basin_dir.is_dir() and any(basin_dir.glob("*.png")):
                movs.append(mov)
                break
    return movs


def _get_map_image(
    plots_dir: Path, exp: str, mov: str,
    tipo: str, viz: str, basin: str,
):
    """Returns the PNG Path for (exp, mov, tipo, viz, basin), or None."""
    if not all([exp, mov, tipo, viz, basin]):
        return None
    basin_dir = plots_dir / exp / mov / tipo / "BACIAS" / basin
    if not basin_dir.is_dir():
        return None
    for png in sorted(basin_dir.glob("*.png")):
        if _viz_label(png.name) == viz:
            return png
    return None


def _pretty_basin(basin: str) -> str:
    return basin.replace("_", " ").title()


def _basin_opts(basins: list) -> list:
    return [{"label": _pretty_basin(b), "value": b} for b in basins]


def _exp_opts(results_index: dict) -> list:
    def _fmt(items, n=4):
        if len(items) <= n:
            return ", ".join(items)
        return ", ".join(items[:n]) + f" (+{len(items) - n})"

    opts = []
    for exp in sorted(results_index.keys()):
        movs = sorted(results_index[exp].keys())
        basins = sorted({
            b for mv in results_index[exp].values() for b in mv
        })
        basin_labels = [_pretty_basin(b) for b in basins]
        label = (
            f"{exp} — MOVs: {_fmt(movs)}"
            f"  |  Bacias: {_fmt(basin_labels)}"
        )
        opts.append({"label": label, "value": exp})
    return opts


def _build_layout(results_index: dict) -> html.Div:
    exps_r = sorted(results_index.keys())
    first_exp_r = exps_r[0] if exps_r else None
    first_basins = sorted({
        basin
        for mov_map in results_index.get(first_exp_r or "", {}).values()
        for basin in mov_map
    })
    first_basin = first_basins[0] if first_basins else None

    slider_marks = {
        0.0: "0.0", 0.05: "0.05", 0.10: "0.10", 0.15: "0.15",
        0.20: "0.20", 0.25: "0.25", 0.30: "0.30", 0.35: "0.35",
        0.40: "0.40", 0.45: "0.45", 0.50: "0.50",
    }

    return html.Div([
        html.H2(
            "PREDEP — Exploração",
            style={"marginBottom": "16px", "fontWeight": "600"},
        ),

        # ── exploração controls ──────────────────────────────────────────────
        html.Div([
            html.Div([
                html.Label("Experimento", style={"fontWeight": "500"}),
                dcc.Dropdown(
                    id="dd-exp-explore",
                    options=_exp_opts(results_index),
                    value=first_exp_r,
                    clearable=False,
                ),
            ], style={**_DD, "minWidth": "300px", "flex": "3"}),
            html.Div([
                html.Label("Cluster (bacia)", style={"fontWeight": "500"}),
                dcc.Dropdown(
                    id="dd-cluster-explore",
                    options=_basin_opts(first_basins),
                    value=first_basin,
                    clearable=False,
                ),
            ], style=_DD),
            html.Div([
                html.Label("Threshold R² / α", style={"fontWeight": "500"}),
                dcc.Slider(
                    id="sl-r2-threshold",
                    min=0.0, max=0.5, step=0.05, value=0.2,
                    marks=slider_marks,
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
            ], style={
                "flex": "2", "minWidth": "260px", "paddingBottom": "6px",
            }),
            html.Div([
                html.Label("Season", style={"fontWeight": "500"}),
                dcc.RadioItems(
                    id="ri-season-explore",
                    options=[
                        {"label": s, "value": s}
                        for s in ["Todas", "DJF", "MAM", "JJA", "SON"]
                    ],
                    value="Todas",
                    inline=True,
                    labelStyle={"marginRight": "10px"},
                ),
            ], style=_RADIO_DIV),
            # ── map sub-row ──────────────────────────────────────────────────
            html.Div(style={
                "flexBasis": "100%", "height": "0",
                "borderTop": "1px solid #e0e0e0", "margin": "4px 0",
            }),
            html.Div([
                html.Label("MoV (mapa)", style={"fontWeight": "500"}),
                dcc.Dropdown(
                    id="dd-mov-map",
                    options=[],
                    value=None,
                    clearable=False,
                    placeholder="selecione um MoV...",
                ),
            ], style=_DD),
        ], style={**_ROW, "marginBottom": "20px"}),

        html.Div(id="explore-content"),
    ], style={
        "fontFamily": "sans-serif",
        "padding": "24px",
        "maxWidth": "1600px",
        "margin": "0 auto",
    })


# Layout populado já no import (necessário para gunicorn/`app:server`,
# onde main() nunca executa). main() apenas re-popula se rodar via CLI.
_valid_brasil = compute_valid_brasil(RESULTS_DIR)
app.layout = _build_layout(scan_results(RESULTS_DIR))


# ── callbacks ────────────────────────────────────────────────────────────────

@app.callback(
    Output("dd-cluster-explore", "options"),
    Output("dd-cluster-explore", "value"),
    Input("dd-exp-explore", "value"),
)
def cb_clusters_explore(exp: str):
    if not exp:
        return [], None
    index = scan_results(RESULTS_DIR)
    basins = sorted({
        basin
        for mov_map in index.get(exp, {}).values()
        for basin in mov_map
    })
    return _basin_opts(basins), (basins[0] if basins else None)


@app.callback(
    Output("dd-mov-map", "options"),
    Output("dd-mov-map", "value"),
    Input("dd-exp-explore",     "value"),
    Input("dd-cluster-explore", "value"),
)
def cb_mov_map_opts(exp: str, basin: str):
    if not exp or not basin:
        return [], None
    index = scan_results(RESULTS_DIR)
    movs = sorted(
        mov for mov, bmap in index.get(exp, {}).items()
        if basin in bmap
    )
    return _opts(movs), (movs[0] if movs else None)


@app.callback(
    Output("explore-content", "children"),
    Input("dd-exp-explore",     "value"),
    Input("dd-cluster-explore", "value"),
    Input("sl-r2-threshold",    "value"),
    Input("ri-season-explore",  "value"),
    Input("dd-mov-map",         "value"),
)
def cb_explore_content(
    exp: str, basin: str, threshold,
    season: str, mov_map: str,
):
    if not exp or not basin:
        return html.P("Selecione um experimento e um cluster.")

    th = float(threshold) if threshold is not None else 0.2
    df = compute_mov_stats(RESULTS_DIR, exp, basin, th)

    if df.empty:
        return html.P("Nenhum dado encontrado para esta seleção.")

    df_view = df if season == "Todas" else df[df["Season"] == season]

    if df_view.empty:
        return html.P(f"Sem dados para a season {season}.")

    # ── data table ───────────────────────────────────────────────────────────
    _i = " ⓘ"
    col_map = [
        ("MoV",            f"MoV{_i}"),
        ("Season",         f"Season{_i}"),
        ("Max_R2",         f"Max R²{_i}"),
        ("Media_R2",       f"Média R²{_i}"),
        ("N_pixels_R2",    f"N pixels R² > {th:.2f}{_i}"),
        ("Pct_pixels_R2",  f"% pixels R²{_i}"),
        ("Melhor_lag",     f"Melhor lag{_i}"),
        ("Max_alpha",      f"Max α{_i}"),
        ("N_pixels_alpha", f"N pixels α > {th:.2f}{_i}"),
        ("Pct_pixels_alpha", f"% pixels α{_i}"),
        ("N_validos",      f"N válidos{_i}"),
    ]
    columns = [
        {
            "name": label, "id": col_id,
            "type": "text" if col_id in ("MoV", "Season") else "numeric",
        }
        for col_id, label in col_map
    ]
    _col_tips = {
        "MoV": (
            "**Modo de Variabilidade**\n\n"
            "Índice climático analisado (ex: ONI, AMO, NAO). "
            "Cada linha corresponde a um MoV × estação."
        ),
        "Season": (
            "**Estação do ano**\n\n"
            "DJF = Dez-Jan-Fev | MAM = Mar-Abr-Mai\n\n"
            "JJA = Jun-Jul-Ago | SON = Set-Out-Nov"
        ),
        "Max_R2": (
            "**Máximo R²**\n\n"
            "Maior coeficiente de determinação (regressão linear) "
            "encontrado em qualquer pixel e lag dentro da bacia. "
            "Mede a componente **linear** da relação MoV → precipitação."
        ),
        "Media_R2": (
            "**Média R²**\n\n"
            "Média do R² sobre todos os pixels e lags válidos da bacia. "
            "Representa o nível médio de correlação linear."
        ),
        "N_pixels_R2": (
            f"**N pixels com R² > {th:.2f}**\n\n"
            "Número de pixels únicos onde o **melhor lag** tem "
            f"R² acima de {th:.2f}. "
            "Indica a extensão espacial do sinal linear."
        ),
        "Pct_pixels_R2": (
            f"**% pixels com R² > {th:.2f}**\n\n"
            "Percentual de pixels válidos da bacia com "
            f"R² > {th:.2f} no melhor lag."
        ),
        "Melhor_lag": (
            "**Melhor lag (meses)**\n\n"
            "Lag com maior R² médio nos pixels válidos da bacia. "
            "Indica o tempo de resposta típico da precipitação ao MoV."
        ),
        "Max_alpha": (
            "**Máximo α (PREDEP)**\n\n"
            "Maior valor de α encontrado em qualquer pixel e lag. "
            "α mede a dependência **não-linear** do MoV sobre a "
            "precipitação (α=0 = independência; α=1 = previsão perfeita). "
            "Pode ser alto mesmo quando R² é baixo."
        ),
        "N_pixels_alpha": (
            f"**N pixels com α > {th:.2f}**\n\n"
            "Número de pixels únicos onde o **melhor lag** tem "
            f"α (PREDEP) acima de {th:.2f}. "
            "Indica a extensão espacial do sinal não-linear."
        ),
        "Pct_pixels_alpha": (
            f"**% pixels com α > {th:.2f}**\n\n"
            "Percentual de pixels válidos da bacia com "
            f"α > {th:.2f} no melhor lag."
        ),
        "N_validos": (
            "**N pixels válidos**\n\n"
            "Total de pixels dentro da bacia com pelo menos "
            "um lag não-NaN. Pixels fora do polígono da bacia "
            "ou sem dados suficientes são excluídos."
        ),
    }
    tooltip_header = {
        col_id: {"value": tip, "type": "markdown"}
        for col_id, tip in _col_tips.items()
    }
    style_cond = [
        {
            "if": {
                "filter_query": f"{{Max_R2}} > {th}",
                "column_id": "Max_R2",
            },
            "backgroundColor": "#fff3e0",
            "color": "#b35900",
            "fontWeight": "600",
        },
        {
            "if": {"row_index": "odd"},
            "backgroundColor": "#fafafa",
        },
    ]
    table = dash_table.DataTable(
        data=df_view.to_dict("records"),
        columns=columns,
        sort_action="native",
        sort_mode="single",
        page_size=60,
        tooltip_header=tooltip_header,
        tooltip_delay=0,
        tooltip_duration=None,
        style_table={"overflowX": "auto"},
        style_cell={
            "fontSize": "13px",
            "padding": "5px 10px",
            "whiteSpace": "nowrap",
            "textAlign": "center",
        },
        style_cell_conditional=[
            {"if": {"column_id": "MoV"}, "textAlign": "left"},
        ],
        style_header={
            "fontWeight": "600",
            "backgroundColor": "#f5f5f5",
            "borderBottom": "2px solid #ddd",
            "textAlign": "center",
            "cursor": "help",
        },
        style_data_conditional=style_cond,
    )

    title_table = (
        f"Tabela detalhada — {season} / {basin} ({exp})"
        if season != "Todas"
        else (
            f"Tabela detalhada — todas as seasons / {basin} ({exp})"
        )
    )

    # ── interactive pixel map (top) ──────────────────────────────────────────
    index_r = scan_results(RESULTS_DIR)
    src = (
        index_r.get(exp, {}).get(mov_map, {}).get(basin)
        if mov_map else None
    )
    map_data = None
    if src is not None and src.is_dir():
        map_data = _map_grid_from_parquet(src, basin, season)
    elif src is not None:
        map_data = _map_grid_from_nc(src, season)

    if map_data is not None:
        lons, lats, best_r2, best_alpha, season_used = map_data

        colorscale = [
            [0.00, "white"],
            [0.15, "#feedde"],
            [0.40, "#fdae6b"],
            [0.70, "#e6550d"],
            [1.00, "#a63603"],
        ]
        rings = _load_basin_rings(CLUSTERS_DIR, basin)

        fig_map = make_subplots(
            rows=1, cols=2,
            subplot_titles=["R²  (regressão linear)", "α  (PREDEP)"],
            horizontal_spacing=0.06,
        )
        for col, (z_data, hover_lbl) in enumerate(
            [(best_r2, "R²"), (best_alpha, "α")], start=1
        ):
            fig_map.add_trace(go.Heatmap(
                x=lons, y=lats, z=z_data,
                coloraxis="coloraxis",
                hovertemplate=(
                    "Lon: %{x:.3f}<br>Lat: %{y:.3f}"
                    f"<br>{hover_lbl}: %{{z:.4f}}<extra></extra>"
                ),
            ), row=1, col=col)
            for ring_lons, ring_lats in rings:
                fig_map.add_trace(go.Scatter(
                    x=ring_lons, y=ring_lats,
                    mode="lines",
                    line=dict(color="black", width=1),
                    showlegend=False,
                    hoverinfo="skip",
                ), row=1, col=col)

        map_title = (
            f"R² e α — {mov_map} | {season_used} | {basin}"
        )
        fig_map.update_layout(
            title=dict(text=map_title, font=dict(size=13), x=0),
            coloraxis=dict(
                colorscale=colorscale,
                cmin=0.05,
                colorbar=dict(title="valor", thickness=14),
            ),
            xaxis=dict(showgrid=False, scaleanchor="y",  scaleratio=1),
            yaxis=dict(showgrid=False),
            xaxis2=dict(showgrid=False, scaleanchor="y2", scaleratio=1),
            yaxis2=dict(showgrid=False),
            margin=dict(l=60, r=60, t=60, b=50),
            height=520,
            dragmode="zoom",
        )
        map_section = dcc.Graph(
            figure=fig_map,
            config={
                "scrollZoom": True,
                "displayModeBar": True,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                "toImageButtonOptions": {"filename": map_title},
            },
            style={"width": "100%", "marginBottom": "20px"},
        )
    else:
        map_section = html.P(
            "Selecione MoV para ver o mapa interativo.",
            style={
                "color": "#aaa", "fontStyle": "italic",
                "marginBottom": "20px",
            },
        )

    # ── table (bottom) ───────────────────────────────────────────────────────
    return [
        map_section,
        html.H4(title_table, style={
            "marginBottom": "8px", "fontWeight": "500", "fontSize": "15px",
        }),
        table,
    ]


def main():
    global PLOTS_DIR, RESULTS_DIR, _valid_brasil

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", "8050")),
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("HOST", "0.0.0.0"),
    )
    parser.add_argument("--plots-dir", type=str, default="")
    parser.add_argument("--results-dir", type=str, default="")
    args = parser.parse_args()

    if args.plots_dir:
        PLOTS_DIR = Path(args.plots_dir)
    if args.results_dir:
        RESULTS_DIR = Path(args.results_dir)

    _valid_brasil = compute_valid_brasil(RESULTS_DIR)
    results_index = scan_results(RESULTS_DIR)
    app.layout = _build_layout(results_index)

    exps_r = sorted(results_index.keys())
    print(f"Experimentos disponíveis: {exps_r}")
    print(f"Abrindo em   http://localhost:{args.port}/")
    app.run(debug=False, port=args.port, host=args.host)


if __name__ == "__main__":
    main()
