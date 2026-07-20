#!/usr/bin/env python3
"""
Visualizador interativo dos plots PREDEP gerados.

Uso:
  python scripts/viewer.py
  python scripts/viewer.py --port 8050
  python scripts/viewer.py --results-dir /caminho/para/results
"""

import argparse
import gc
import json
import os
import re
from pathlib import Path

import flask
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.colors as pcolors
from plotly.subplots import make_subplots
import xarray as xr
from dash import Dash, Input, Output, State, dash_table, dcc, html

ROOT = Path(__file__).parent
PLOTS_DIR = ROOT / "plots"
RESULTS_DIR = ROOT / "results"
CLUSTERS_DIR = ROOT / "data" / "clusters"

# LRU-1 cache for SOM (small, ~0.8 MB per entry)
_som_cache: dict = {}
_som_cache_key = None

# Lazy cache for scan_results (lightweight, only paths)
_results_index_cache: dict | None = None
_results_index_mtime: float = 0.0


def _get_results_index(results_dir: Path) -> dict:
    """Cached scan_results — avoids re-scanning on every callback."""
    global _results_index_cache, _results_index_mtime
    gran_dir = results_dir / "predep_granular_brazil"
    try:
        cur_mtime = gran_dir.stat().st_mtime if gran_dir.exists() else 0.0
    except OSError:
        cur_mtime = 0.0
    if _results_index_cache is None or cur_mtime != _results_index_mtime:
        _results_index_cache = scan_results(results_dir)
        _results_index_mtime = cur_mtime
    return _results_index_cache



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
        if not exp_dir.is_dir() or not re.match(r"^exp[\w]+$", exp_dir.name):
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
            r"^exp[\w]+$", exp_dir.name
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
        if not exp_dir.is_dir() or not re.match(r"^exp[\w]+$", exp_dir.name):
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
        if not exp_dir.is_dir() or not re.match(r"^exp[\w]+$", exp_dir.name):
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


def scan_som(results_dir: Path) -> dict:
    """
    Retorna {exp: {"n_regimes": [lista_de_k]}} para experimentos com artefatos
    SOM em results/predep_som/.  Suporta layout novo (exp*/n{k:02d}/) e legado
    (exp*/ flat, assume k=7).
    """
    index: dict = {}
    som_dir = results_dir / "predep_som"
    if not som_dir.exists():
        return index
    for exp_dir in sorted(som_dir.iterdir()):
        if not exp_dir.is_dir() or not re.match(r"^exp", exp_dir.name):
            continue
        n_dirs = sorted(exp_dir.glob("n[0-9][0-9]"),
                        key=lambda p: int(p.name[1:]))
        avail = [
            int(nd.name[1:]) for nd in n_dirs
            if (nd / "som_pixels.parquet").exists()
            and (nd / "som_meta.json").exists()
        ]
        if avail:
            index[exp_dir.name] = {"n_regimes": avail}
        elif ((exp_dir / "som_pixels.parquet").exists()
              and (exp_dir / "som_meta.json").exists()):
            index[exp_dir.name] = {"n_regimes": [7], "_flat": True}
    return index


def load_som(results_dir: Path, exp: str, n_regimes: int = 7):
    """Carrega (DataFrame de pixels, dict de meta) do SOM; LRU-1 cache
    por (exp, n_regimes).  Procura em n{k:02d}/ primeiro, depois layout legado."""
    global _som_cache, _som_cache_key
    key = (str(results_dir), exp, n_regimes)
    if key == _som_cache_key and key in _som_cache:
        return _som_cache[key]
    # Evict previous entry
    _som_cache.clear()
    base = results_dir / "predep_som" / exp
    sub = base / f"n{n_regimes:02d}"
    if sub.exists():
        pix, meta_path = sub / "som_pixels.parquet", sub / "som_meta.json"
    else:
        pix, meta_path = base / "som_pixels.parquet", base / "som_meta.json"
    if not pix.exists():
        return None
    df = pd.read_parquet(pix)
    meta = json.loads(meta_path.read_text())
    _som_cache[key] = (df, meta)
    _som_cache_key = key
    return _som_cache[key]


def compute_mov_stats(
    results_dir: Path, exp: str, basin: str, threshold: float,
) -> pd.DataFrame:
    """
    Para cada MoV disponível em (exp, basin), carrega o NetCDF e retorna
    um DataFrame com estatísticas por (MoV, Season):
      Max_R2, Media_R2, N_pixels_R2 (> threshold), Pct_pixels_R2,
      Melhor_lag, Max_alpha, N_pixels_alpha (> threshold), Pct_pixels_alpha,
      N_validos
    """
    index = _get_results_index(results_dir)
    seasons_order = ["DJF", "MAM", "JJA", "SON"]
    movs_items = sorted(index.get(exp, {}).items())
    rows = []
    # Columns needed for stats (avoid loading 'basin' when not filtering)
    _cols_brasil = ["latitude", "longitude", "lag", "r2", "alpha"]
    _cols_basin = ["basin", "latitude", "longitude", "lag", "r2", "alpha"]
    for i, (mov, basin_map) in enumerate(movs_items):
        if basin == "Brasil":
            src = next(iter(basin_map.values()), None)
        else:
            src = basin_map.get(basin)
        if src is None:
            continue

        if src.is_dir():
            # ── Parquet mode (formato longo: 1 linha por pixel×lag) ───────
            mov_dir = src
            for season in seasons_order:
                pq = mov_dir / f"{season}.parquet"
                if not pq.exists():
                    continue
                cols = _cols_brasil if basin == "Brasil" else _cols_basin
                df_s = pd.read_parquet(pq, columns=cols)
                df_b = df_s if basin == "Brasil" \
                    else df_s[df_s["basin"] == basin]
                del df_s
                df_b = df_b[df_b["lag"].isin(_ALLOWED_LAGS)]
                if df_b.empty:
                    del df_b
                    continue
                # max sobre lags por pixel
                g = df_b.groupby(["latitude", "longitude"])
                r2_pp = g["r2"].max()
                al_pp = g["alpha"].max()
                n_valid = int(len(r2_pp))
                r2_max = float(r2_pp.max())
                r2_mean = float(df_b["r2"].mean())
                n_above_r2 = int((r2_pp > threshold).sum())
                pct_r2 = round(100.0 * n_above_r2 / n_valid, 1)
                # lag com maior R² médio espacial
                best_lag = int(df_b.groupby("lag")["r2"].mean().idxmax())
                alpha_max = float(al_pp.max())
                alpha_mean = float(al_pp.mean())
                n_above_alpha = int((al_pp > threshold).sum())
                pct_alpha = round(100.0 * n_above_alpha / n_valid, 1)
                del df_b, g, r2_pp, al_pp
                rows.append({
                    "MoV": mov, "Season": season,
                    "Max_R2": round(r2_max, 4),
                    "Media_R2": round(r2_mean, 4),
                    "N_pixels_R2": n_above_r2,
                    "Pct_pixels_R2": pct_r2,
                    "Melhor_lag": best_lag,
                    "Max_alpha": round(alpha_max, 4),
                    "Media_alpha": round(alpha_mean, 4),
                    "N_pixels_alpha": n_above_alpha,
                    "Pct_pixels_alpha": pct_alpha,
                    "N_validos": n_valid,
                })
            gc.collect()
            continue

        # ── NC mode ───────────────────────────────────────────────────────
        nc_path = src
        ds = xr.open_dataset(nc_path)
        r2 = ds["r2"].values           # (season, lag, lat, lon)
        alpha = ds["alpha_core"].values
        ds_seasons = list(ds.coords["season"].values)
        lags_all = list(ds.coords["lag"].values)
        ds.close()
        allowed_idx = [i for i, x in enumerate(lags_all) if int(x) in _ALLOWED_LAGS]
        lags = [lags_all[i] for i in allowed_idx]
        r2 = r2[:, allowed_idx, :, :]
        alpha = alpha[:, allowed_idx, :, :]

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
            alpha_mean = float(np.nanmean(alpha_max_pixel[valid_mask]))
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
                "Media_alpha": round(alpha_mean, 4),
                "N_pixels_alpha": n_above_alpha,
                "Pct_pixels_alpha": pct_alpha,
                "N_validos": n_valid,
            })

        del r2, alpha
        gc.collect()

    return pd.DataFrame(rows)


# ── app ──────────────────────────────────────────────────────────────────────

app = Dash(__name__, title="PREDEP Viewer")
server = app.server


@server.route("/plots/<path:filepath>")
def serve_plot(filepath: str):
    return flask.send_from_directory(str(PLOTS_DIR), filepath)


@server.route("/gallery/<exp>/<mov>/<basin>")
def plot_gallery(exp: str, mov: str, basin: str):
    """Página HTML com todos os PNGs exportados para (exp, mov, basin)."""
    tipos = ("PREDEP", "REGRESSAO", "COMPARACAO")
    sections = []
    for tipo in tipos:
        if basin == "Brasil":
            basin_dir = PLOTS_DIR / exp / mov / tipo / "BRAZIL"
        else:
            basin_dir = PLOTS_DIR / exp / mov / tipo / "BACIAS" / basin
        if not basin_dir.is_dir():
            continue
        imgs = sorted(basin_dir.glob("*.png"))
        if not imgs:
            continue
        cards = "".join(
            f'<figure style="margin:0 0 24px 0">'
            f'<figcaption style="font:13px sans-serif;color:#555;'
            f'margin-bottom:6px">{p.name}</figcaption>'
            f'<img src="/plots/{p.relative_to(PLOTS_DIR).as_posix()}" '
            f'style="max-width:100%;border:1px solid #eee;border-radius:4px">'
            f'</figure>'
            for p in imgs
        )
        sections.append(
            f'<h2 style="font:600 18px sans-serif;color:#333;'
            f'border-bottom:2px solid #eee;padding-bottom:4px">{tipo}</h2>'
            f'{cards}'
        )

    title = f"{mov} · {_pretty_basin(basin)} · {exp}"
    if not sections:
        body = (
            '<p style="font:15px sans-serif;color:#999">'
            'Nenhum plot exportado encontrado para esta seleção.</p>'
        )
    else:
        body = "".join(sections)

    html_doc = (
        "<!doctype html><html lang='pt-br'><head><meta charset='utf-8'>"
        f"<title>Plots — {title}</title></head>"
        "<body style='max-width:1100px;margin:0 auto;padding:24px'>"
        f"<h1 style='font:600 22px sans-serif;color:#222'>Plots — {title}</h1>"
        f"{body}</body></html>"
    )
    return flask.Response(html_doc, mimetype="text/html")


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


# nome da bacia no Parquet → nome do diretório de shapefile (quando diferem)
_BASIN_DIR_ALIAS = {"parnaiba": "paranaiba"}


def _all_basins(clusters_dir: Path) -> list:
    """Nomes (padrão Parquet) de todas as bacias com shapefile."""
    if not clusters_dir.is_dir():
        return []
    dir_to_pq = {v: k for k, v in _BASIN_DIR_ALIAS.items()}
    out = []
    for d in sorted(clusters_dir.iterdir()):
        if d.is_dir() and d.name != "bacias":
            out.append(dir_to_pq.get(d.name, d.name))
    return out


def _load_basin_rings(clusters_dir: Path, basin: str, step: int = 1) -> list:
    """Returns list of (lons, lats) rings for the basin shapefile."""
    try:
        import shapefile as pyshp
    except ImportError:
        return []
    dir_name = _BASIN_DIR_ALIAS.get(basin, basin)
    basin_dir = clusters_dir / dir_name
    shp_files = list(basin_dir.glob("*.shp"))
    if not shp_files:
        return []
    sf = pyshp.Reader(str(shp_files[0]))
    rings = []
    for shape in sf.shapes():
        pts = shape.points
        if step > 1 and len(pts) > 2 * step:
            pts = pts[::step] + [pts[-1]]
        rings.append(([p[0] for p in pts], [p[1] for p in pts]))
    return rings


def _add_basin_rings(fig, overlay: list, available: set, ring_step: int, col: int):
    """Sobrepõe contornos de bacia (preenche cinza as ausentes). Reuso geral."""
    for b in overlay:
        rings_b = _load_basin_rings(CLUSTERS_DIR, b, step=ring_step)
        missing = b not in available
        for rl, ra in rings_b:
            if missing:
                fig.add_trace(go.Scatter(
                    x=rl, y=ra, mode="lines", fill="toself",
                    fillcolor="rgba(200,200,200,0.55)",
                    line=dict(color="#888", width=0.5),
                    showlegend=False, hoverinfo="skip",
                ), row=1, col=col)
            else:
                fig.add_trace(go.Scatter(
                    x=rl, y=ra, mode="lines",
                    line=dict(color="black", width=0.8),
                    showlegend=False, hoverinfo="skip",
                ), row=1, col=col)


_SEASONS_ALL = ["DJF", "MAM", "JJA", "SON"]
_ALLOWED_LAGS = [0, 1, 3, 6, 9, 12]


def _alpha_colorscale() -> list:
    """
    Colorscale discreta estilo _make_alpha_cmap (paper), usar com cmin=0,
    cmax=1: [0,0.1) cinza; [0.1,1.0] 9 degraus laranja pálido → intenso.
    """
    gray = (166, 166, 166)
    light = (255, 224, 179)
    dark = (255, 102, 0)
    bins = [gray]
    for i in range(9):
        t = i / 8
        bins.append(tuple(
            round(light[c] + t * (dark[c] - light[c])) for c in range(3)
        ))
    scale = []
    for k, (r, g, b) in enumerate(bins):
        col = f"rgb({r},{g},{b})"
        scale.append([k / 10, col])
        scale.append([(k + 1) / 10, col])
    return scale


_LAG_PALETTE = [
    "#2600CC", "#001AF2", "#0059E6", "#008CD9", "#00B8CC",
    "#00CCA6", "#00CC66", "#00C71A", "#40C700", "#8CCC00",
    "#CCCC00", "#FAB800", "#FF8500", "#FF4C00", "#FF0000", "#B20000",
]


def _lag_colorscale(lags: list) -> list:
    """
    Colorscale discreta para lags usando paleta fixa _LAG_PALETTE
    (violeta → azul → verde → amarelo → laranja → vermelho), com
    fronteiras nos midpoints dos lags. Usar com zmin=min(lags), zmax=max(lags).
    """
    n = len(lags)
    if n == 0:
        return [[0.0, _LAG_PALETTE[0]], [1.0, _LAG_PALETTE[0]]]
    positions = [i / max(n - 1, 1) for i in range(n)]
    colors = pcolors.sample_colorscale(_LAG_PALETTE, positions)
    arr = [float(x) for x in lags]
    lo, hi = arr[0], arr[-1]
    span = (hi - lo) or 1.0
    if n == 1:
        return [[0.0, colors[0]], [1.0, colors[0]]]
    bounds = [lo]
    bounds += [(arr[i] + arr[i + 1]) / 2 for i in range(n - 1)]
    bounds += [hi]
    scale = []
    for i, col in enumerate(colors):
        a = (bounds[i] - lo) / span
        b = (bounds[i + 1] - lo) / span
        scale.append([a, col])
        scale.append([b, col])
    scale[0][0] = 0.0
    scale[-1][0] = 1.0
    return scale


def _available_lags(src: Path, season: str) -> list:
    """Lista ordenada de lags disponíveis (Parquet ou NetCDF), restrita a _ALLOWED_LAGS."""
    if src.is_dir():
        seasons = [season] if season != "Todas" else _SEASONS_ALL
        for s in seasons:
            pq = src / f"{s}.parquet"
            if pq.exists():
                lags = pd.read_parquet(pq)["lag"].unique()
                return sorted(int(x) for x in lags if int(x) in _ALLOWED_LAGS)
        return []
    ds = xr.open_dataset(src)
    lags = [int(x) for x in ds.coords["lag"].values]
    ds.close()
    return sorted(x for x in lags if x in _ALLOWED_LAGS)


def _map_layers(src: Path, basin: str, season: str, lag):
    """
    Retorna dict com grades 2D para o mapa:
      lons, lats, r2, alpha, best_lag_r2, best_lag_alpha, season_used
    `lag` = "Máximo" (agrega sobre lags) ou um int (lag específico).
    best_lag_* só é preenchido no modo "Máximo".
    Memory-optimized: processes seasons incrementally instead of concat.
    """
    is_max = (lag == "Máximo" or lag is None)
    season_used = season if season != "Todas" else "Todas"

    if src.is_dir():
        seasons = [season] if season != "Todas" else _SEASONS_ALL
        # Column pruning: skip 'basin' when not filtering
        _cols = (["basin", "latitude", "longitude", "lag", "r2", "alpha"]
                 if basin != "Brasil"
                 else ["latitude", "longitude", "lag", "r2", "alpha"])

        # Incremental accumulation: load one season at a time, merge into
        # running max per (pixel, lag). Avoids concat of all seasons (~154MB
        # peak → ~35MB peak).
        acc = None
        for s in seasons:
            pq = src / f"{s}.parquet"
            if not pq.exists():
                continue
            df = pd.read_parquet(pq, columns=_cols)
            if basin != "Brasil":
                df = df[df["basin"] == basin]
                df = df.drop(columns=["basin"], errors="ignore")
            df = df[df["lag"].isin(_ALLOWED_LAGS)]
            if df.empty:
                del df
                continue
            # aggregate this season by (pixel, lag) → max
            df = df.groupby(
                ["latitude", "longitude", "lag"], as_index=False
            ).agg(r2=("r2", "max"), alpha=("alpha", "max"))
            if acc is None:
                acc = df
            else:
                acc = pd.concat([acc, df], ignore_index=True)
                acc = acc.groupby(
                    ["latitude", "longitude", "lag"], as_index=False
                ).agg(r2=("r2", "max"), alpha=("alpha", "max"))
            del df
        gc.collect()

        if acc is None or acc.empty:
            return None
        df = acc
        del acc

        if is_max:
            idx_r2 = df.groupby(["latitude", "longitude"])["r2"].idxmax()
            idx_al = df.groupby(["latitude", "longitude"])["alpha"].idxmax()
            rows_r2 = df.loc[idx_r2.values]
            rows_al = df.loc[idx_al.values]
            agg = pd.DataFrame({
                "latitude": rows_r2["latitude"].values,
                "longitude": rows_r2["longitude"].values,
                "r2": rows_r2["r2"].values,
                "alpha": rows_al["alpha"].values,
                "r2_albest": rows_al["r2"].values,
                "best_lag_r2": rows_r2["lag"].values,
                "best_lag_alpha": rows_al["lag"].values,
            })
        else:
            agg = df[df["lag"] == int(lag)].copy()
            if agg.empty:
                return None
            agg["r2_albest"] = agg["r2"].values
            agg["best_lag_r2"] = np.nan
            agg["best_lag_alpha"] = np.nan
        del df

        def _piv(col):
            return agg.pivot(
                index="latitude", columns="longitude", values=col
            ).sort_index()

        p_r2 = _piv("r2")
        result = {
            "lons": p_r2.columns.values,
            "lats": p_r2.index.values,
            "r2": p_r2.values,
            "alpha": _piv("alpha").values,
            "r2_albest": _piv("r2_albest").values,
            "best_lag_r2": _piv("best_lag_r2").values,
            "best_lag_alpha": _piv("best_lag_alpha").values,
            "season_used": season_used,
        }
        del agg
        gc.collect()
        return result

    # ── NetCDF ────────────────────────────────────────────────────────────
    ds = xr.open_dataset(src)
    lats = ds.coords["latitude"].values
    lons = ds.coords["longitude"].values
    seasons_nc = list(ds.coords["season"].values)
    lags_all_nc = [int(x) for x in ds.coords["lag"].values]
    r2_arr = ds["r2"].values        # (season, lag, lat, lon)
    alpha_arr = ds["alpha_core"].values
    ds.close()
    allowed_idx_nc = [i for i, x in enumerate(lags_all_nc) if x in _ALLOWED_LAGS]
    lags_nc = [lags_all_nc[i] for i in allowed_idx_nc]
    r2_arr = r2_arr[:, allowed_idx_nc, :, :]
    alpha_arr = alpha_arr[:, allowed_idx_nc, :, :]

    if season != "Todas" and season in seasons_nc:
        si = [seasons_nc.index(season)]
    else:
        si = list(range(len(seasons_nc)))
    with np.errstate(all="ignore"):
        r2_sl = np.nanmax(r2_arr[si], axis=0)      # (lag, lat, lon)
        al_sl = np.nanmax(alpha_arr[si], axis=0)

    lag_arr = np.array(lags_nc)
    out = {"lons": lons, "lats": lats, "season_used": season_used,
           "best_lag_r2": None, "best_lag_alpha": None}
    with np.errstate(all="ignore"):
        if is_max:
            out["r2"] = np.nanmax(r2_sl, axis=0)
            out["alpha"] = np.nanmax(al_sl, axis=0)
            valid = ~np.all(np.isnan(r2_sl), axis=0)
            blr = np.full(r2_sl.shape[1:], np.nan)
            bla = np.full(al_sl.shape[1:], np.nan)
            r2_alb = np.full(r2_sl.shape[1:], np.nan)
            if valid.any():
                ir2 = np.nanargmax(r2_sl[:, valid], axis=0)
                ial = np.nanargmax(al_sl[:, valid], axis=0)
                blr[valid] = lag_arr[ir2]
                bla[valid] = lag_arr[ial]
                # R² no lag ótimo do α (mesma célula)
                r2_valid = r2_sl[:, valid]
                r2_alb[valid] = r2_valid[ial, np.arange(ial.size)]
            out["best_lag_r2"] = blr
            out["best_lag_alpha"] = bla
            out["r2_albest"] = r2_alb
        else:
            li = lags_nc.index(int(lag))
            out["r2"] = r2_sl[li]
            out["alpha"] = al_sl[li]
            out["r2_albest"] = r2_sl[li]
    return out


def _mov_colorscale(movs: list) -> list:
    """Discrete colorscale for MoV categorical map, grouped by phenomenon."""
    n = len(movs)
    if n == 0:
        return [[0, "#ccc"], [1, "#ccc"]]
        
    groups = {
        "Blues": ["NIN12", "NIN03", "NIN034", "NIN04", "ONI", "SOI"],
        "Reds": ["AMO", "NAO", "TNA", "TSA", "ATL3", "SAODI", "SASDI"],
        "Greens": ["PDO", "PNA", "PSA1", "PSA2"],
        "Purples": ["IOD", "IOSD"],
        "Oranges": ["AO", "AAO", "QBO"]
    }
    
    # Pre-compute colors for each group using a continuous scale sampled at distinct points
    group_colors = {}
    for scale_name, members in groups.items():
        if not members:
            continue
        # Evitamos os extremos muito claros (0.0) ou muito escuros (1.0)
        # Se for só 1 elemento, pegamos o meio (0.6)
        if len(members) == 1:
            positions = [0.6]
        else:
            positions = [0.3 + 0.6 * (i / (len(members) - 1)) for i in range(len(members))]
        colors = pcolors.sample_colorscale(scale_name, positions)
        for mov_name, color in zip(members, colors):
            group_colors[mov_name] = color
            
    scale = []
    for i, mov in enumerate(movs):
        # Fallback to dark gray se o MoV não estiver nos grupos conhecidos
        col = group_colors.get(mov, "#333333")
        scale.append([i / n, col])
        scale.append([(i + 1) / n, col])
    return scale


def _compute_overview_layers(
    results_dir: Path, exp: str, basin: str, season: str,
    include_basin_names: bool = False,
) -> dict | None:
    """
    Computes an overview across ALL MoVs for a given (exp, basin, season).
    For each pixel, determines:
      - max R² across all MoVs, and which MoV/lag produced it
      - max PREDEP across all MoVs, and which MoV/lag produced it
      - difference (max PREDEP − max R²)

    Memory-efficient: processes one MoV at a time (~28MB per parquet load),
    updates running-best grids, then releases.

    Returns dict with 2D arrays:
      lons, lats, r2, alpha, diff,
      mov_r2 (int index), mov_alpha (int index),
      lag_r2, lag_alpha, mov_names (list of str)
    """
    index = _get_results_index(results_dir)
    movs_items = sorted(index.get(exp, {}).items())
    if not movs_items:
        return None

    seasons = [season] if season != "Todas" else _SEASONS_ALL
    _cols_brasil = ["latitude", "longitude", "lag", "r2", "alpha"]
    _cols_basin = ["basin", "latitude", "longitude", "lag", "r2", "alpha"]

    # Running-best DataFrames: one row per pixel, tracking max across MoVs
    # Columns: latitude, longitude, best_r2, best_r2_mov, best_r2_lag,
    #          best_alpha, best_alpha_mov, best_alpha_lag
    best = None
    mov_names = []

    for mov_idx, (mov, basin_map) in enumerate(movs_items):
        if basin == "Brasil":
            src = next(iter(basin_map.values()), None)
        else:
            src = basin_map.get(basin)
        if src is None:
            continue
        if not src.is_dir():
            # NC mode not supported for overview (all data is Parquet)
            continue

        mov_names.append(mov)
        mi = len(mov_names) - 1  # index into mov_names

        # Load this MoV's data incrementally per season
        acc = None
        cols = _cols_brasil if basin == "Brasil" else _cols_basin
        for s in seasons:
            pq = src / f"{s}.parquet"
            if not pq.exists():
                continue
            df = pd.read_parquet(pq, columns=cols)
            if basin != "Brasil":
                df = df[df["basin"] == basin]
                df = df.drop(columns=["basin"], errors="ignore")
            df = df[df["lag"].isin(_ALLOWED_LAGS)]
            if df.empty:
                del df
                continue
            # Aggregate across seasons per (pixel, lag) → max
            df = df.groupby(
                ["latitude", "longitude", "lag"], as_index=False
            ).agg(r2=("r2", "max"), alpha=("alpha", "max"))
            if acc is None:
                acc = df
            else:
                acc = pd.concat([acc, df], ignore_index=True)
                acc = acc.groupby(
                    ["latitude", "longitude", "lag"], as_index=False
                ).agg(r2=("r2", "max"), alpha=("alpha", "max"))
            del df

        if acc is None or acc.empty:
            gc.collect()
            continue

        # For this MoV: best R² per pixel (max over lags), with lag info
        g = acc.groupby(["latitude", "longitude"])
        idx_r2 = g["r2"].idxmax()
        idx_al = g["alpha"].idxmax()
        rows_r2 = acc.loc[idx_r2.values]
        rows_al = acc.loc[idx_al.values]

        mov_best = pd.DataFrame({
            "latitude": rows_r2["latitude"].values,
            "longitude": rows_r2["longitude"].values,
            "r2": rows_r2["r2"].values,
            "r2_lag": rows_r2["lag"].values,
            "alpha": rows_al["alpha"].values,
            "alpha_lag": rows_al["lag"].values,
        })
        del acc, g, idx_r2, idx_al, rows_r2, rows_al

        if best is None:
            best = mov_best.copy()
            best["r2_mov"] = mi
            best["alpha_mov"] = mi
        else:
            # Merge on pixel coordinates
            merged = best.merge(
                mov_best, on=["latitude", "longitude"],
                how="outer", suffixes=("", "_new"),
            )
            # Update best R²
            better_r2 = merged["r2_new"].fillna(-1) > merged["r2"].fillna(-1)
            merged.loc[better_r2, "r2"] = merged.loc[better_r2, "r2_new"]
            merged.loc[better_r2, "r2_lag"] = merged.loc[better_r2, "r2_lag_new"]
            merged.loc[better_r2, "r2_mov"] = mi
            # Update best alpha
            better_al = merged["alpha_new"].fillna(-1) > merged["alpha"].fillna(-1)
            merged.loc[better_al, "alpha"] = merged.loc[better_al, "alpha_new"]
            merged.loc[better_al, "alpha_lag"] = merged.loc[better_al, "alpha_lag_new"]
            merged.loc[better_al, "alpha_mov"] = mi
            # Clean up
            best = merged[["latitude", "longitude",
                           "r2", "r2_lag", "r2_mov",
                           "alpha", "alpha_lag", "alpha_mov"]].copy()
            del merged

        del mov_best
        gc.collect()

    if best is None or best.empty:
        return None

    # Compute difference (max PREDEP − max R²)
    best["diff"] = best["alpha"] - best["r2"]

    # Optional: attach per-pixel basin/bacia name (static geography, so a
    # single lightweight read from the first available MoV/season suffices)
    if include_basin_names and basin == "Brasil":
        first_src = None
        for _mov, basin_map in movs_items:
            cand = next(iter(basin_map.values()), None)
            if cand is not None and cand.is_dir():
                first_src = cand
                break
        if first_src is not None:
            for s in seasons:
                pq = first_src / f"{s}.parquet"
                if not pq.exists():
                    continue
                bdf = pd.read_parquet(
                    pq, columns=["latitude", "longitude", "basin"]
                )
                bdf = bdf.drop_duplicates(subset=["latitude", "longitude"])
                best = best.merge(
                    bdf, on=["latitude", "longitude"], how="left"
                )
                break

    # Pivot to 2D grids
    def _piv(col):
        return best.pivot(
            index="latitude", columns="longitude", values=col
        ).sort_index()

    p_r2 = _piv("r2")
    result = {
        "lons": p_r2.columns.values,
        "lats": p_r2.index.values,
        "r2": p_r2.values,
        "alpha": _piv("alpha").values,
        "diff": _piv("diff").values,
        "mov_r2": _piv("r2_mov").values,
        "mov_alpha": _piv("alpha_mov").values,
        "lag_r2": _piv("r2_lag").values,
        "lag_alpha": _piv("alpha_lag").values,
        "mov_names": mov_names,
        "basin": _piv("basin").values if "basin" in best.columns else None,
    }
    del best
    gc.collect()
    return result


# Lazy cache for the fixed-config "Overview (alternativo)" tab — the tab has
# no filters (always exp_brasil / Brasil / Todas), so compute+downsample once
# per process and reuse for every dropdown change in either column.
_brasil_overview_cache: dict | None = None


def _get_brasil_overview_layers() -> dict | None:
    """Cached, downsampled Brasil-wide overview layers with basin names,
    fixed to (_FIXED_EXP, 'Brasil', 'Todas'). Computed once per process."""
    global _brasil_overview_cache
    if _brasil_overview_cache is not None:
        return _brasil_overview_cache
    layers = _compute_overview_layers(
        RESULTS_DIR, _FIXED_EXP, "Brasil", "Todas", include_basin_names=True,
    )
    if not layers:
        return None
    lons, lats = layers["lons"], layers["lats"]
    if len(lats) > 50 and len(lons) > 50:
        lons, lats = lons[::2], lats[::2]
        for k in ["r2", "alpha", "diff", "mov_r2", "mov_alpha",
                  "lag_r2", "lag_alpha", "basin"]:
            v = layers.get(k)
            if v is not None and hasattr(v, "__getitem__"):
                layers[k] = v[::2, ::2]
        layers["lons"], layers["lats"] = lons, lats
    _brasil_overview_cache = layers
    return _brasil_overview_cache


_brasil_perlags_cache: dict = {}  # keyed by season string


def _compute_brasil_per_lag(season: str) -> dict | None:
    """Para cada lag permitido, calcula max R² e max PREDEP sobre todos os MoVs × seasons.
    Retorna {lag_int: {"lons", "lats", "r2", "alpha"}}."""
    index = _get_results_index(RESULTS_DIR)
    movs_items = sorted(index.get(_FIXED_EXP, {}).items())
    if not movs_items:
        return None

    seasons = [season] if season != "Todas" else _SEASONS_ALL
    best = None  # DataFrame: latitude, longitude, lag, r2, alpha

    for mov, basin_map in movs_items:
        src = next(iter(basin_map.values()), None)
        if src is None or not src.is_dir():
            continue
        acc = None
        for s in seasons:
            pq = src / f"{s}.parquet"
            if not pq.exists():
                continue
            df = pd.read_parquet(
                pq, columns=["latitude", "longitude", "lag", "r2", "alpha"]
            )
            df = df[df["lag"].isin(_ALLOWED_LAGS)]
            if df.empty:
                del df
                continue
            df = df.groupby(
                ["latitude", "longitude", "lag"], as_index=False
            ).agg(r2=("r2", "max"), alpha=("alpha", "max"))
            if acc is None:
                acc = df
            else:
                acc = (
                    pd.concat([acc, df], ignore_index=True)
                    .groupby(["latitude", "longitude", "lag"], as_index=False)
                    .agg(r2=("r2", "max"), alpha=("alpha", "max"))
                )
            del df
        if acc is None:
            gc.collect()
            continue
        if best is None:
            best = acc
        else:
            best = (
                pd.concat([best, acc], ignore_index=True)
                .groupby(["latitude", "longitude", "lag"], as_index=False)
                .agg(r2=("r2", "max"), alpha=("alpha", "max"))
            )
        del acc
        gc.collect()

    if best is None or best.empty:
        return None

    result = {}
    for lag in sorted(best["lag"].unique()):
        sub = best[best["lag"] == lag]
        if sub.empty:
            continue
        p_r2 = sub.pivot(
            index="latitude", columns="longitude", values="r2"
        ).sort_index()
        p_al = sub.pivot(
            index="latitude", columns="longitude", values="alpha"
        ).sort_index()
        lons_p = p_r2.columns.values
        lats_p = p_r2.index.values
        r2_v = p_r2.values
        al_v = p_al.values
        if len(lats_p) > 50:
            lons_p = lons_p[::2]
            lats_p = lats_p[::2]
            r2_v = r2_v[::2, ::2]
            al_v = al_v[::2, ::2]
        result[int(lag)] = {
            "lons": lons_p, "lats": lats_p, "r2": r2_v, "alpha": al_v,
        }
    return result


def _get_brasil_per_lag(season: str = "Todas") -> dict | None:
    global _brasil_perlags_cache
    if season in _brasil_perlags_cache:
        return _brasil_perlags_cache[season]
    result = _compute_brasil_per_lag(season)
    if result is not None:
        _brasil_perlags_cache[season] = result
    return result


def _compute_lag0_layers(
    results_dir: Path, exp: str, basin: str, season: str,
) -> dict | None:
    """
    For each pixel, finds the MoV with max R² and max PREDEP restricted to lag=0.
    Returns dict with 2D arrays: lons, lats, r2, alpha, mov_r2, mov_alpha, mov_names.
    """
    index = _get_results_index(results_dir)
    movs_items = sorted(index.get(exp, {}).items())
    if not movs_items:
        return None

    seasons = [season] if season != "Todas" else _SEASONS_ALL
    _cols_brasil = ["latitude", "longitude", "lag", "r2", "alpha"]
    _cols_basin = ["basin", "latitude", "longitude", "lag", "r2", "alpha"]

    best = None
    mov_names = []

    for _mov_idx, (mov, basin_map) in enumerate(movs_items):
        if basin == "Brasil":
            src = next(iter(basin_map.values()), None)
        else:
            src = basin_map.get(basin)
        if src is None or not src.is_dir():
            continue

        cols = _cols_brasil if basin == "Brasil" else _cols_basin
        acc = None
        for s in seasons:
            pq = src / f"{s}.parquet"
            if not pq.exists():
                continue
            df = pd.read_parquet(pq, columns=cols)
            df = df[df["lag"] == 0]
            if df.empty:
                del df
                continue
            if basin != "Brasil":
                df = df[df["basin"] == basin]
                df = df.drop(columns=["basin"], errors="ignore")
            if df.empty:
                del df
                continue
            df = df.drop(columns=["lag"])
            df = df.groupby(["latitude", "longitude"], as_index=False).agg(
                r2=("r2", "max"), alpha=("alpha", "max")
            )
            if acc is None:
                acc = df
            else:
                acc = pd.concat([acc, df], ignore_index=True)
                acc = acc.groupby(["latitude", "longitude"], as_index=False).agg(
                    r2=("r2", "max"), alpha=("alpha", "max")
                )
            del df

        if acc is None or acc.empty:
            gc.collect()
            continue

        mov_names.append(mov)
        mi = len(mov_names) - 1

        if best is None:
            best = acc.copy()
            best["r2_mov"] = mi
            best["alpha_mov"] = mi
        else:
            merged = best.merge(
                acc, on=["latitude", "longitude"],
                how="outer", suffixes=("", "_new"),
            )
            better_r2 = merged["r2_new"].fillna(-1) > merged["r2"].fillna(-1)
            merged.loc[better_r2, "r2"] = merged.loc[better_r2, "r2_new"]
            merged.loc[better_r2, "r2_mov"] = mi
            better_al = merged["alpha_new"].fillna(-1) > merged["alpha"].fillna(-1)
            merged.loc[better_al, "alpha"] = merged.loc[better_al, "alpha_new"]
            merged.loc[better_al, "alpha_mov"] = mi
            best = merged[["latitude", "longitude",
                           "r2", "r2_mov", "alpha", "alpha_mov"]].copy()
            del merged

        del acc
        gc.collect()

    if best is None or best.empty:
        return None

    def _piv(col):
        return best.pivot(
            index="latitude", columns="longitude", values=col
        ).sort_index()

    p_r2 = _piv("r2")
    result = {
        "lons": p_r2.columns.values,
        "lats": p_r2.index.values,
        "r2":       p_r2.values,
        "alpha":    _piv("alpha").values,
        "mov_r2":   _piv("r2_mov").values,
        "mov_alpha": _piv("alpha_mov").values,
        "mov_names": mov_names,
    }
    del best
    gc.collect()
    return result


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

    # Mostra apenas exp_brasil; oculta os experimentos individuais por MoV
    VISIBLE = {"exp_brasil"}

    opts = []
    for exp in sorted(results_index.keys()):
        if exp not in VISIBLE:
            continue
        movs = sorted(results_index[exp].keys())
        basins = sorted({
            b for mv in results_index[exp].values() for b in mv
        })
        basin_labels = [_pretty_basin(b) for b in basins]
        label = (
            f"MOVs: {_fmt(movs)}"
            f"  |  Bacias: {_fmt(basin_labels)}"
        )
        opts.append({"label": label, "value": exp})
    return opts


_SOM_VIEW_OPTS = [
    {"label": "Regimes", "value": "regime"},
    {"label": "Atipicidade", "value": "atypicality"},
    {"label": "Fronteiras", "value": "boundary"},
    {"label": "Component planes", "value": "component"},
]


def _hi(text: str) -> html.Span:
    """Ícone ⓘ com tooltip via title=."""
    return html.Span(" ⓘ", title=text,
                     style={"cursor": "help", "color": "#888", "fontSize": "13px"})


def _som_tab_layout(first_som) -> html.Div:
    """Controles + área da aba 'SOM'. Parâmetros fixos via busca: k=7, todos MoVs."""
    if not first_som:
        return html.P(
            "Nenhum artefato SOM encontrado. Gere com "
            "`modal run src/modal/som_insights_modal.py --exp-name exp_brasil --parquets` "
            "(repo irc_predep_bootstrap) para popular results/predep_som/.",
            style={"color": "#999", "fontStyle": "italic", "padding": "16px"},
        )
    return html.Div([
        html.Div([
            html.Div([
                html.Label(
                    ["SOM treinado em", _hi(
                        "Escolhe qual SOM carregar. 'Geral' usa todas as estações "
                        "(DJF+MAM+JJA+SON) simultaneamente. As opções sazonais "
                        "treinam um SOM independente usando apenas os lags e MoVs "
                        "daquela estação — revelam padrões de previsibilidade "
                        "exclusivos de cada trimestre climático."
                    )],
                    style={"fontWeight": "500"},
                ),
                dcc.RadioItems(
                    id="ri-som-training-season",
                    options=[
                        {"label": "Geral", "value": "geral"},
                        {"label": "DJF", "value": "DJF"},
                        {"label": "MAM", "value": "MAM"},
                        {"label": "JJA", "value": "JJA"},
                        {"label": "SON", "value": "SON"},
                    ],
                    value="geral", inline=True,
                    labelStyle={"marginRight": "12px"},
                ),
            ], style=_RADIO_DIV),
            html.Div([
                html.Label(
                    ["Visão SOM", _hi(
                        "Tipo de mapa a exibir. O Self-Organizing Map (MiniSom) é "
                        "treinado sobre α (PREDEP, ∈ [0,1]) — índice de "
                        "previsibilidade de precipitação superior ao R². 1 amostra = "
                        "1 pixel. O SOM preserva topologia: pixels semelhantes caem "
                        "em neurônios vizinhos. Os regimes são agrupamentos de "
                        "neurônios via KMeans."
                    )],
                    style={"fontWeight": "500"},
                ),
                dcc.RadioItems(
                    id="ri-som-view", options=_SOM_VIEW_OPTS,
                    value="regime", inline=True,
                    labelStyle={"marginRight": "12px"},
                ),
            ], style=_RADIO_DIV),
        ], style={**_ROW, "marginBottom": "12px", "flexWrap": "wrap", "gap": "16px"}),
        html.Div(id="som-content"),
    ])


_FIXED_EXP = "exp_brasil"


def _build_layout(results_index: dict) -> html.Div:
    # Filtra para apenas o experimento com todos os MoVs e Brasil completo
    if _FIXED_EXP in results_index:
        results_index = {_FIXED_EXP: results_index[_FIXED_EXP]}
    exps_r = sorted(results_index.keys())
    first_exp_r = exps_r[0] if exps_r else None
    first_basins = sorted({
        basin
        for mov_map in results_index.get(first_exp_r or "", {}).values()
        for basin in mov_map
    })
    first_basin = first_basins[0] if first_basins else None

    som_index = scan_som(RESULTS_DIR)
    first_som = _FIXED_EXP if _FIXED_EXP in som_index else (
        sorted(som_index.keys())[0] if som_index else None
    )

    explore_controls = html.Div([
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
                html.Label("Bacias", style={"fontWeight": "500"}),
                dcc.Dropdown(
                    id="dd-cluster-explore",
                    options=_basin_opts(first_basins),
                    value=first_basin,
                    clearable=False,
                ),
            ], style=_DD),
            html.Div([
                html.Label("Season", style={"fontWeight": "500"}),
                dcc.RadioItems(
                    id="ri-season-explore",
                    options=[
                        {"label": "Todas", "value": "Todas"},
                        {"label": "DJF", "value": "DJF"},
                        {"label": "MAM", "value": "MAM"},
                        {"label": "JJA", "value": "JJA"},
                        {"label": "SON", "value": "SON"},
                    ],
                    value="DJF",
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
                html.Label("MoVs", style={"fontWeight": "500"}),
                dcc.Dropdown(
                    id="dd-mov-map",
                    options=[],
                    value=None,
                    clearable=False,
                    placeholder="selecione um MoV...",
                ),
            ], style=_DD),
            html.Div([
                html.Label("Lag", style={"fontWeight": "500"}),
                dcc.Dropdown(
                    id="dd-lag-map",
                    options=[{"label": "Máximo", "value": "Máximo"}],
                    value="Máximo",
                    clearable=False,
                ),
            ], style={**_DD, "minWidth": "120px"}),
            html.Div([
                html.Label("Visão", style={"fontWeight": "500"}),
                dcc.RadioItems(
                    id="ri-map-view",
                    options=[
                        {"label": "R² + PREDEP", "value": "ralpha"},
                        {"label": "Lag ótimo", "value": "lag"},
                        {"label": "Diferença (PREDEP−R²)", "value": "diff"},
                    ],
                    value="ralpha",
                    inline=True,
                    labelStyle={"marginRight": "12px"},
                ),
            ], style=_RADIO_DIV),
        ], style={**_ROW, "marginBottom": "20px"})

    overview_controls = html.Div([
        html.Div([
            html.Label("Experimento", style={"fontWeight": "500"}),
            dcc.Dropdown(
                id="dd-exp-overview",
                options=_exp_opts(results_index),
                value=first_exp_r,
                clearable=False,
            ),
        ], style={**_DD, "minWidth": "300px", "flex": "3"}),
        html.Div([
            html.Label("Bacias", style={"fontWeight": "500"}),
            dcc.Dropdown(
                id="dd-cluster-overview",
                options=_basin_opts(first_basins),
                value=first_basin,
                clearable=False,
            ),
        ], style=_DD),
        html.Div([
            html.Label("Season", style={"fontWeight": "500"}),
            dcc.RadioItems(
                id="ri-season-overview",
                options=[
                    {"label": "Todas", "value": "Todas"},
                    {"label": "DJF", "value": "DJF"},
                    {"label": "MAM", "value": "MAM"},
                    {"label": "JJA", "value": "JJA"},
                    {"label": "SON", "value": "SON"},
                ],
                value="Todas",
                inline=True,
                labelStyle={"marginRight": "10px"},
            ),
        ], style=_RADIO_DIV),
        html.Div([
            html.Label("Threshold",
                       style={"fontWeight": "500", "marginBottom": "4px"}),
            dcc.Slider(
                id="sl-threshold-overview",
                min=0.0, max=1.0, step=0.05, value=0.0,
                marks={v: f"{v:.2g}" for v in [0, 0.2, 0.4, 0.6, 0.8, 1.0]},
                tooltip={"placement": "bottom", "always_visible": True},
            ),
        ], style={"flex": "2", "minWidth": "300px", "paddingTop": "4px"}),
    ], style={**_ROW, "marginBottom": "20px", "flexWrap": "wrap", "alignItems": "flex-end"})

    overview_alt_body = html.Div([
        html.Div([
            html.Label("Estação:", style={"fontWeight": "500", "marginRight": "8px"}),
            dcc.RadioItems(
                id="ri-season-overview-alt",
                options=[
                    {"label": "Todas", "value": "Todas"},
                    {"label": "DJF", "value": "DJF"},
                    {"label": "MAM", "value": "MAM"},
                    {"label": "JJA", "value": "JJA"},
                    {"label": "SON", "value": "SON"},
                ],
                value="Todas",
                inline=True,
                labelStyle={"marginRight": "10px"},
            ),
        ], style=_RADIO_DIV),
        html.Div([
            html.Label("Threshold",
                       style={"fontWeight": "500", "marginBottom": "4px"}),
            dcc.Slider(
                id="sl-threshold-overview-alt",
                min=0.0, max=1.0, step=0.05, value=0.0,
                marks={v: f"{v:.2g}" for v in [0, 0.2, 0.4, 0.6, 0.8, 1.0]},
                tooltip={"placement": "bottom", "always_visible": True},
            ),
        ], style={"marginBottom": "16px", "padding": "0 8px"}),
        dcc.Loading(
            html.Div(id="overview-alt-perlags-content"),
            type="circle",
            color="#e07b39",
        ),
    ])

    lag0_controls = html.Div([
        html.Div([
            html.Label("Experimento", style={"fontWeight": "500"}),
            dcc.Dropdown(
                id="dd-exp-lag0",
                options=_exp_opts(results_index),
                value=first_exp_r,
                clearable=False,
            ),
        ], style={**_DD, "minWidth": "300px", "flex": "3"}),
        html.Div([
            html.Label("Bacias", style={"fontWeight": "500"}),
            dcc.Dropdown(
                id="dd-cluster-lag0",
                options=_basin_opts(first_basins),
                value=first_basin,
                clearable=False,
            ),
        ], style=_DD),
        html.Div([
            html.Label("Season", style={"fontWeight": "500"}),
            dcc.RadioItems(
                id="ri-season-lag0",
                options=[
                    {"label": "Todas", "value": "Todas"},
                    {"label": "DJF", "value": "DJF"},
                    {"label": "MAM", "value": "MAM"},
                    {"label": "JJA", "value": "JJA"},
                    {"label": "SON", "value": "SON"},
                ],
                value="Todas",
                inline=True,
                labelStyle={"marginRight": "10px"},
            ),
        ], style=_RADIO_DIV),
    ], style={**_ROW, "marginBottom": "20px"})

    destaque_controls = html.Div([
        html.Div([
            html.Label("Experimento", style={"fontWeight": "500"}),
            dcc.Dropdown(
                id="dd-exp-destaque",
                options=_exp_opts(results_index),
                value=first_exp_r,
                clearable=False,
            ),
        ], style={**_DD, "minWidth": "300px", "flex": "3"}),
        html.Div([
            html.Label("Bacias", style={"fontWeight": "500"}),
            dcc.Dropdown(
                id="dd-cluster-destaque",
                options=_basin_opts(first_basins),
                value=first_basin,
                clearable=False,
            ),
        ], style=_DD),
        html.Div([
            html.Label("Season", style={"fontWeight": "500"}),
            dcc.RadioItems(
                id="ri-season-destaque",
                options=[
                    {"label": "Todas", "value": "Todas"},
                    {"label": "DJF", "value": "DJF"},
                    {"label": "MAM", "value": "MAM"},
                    {"label": "JJA", "value": "JJA"},
                    {"label": "SON", "value": "SON"},
                ],
                value="Todas",
                inline=True,
                labelStyle={"marginRight": "10px"},
            ),
        ], style=_RADIO_DIV),
        html.Div([
            html.Label("Melhor MoV — R²", style={"fontWeight": "500"}),
            dcc.Dropdown(
                id="dd-mov-destaque-r2",
                options=[],
                value=None,
                clearable=False,
                placeholder="calculando…",
            ),
        ], style={**_DD, "minWidth": "220px", "flex": "2"}),
        html.Div([
            html.Label("Melhor MoV — PREDEP", style={"fontWeight": "500"}),
            dcc.Dropdown(
                id="dd-mov-destaque-alpha",
                options=[],
                value=None,
                clearable=False,
                placeholder="calculando…",
            ),
        ], style={**_DD, "minWidth": "220px", "flex": "2"}),
    ], style={**_ROW, "marginBottom": "20px"})

    return html.Div([
        html.H2(
            "PREDEP Visualization",
            style={"marginBottom": "16px", "fontWeight": "600"},
        ),
        dcc.Tabs([
            dcc.Tab(label="Overview", children=[
                overview_controls,
                dcc.Loading(
                    html.Div(id="overview-content"),
                    type="circle",
                    color="#e07b39",
                ),
            ]),
            dcc.Tab(label="Exploração", children=[
                explore_controls,
                dcc.Store(id="stats-store"),
                dcc.Loading(
                    html.Div(id="explore-content"),
                    type="circle",
                    color="#e07b39",
                ),
            ]),
            dcc.Tab(label="Lag 0", children=[
                lag0_controls,
                dcc.Loading(
                    html.Div(id="lag0-content"),
                    type="circle",
                    color="#e07b39",
                ),
            ]),
            dcc.Tab(label="Lag's", children=[
                overview_alt_body,
            ]),
            dcc.Tab(label="MoV Vencedor", children=[
                destaque_controls,
                dcc.Loading(
                    html.Div(id="destaque-content"),
                    type="circle",
                    color="#e07b39",
                ),
            ]),
            dcc.Tab(label="SOM", children=[
                html.Div(
                    _som_tab_layout(first_som),
                    style={"marginTop": "16px"},
                ),
            ]),
        ]),
    ], style={
        "fontFamily": "sans-serif",
        "padding": "24px",
        "maxWidth": "1600px",
        "margin": "0 auto",
    })


# Layout populado já no import (necessário para gunicorn/`app:server`,
# onde main() nunca executa). main() apenas re-popula se rodar via CLI.
_valid_brasil = compute_valid_brasil(RESULTS_DIR)
app.layout = _build_layout(_get_results_index(RESULTS_DIR))


# ── callbacks ────────────────────────────────────────────────────────────────

@app.callback(
    Output("dd-cluster-explore", "options"),
    Output("dd-cluster-explore", "value"),
    Input("dd-exp-explore", "value"),
)
def cb_clusters_explore(exp: str):
    if not exp:
        return [], None
    index = _get_results_index(RESULTS_DIR)
    basins = sorted({
        basin
        for mov_map in index.get(exp, {}).values()
        for basin in mov_map
    })
    if not basins:
        return [], None
    opts = ([{"label": "Brasil", "value": "Brasil"}]
            + _basin_opts(basins))
    return opts, "Brasil"


@app.callback(
    Output("dd-mov-map", "options"),
    Output("dd-mov-map", "value"),
    Input("dd-exp-explore",     "value"),
    Input("dd-cluster-explore", "value"),
)
def cb_mov_map_opts(exp: str, basin: str):
    if not exp or not basin:
        return [], None
    index = _get_results_index(RESULTS_DIR)
    movs = sorted(
        mov for mov, bmap in index.get(exp, {}).items()
        if basin == "Brasil" or basin in bmap
    )
    return _opts(movs), (movs[0] if movs else None)


@app.callback(
    Output("dd-lag-map", "options"),
    Output("dd-lag-map", "value"),
    Input("dd-exp-explore",     "value"),
    Input("dd-cluster-explore", "value"),
    Input("dd-mov-map",         "value"),
    Input("ri-season-explore",  "value"),
)
def cb_lag_map_opts(exp: str, basin: str, mov_map: str, season: str):
    base = [{"label": "Máximo", "value": "Máximo"}]
    if not (exp and basin and mov_map):
        return base, "Máximo"
    index = _get_results_index(RESULTS_DIR)
    bmap = index.get(exp, {}).get(mov_map, {})
    src = next(iter(bmap.values()), None) if basin == "Brasil" \
        else bmap.get(basin)
    if src is None:
        return base, "Máximo"
    lags = _available_lags(src, season)
    opts = base + [{"label": f"Lag {x}", "value": x} for x in lags]
    return opts, "Máximo"


@app.callback(
    Output("stats-store", "data"),
    Input("dd-exp-explore",     "value"),
    Input("dd-cluster-explore", "value"),
)
def cb_load_stats(exp: str, basin: str):
    if not exp or not basin:
        return None
    th = 0.2
    df = compute_mov_stats(RESULTS_DIR, exp, basin, th)
    return df.to_json(date_format="iso", orient="split")


@app.callback(
    Output("explore-content", "children"),
    Input("stats-store",       "data"),
    Input("ri-season-explore", "value"),
    Input("dd-mov-map",        "value"),
    Input("dd-lag-map",        "value"),
    Input("ri-map-view",       "value"),
    State("dd-exp-explore",     "value"),
    State("dd-cluster-explore", "value"),
)
def cb_explore_content(
    stats_json: str,
    season: str, mov_map: str, lag_map, map_view: str,
    exp: str, basin: str,
):
    if not stats_json or not exp or not basin:
        return html.P("Selecione um experimento e um cluster.")

    th = 0.2
    df = pd.read_json(stats_json, orient="split")

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
        ("Max_alpha",      f"Max PREDEP{_i}"),
        ("N_pixels_alpha", f"N pixels PREDEP > {th:.2f}{_i}"),
        ("Pct_pixels_alpha", f"% pixels PREDEP{_i}"),
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
            "**Máximo PREDEP**\n\n"
            "Maior valor de PREDEP encontrado em qualquer pixel e lag. "
            "PREDEP mede a dependência **não-linear** do MoV sobre a "
            "precipitação (PREDEP=0 = independência; PREDEP=1 = previsão "
            "perfeita). Pode ser alto mesmo quando R² é baixo."
        ),
        "N_pixels_alpha": (
            f"**N pixels com PREDEP > {th:.2f}**\n\n"
            "Número de pixels únicos onde o **melhor lag** tem "
            f"PREDEP acima de {th:.2f}. "
            "Indica a extensão espacial do sinal não-linear."
        ),
        "Pct_pixels_alpha": (
            f"**% pixels com PREDEP > {th:.2f}**\n\n"
            "Percentual de pixels válidos da bacia com "
            f"PREDEP > {th:.2f} no melhor lag."
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
    index_r = _get_results_index(RESULTS_DIR)
    bmap_r = index_r.get(exp, {}).get(mov_map, {}) if mov_map else {}
    if basin == "Brasil":
        src = next(iter(bmap_r.values()), None)
    else:
        src = bmap_r.get(basin)
    # "Lag ótimo" sempre agrega sobre lags; demais visões respeitam o seletor
    lag_arg = "Máximo" if (map_view == "lag" or not lag_map) else lag_map
    layers = _map_layers(src, basin, season, lag_arg) if src else None

    if layers is not None:
        lons, lats = layers["lons"], layers["lats"]
        season_used = layers["season_used"]
        lag_txt = ("máx sobre lags" if lag_arg == "Máximo"
                   else f"lag {lag_arg}")

        # Downsample 2× for Brasil-wide maps (~70k → ~17.5k pixels)
        # to reduce Plotly JSON payload and browser memory.
        if basin == "Brasil" and len(lats) > 50 and len(lons) > 50:
            lons = lons[::2]
            lats = lats[::2]
            for k in ("r2", "alpha", "r2_albest", "best_lag_r2",
                       "best_lag_alpha"):
                v = layers.get(k)
                if v is not None and hasattr(v, '__getitem__'):
                    layers[k] = v[::2, ::2]

        alpha_scale = _alpha_colorscale()

        if map_view == "diff":
            # α − R² no MESMO lag (paper): específico → α_L − r2_L;
            # Máximo → α_max − R² no lag ótimo do α
            r2_same = layers["r2_albest"]
            alpha_v = layers["alpha"]
            diff = alpha_v - r2_same
            # cinza onde ambos < 0.1 (idêntico ao estático)
            low = (alpha_v < 0.1) & (r2_same < 0.1)
            diff = np.where(low | np.isnan(alpha_v) | np.isnan(r2_same),
                            np.nan, diff)
            gray = np.where(low, 0.0, np.nan)
            # faixa pelos percentis 1–99% simétricos (idêntico ao estático)
            finite = diff[np.isfinite(diff)]
            if finite.size:
                p1 = float(np.percentile(finite, 1))
                p99 = float(np.percentile(finite, 99))
                vabs = max(abs(p1), abs(p99), 0.05)
                zmin, zmax = max(p1, -vabs), min(p99, vabs)
            else:
                zmin, zmax = -1.0, 1.0
            fig_map = make_subplots(
                rows=1, cols=1,
                subplot_titles=[f"PREDEP − R²  ({lag_txt})"],
            )
            fig_map.add_trace(go.Heatmap(
                x=lons, y=lats, z=gray,
                colorscale=[[0, "#bfbfbf"], [1, "#bfbfbf"]],
                showscale=False, hoverinfo="skip",
            ), row=1, col=1)
            fig_map.add_trace(go.Heatmap(
                x=lons, y=lats, z=diff,
                colorscale="RdBu_r", zmid=0, zmin=zmin, zmax=zmax,
                colorbar=dict(
                    title="PREDEP − R²<br>← R² | PREDEP →", thickness=14
                ),
                hovertemplate=(
                    "Lon: %{x:.3f}<br>Lat: %{y:.3f}"
                    "<br>PREDEP−R²: %{z:.4f}<extra></extra>"
                ),
            ), row=1, col=1)
            map_title = f"PREDEP − R² — {mov_map} | {season_used} | {basin}"
            fig_map.update_layout(
                title=dict(text=map_title, font=dict(size=13), x=0),
                xaxis=dict(showgrid=False, scaleanchor="y", scaleratio=1),
                yaxis=dict(showgrid=False),
                margin=dict(l=60, r=60, t=60, b=50),
                height=520, dragmode="zoom",
            )

        elif map_view == "lag":
            avail = _available_lags(src, season)
            lmin, lmax = (min(avail), max(avail)) if avail else (0, 12)
            lag_scale = _lag_colorscale(avail)
            fig_map = make_subplots(
                rows=1, cols=2,
                subplot_titles=["Lag ótimo R²", "Lag ótimo PREDEP"],
                horizontal_spacing=0.06,
            )
            panels = [
                (layers["best_lag_r2"], layers["r2"], "R²", 1),
                (layers["best_lag_alpha"], layers["alpha"], "PREDEP", 2),
            ]
            for best_lag, val, lbl, col in panels:
                bl = np.array(best_lag, dtype=float)
                gray = np.where(~np.isnan(val), 0.0, np.nan)
                bl = np.where((val < 0.1) | np.isnan(val), np.nan, bl)
                # fundo cinza nos pixels com sinal fraco
                fig_map.add_trace(go.Heatmap(
                    x=lons, y=lats, z=gray,
                    colorscale=[[0, "#dcdcdc"], [1, "#dcdcdc"]],
                    showscale=False, hoverinfo="skip",
                ), row=1, col=col)
                fig_map.add_trace(go.Heatmap(
                    x=lons, y=lats, z=bl,
                    colorscale=lag_scale, zmin=lmin, zmax=lmax,
                    colorbar=dict(
                        title="Lag (meses)", thickness=14,
                        tickvals=avail,
                    ),
                    showscale=(col == 2),
                    hovertemplate=(
                        "Lon: %{x:.3f}<br>Lat: %{y:.3f}"
                        f"<br>lag ótimo ({lbl}): %{{z}}<extra></extra>"
                    ),
                ), row=1, col=col)
            map_title = (
                f"Lag ótimo (sinal ≥ 0.1) — {mov_map} | {season_used} "
                f"| {basin}"
            )
            fig_map.update_layout(
                title=dict(text=map_title, font=dict(size=13), x=0),
                xaxis=dict(showgrid=False, scaleanchor="y", scaleratio=1),
                yaxis=dict(showgrid=False),
                xaxis2=dict(showgrid=False, scaleanchor="y2", scaleratio=1),
                yaxis2=dict(showgrid=False),
                margin=dict(l=60, r=60, t=60, b=50),
                height=520, dragmode="zoom",
            )

        else:  # "ralpha" — R² e PREDEP lado a lado (padrão)
            fig_map = make_subplots(
                rows=1, cols=2,
                subplot_titles=["R²  (regressão linear)", "PREDEP"],
                horizontal_spacing=0.06,
            )
            for col, (z_data, lbl) in enumerate(
                [(layers["r2"], "R²"), (layers["alpha"], "PREDEP")], start=1
            ):
                fig_map.add_trace(go.Heatmap(
                    x=lons, y=lats, z=z_data,
                    coloraxis="coloraxis",
                    hovertemplate=(
                        "Lon: %{x:.3f}<br>Lat: %{y:.3f}"
                        f"<br>{lbl}: %{{z:.4f}}<extra></extra>"
                    ),
                ), row=1, col=col)
            map_title = (
                f"R² e PREDEP ({lag_txt}) — {mov_map} | {season_used}"
                f" | {basin}"
            )
            fig_map.update_layout(
                title=dict(text=map_title, font=dict(size=13), x=0),
                coloraxis=dict(
                    colorscale=alpha_scale, cmin=0.0, cmax=1.0,
                    colorbar=dict(
                        title="valor", thickness=14, tick0=0, dtick=0.1
                    ),
                ),
                xaxis=dict(showgrid=False, scaleanchor="y", scaleratio=1),
                yaxis=dict(showgrid=False),
                xaxis2=dict(showgrid=False, scaleanchor="y2", scaleratio=1),
                yaxis2=dict(showgrid=False),
                margin=dict(l=60, r=60, t=60, b=50),
                height=520, dragmode="zoom",
            )

        map_section = dcc.Graph(
            figure=fig_map,
            config={
                "scrollZoom": True,
                "displayModeBar": True,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                "toImageButtonOptions": {"filename": map_title},
            },
            style={"width": "100%", "marginBottom": "20px", "minHeight": "520px"},
        )
    else:
        map_section = html.P(
            "Selecione MoV para ver o mapa interativo.",
            style={
                "color": "#aaa", "fontStyle": "italic",
                "marginBottom": "20px",
            },
        )

    # ── floating button: abre galeria de plots exportados ────────────────────
    gallery_btn = None
    if mov_map:
        gallery_btn = html.A(
            "🖼  Ver plots exportados",
            href=f"/gallery/{exp}/{mov_map}/{basin}",
            target="_blank",
            style={
                "position": "fixed",
                "bottom": "24px",
                "right": "24px",
                "zIndex": "1000",
                "background": "#e07b39",
                "color": "white",
                "padding": "12px 18px",
                "borderRadius": "24px",
                "fontFamily": "sans-serif",
                "fontWeight": "600",
                "fontSize": "14px",
                "textDecoration": "none",
                "boxShadow": "0 2px 8px rgba(0,0,0,0.25)",
            },
        )

    # ── table (bottom) ───────────────────────────────────────────────────────
    children = [
        map_section,
        html.H4(title_table, style={
            "marginBottom": "8px", "fontWeight": "500", "fontSize": "15px",
        }),
        table,
    ]
    if gallery_btn is not None:
        children.append(gallery_btn)
    return children


def _build_regime_legend(regimes_meta: list) -> html.Div:
    """Painel abaixo do mapa de regimes com lag e trimestre por MoV."""
    has_lag_info = any(rm.get("combo_best_lag") for rm in regimes_meta)
    rows = []
    for rm in regimes_meta:
        combo      = rm.get("combo", [])
        lags       = rm.get("combo_best_lag")   or [None] * len(combo)
        seasons    = rm.get("combo_best_season") or [None] * len(combo)
        dom        = rm.get("dominant_season")

        chips: list = []
        for i, (mv, lg, ss) in enumerate(zip(combo, lags, seasons)):
            if i > 0:
                chips.append(" + ")
            if lg is not None and ss is not None:
                chips.append(html.Span([
                    html.B(mv),
                    html.Span(
                        f" ({lg}m, {ss})",
                        style={"color": "#888", "fontSize": "11px"},
                    ),
                ]))
            else:
                chips.append(html.B(mv))

        rows.append(html.Tr([
            html.Td(
                html.Span(style={
                    "display": "inline-block", "width": "13px", "height": "13px",
                    "background": rm["color_hex"], "borderRadius": "2px",
                    "verticalAlign": "middle",
                }),
                style={"padding": "3px 6px 3px 2px"},
            ),
            html.Td(
                f"R{rm['id']}",
                style={"fontWeight": "700", "padding": "3px 10px 3px 0",
                       "whiteSpace": "nowrap"},
            ),
            html.Td(
                html.Span(
                    dom or "—",
                    style={"background": "#e0e8f8", "borderRadius": "3px",
                           "padding": "1px 6px", "fontSize": "11px",
                           "fontWeight": "600", "letterSpacing": "0.02em"},
                ),
                style={"padding": "3px 10px 3px 0", "whiteSpace": "nowrap"},
            ),
            html.Td(chips, style={"fontSize": "12px", "padding": "3px 10px 3px 0"}),
            html.Td(
                f"n={rm['size']:,}",
                style={"color": "#999", "fontSize": "11px", "padding": "3px 0",
                       "whiteSpace": "nowrap"},
            ),
        ]))

    note = (
        "" if has_lag_info
        else " (regenere o parquet com som_to_parquet.py para ver lag e trimestre)"
    )
    return html.Div([
        html.P(
            f"Detalhes por regime — lag e trimestre climático mais frequente por MoV{note}:",
            style={"fontSize": "12px", "color": "#666", "margin": "0 0 6px 0"},
        ),
        html.Table(
            [
                html.Thead(html.Tr([
                    html.Th("", style={"width": "18px"}),
                    html.Th("Regime", style={
                        "textAlign": "left", "fontSize": "11px",
                        "fontWeight": "600", "padding": "2px 10px 4px 0",
                    }),
                    html.Th("Trimestre", style={
                        "textAlign": "left", "fontSize": "11px",
                        "fontWeight": "600", "padding": "2px 10px 4px 0",
                    }),
                    html.Th("MoVs  (lag, estação mais frequente)", style={
                        "textAlign": "left", "fontSize": "11px",
                        "fontWeight": "600", "padding": "2px 10px 4px 0",
                    }),
                    html.Th("Pixels", style={
                        "textAlign": "left", "fontSize": "11px",
                        "fontWeight": "600", "padding": "2px 0 4px 0",
                    }),
                ])),
                html.Tbody(rows),
            ],
            style={"borderCollapse": "collapse", "width": "100%"},
        ),
    ], style={
        "background": "#f7f8fa",
        "borderRadius": "6px",
        "padding": "12px 16px",
        "marginTop": "8px",
    })


_SOM_HELP = {
    "regime": (
        "**Regimes de previsibilidade** — cada pixel recebe a cor do regime "
        "(cluster do SOM) ao qual pertence. Pixels do mesmo regime têm assinatura "
        "climática parecida: os mesmos MoVs tendem a ser os melhores preditores de "
        "precipitação naquela área.\n\n"
        "No modo **best_mov** (usado aqui), cada pixel é descrito por 23 features — "
        "o melhor α de cada MoV (máximo sobre lag × estação). O SOM agrupa pixels "
        "por *quais* MoVs preveem melhor, não por quanto. A legenda **Rk** mostra "
        "a combinação dos 3 MoVs com maior α bruto naquele regime."
    ),
    "atypicality": (
        "**Atipicidade (erro de quantização)** — distância euclidiana, no espaço "
        "normalizado (z-score das 23 features), entre o pixel e o neurônio mais "
        "próximo do SOM (BMU — Best Matching Unit).\n\n"
        "- **Claro (baixo)**: pixel bem representado pelo seu neurônio; assinatura "
        "típica do regime.\n"
        "- **Escuro (alto)**: pixel atípico ou em zona de transição; o SOM não "
        "encontrou um neurônio próximo, indicando uma mistura de regimes ou padrão "
        "raro.\n\n"
        "Útil para identificar regiões de fronteira difusa ou comportamento climático "
        "ambíguo."
    ),
    "boundary": (
        "**Fronteiras entre regimes (U-matrix projetada)** — para cada pixel, "
        "valor da U-matrix do neurônio que o representa (BMU). A U-matrix mede a "
        "distância média do neurônio aos seus vizinhos na grade SOM.\n\n"
        "- **Alto (escuro)**: BMU no limite entre grupos de neurônios distintos → "
        "fronteira geográfica entre regimes.\n"
        "- **Baixo (claro)**: BMU em área homogênea da grade → interior de um "
        "regime.\n\n"
        "Complementa o mapa de regimes mostrando onde as transições são abruptas "
        "versus graduais."
    ),
    "component": (
        "**Component plane** — α (PREDEP) do MoV selecionado, por pixel. "
        "No modo *best_mov*, exibe o **melhor α** desse MoV para o pixel "
        "(máximo sobre os 6 lags × 4 estações).\n\n"
        "α ∈ [0, 1] é a estatística de previsibilidade PREDEP — análogo ao R² "
        "da regressão linear, mas calculado via teoria da informação.\n\n"
        "A escala é **compartilhada entre todos os MoVs**, permitindo comparar "
        "diretamente o poder preditivo relativo de cada índice climático em cada "
        "região."
    ),
}


_SOM_TRAINING_SEASON_RUN_ID = {
    "geral": _FIXED_EXP,
    "DJF":   f"{_FIXED_EXP}_DJF",
    "MAM":   f"{_FIXED_EXP}_MAM",
    "JJA":   f"{_FIXED_EXP}_JJA",
    "SON":   f"{_FIXED_EXP}_SON",
}


@app.callback(
    Output("som-content", "children"),
    Input("ri-som-view", "value"),
    Input("ri-som-training-season", "value"),
)
def cb_som_content(view: str, training_season: str):
    run_id = _SOM_TRAINING_SEASON_RUN_ID.get(training_season or "geral", _FIXED_EXP)
    exp = run_id
    n_regimes = 7
    loaded = load_som(RESULTS_DIR, run_id, n_regimes=n_regimes)
    if loaded is None:
        if training_season and training_season != "geral":
            cmd = (
                f"python scripts/som_to_parquet.py "
                f"--exp exp_brasil --feature-mode best_mov --n-regimes 7 "
                f"--movs AAO,AMO,AO,ATL3,NAO,NIN03,NIN034,NIN04,NIN12,ONI,"
                f"PDO,PSA1,PSA2,SAODI,SASDI,SOI,TNA,TSA --strict "
                f"--season {training_season} --run-id {run_id}"
            )
            return html.Div([
                html.P(
                    f"SOM para a estação {training_season} ainda não foi gerado.",
                    style={"color": "#999", "margin": "0 0 6px 0"},
                ),
                html.Pre(
                    cmd,
                    style={
                        "background": "#1e1e1e", "color": "#d4d4d4",
                        "borderRadius": "6px", "padding": "12px 16px",
                        "fontSize": "12px", "overflowX": "auto",
                        "whiteSpace": "pre-wrap",
                    },
                ),
            ], style={"padding": "16px"})
        return html.P("Artefato SOM não encontrado para exp_brasil.")
    df, meta = loaded

    def _piv(col):
        return df.pivot(
            index="latitude", columns="longitude", values=col
        ).sort_index()

    reverse = False
    regime_view = (view == "regime")
    if view == "component":
        mov = meta["mov_names"][0]
        p = _piv(f"{mov}_alpha")
        colorscale = "ylorrd"
        cmin, cmax = meta["component_alpha_vmin"], meta["component_alpha_vmax"]
        _fm = meta.get("feature_mode", "full")
        a_lbl = {"best_mov": "melhor α", "precip_latlon": "valor"}.get(_fm, "α médio")
        hov = f"{a_lbl} ({mov})"
        title = f"Component plane — {a_lbl} de {mov} | {exp}"
        cbar = dict(title=a_lbl, thickness=14)
    elif view == "atypicality":
        p = _piv("atypicality")
        colorscale, reverse = "magma", True
        cmin, cmax = meta["atypicality_min"], meta["atypicality_max"]
        cbar = dict(title="atipicidade", thickness=14)
        hov, title = "atipicidade", f"Atipicidade (erro de quantização) | {exp}"
    elif view == "boundary":
        p = _piv("boundary")
        colorscale, reverse = "ice", True
        cmin, cmax = meta["boundary_min"], meta["boundary_max"]
        cbar = dict(title="U-matrix", thickness=14)
        hov, title = "fronteira", f"Fronteiras entre regimes (U-matrix) | {exp}"
    else:  # regime
        p = _piv("regime")
        n = meta["n_regimes"]
        cols_hex = [r["color_hex"] for r in meta["regimes"]]
        colorscale = []
        for i, c in enumerate(cols_hex):
            colorscale += [[i / n, c], [(i + 1) / n, c]]
        cmin, cmax = -0.5, n - 0.5
        cbar = dict(
            title="Regime", thickness=14, tickmode="array",
            tickvals=list(range(n)),
            ticktext=[f"R{r['id']}: {r.get('label', r['distinctive_mov'])}"
                      for r in meta["regimes"]],
        )
        hov, title = "regime", f"Regimes de previsibilidade (SOM) | {exp}"

    lons, lats, z = p.columns.values, p.index.values, p.values
    heat = dict(
        x=lons, y=lats, z=z, colorscale=colorscale, zmin=cmin, zmax=cmax,
        colorbar=cbar,
        hovertemplate=(
            "Lon: %{x:.2f}<br>Lat: %{y:.2f}"
            + (f"<br>{hov}: %{{z:.0f}}<extra></extra>" if regime_view
               else f"<br>{hov}: %{{z:.3f}}<extra></extra>")
        ),
    )
    if reverse:
        heat["reversescale"] = True

    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Heatmap(**heat), row=1, col=1)
    fig.update_layout(
        title=dict(text=title, font=dict(size=13), x=0),
        xaxis=dict(showgrid=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(showgrid=False),
        margin=dict(l=60, r=60, t=60, b=50),
        height=1120, dragmode="zoom",
    )
    graph = dcc.Graph(
        figure=fig,
        config={
            "scrollZoom": True, "displayModeBar": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            "toImageButtonOptions": {"filename": title},
        },
        style={"width": "100%", "marginBottom": "12px", "height": "1120px"},
    )
    if regime_view:
        return html.Div([graph, _build_regime_legend(meta["regimes"])])
    return graph


@app.callback(
    Output("dd-cluster-overview", "options"),
    Output("dd-cluster-overview", "value"),
    Input("dd-exp-overview", "value"),
)
def cb_clusters_overview(exp: str):
    if not exp:
        return [], None
    index = _get_results_index(RESULTS_DIR)
    basins = sorted({
        basin
        for mov_map in index.get(exp, {}).values()
        for basin in mov_map
    })
    if not basins:
        return [], None
    opts = ([{"label": "Brasil", "value": "Brasil"}]
            + _basin_opts(basins))
    return opts, "Brasil"


@app.callback(
    Output("overview-content", "children"),
    Input("dd-exp-overview", "value"),
    Input("dd-cluster-overview", "value"),
    Input("ri-season-overview", "value"),
    Input("sl-threshold-overview", "value"),
)
def cb_overview_content(exp: str, basin: str, season: str, threshold: float):
    if not exp or not basin:
        return html.P("Selecione um experimento e um cluster.")

    layers = _compute_overview_layers(RESULTS_DIR, exp, basin, season)
    if not layers:
        return html.P("Nenhum dado encontrado para esta seleção.")
    
    lons, lats = layers["lons"], layers["lats"]
    
    # Downsample para Brasil
    if basin == "Brasil" and len(lats) > 50 and len(lons) > 50:
        lons = lons[::2]
        lats = lats[::2]
        for k in ["r2", "alpha", "diff", "mov_r2", "mov_alpha", "lag_r2", "lag_alpha"]:
            v = layers.get(k)
            if v is not None and hasattr(v, '__getitem__'):
                layers[k] = v[::2, ::2]

    # ── Threshold: por métrica — R² maps só onde R² >= thr, PREDEP só onde alpha >= thr
    thr = threshold or 0.0
    footprint_gray = np.where(
        ~np.isnan(layers["r2"]) | ~np.isnan(layers["alpha"]),
        0.0, np.nan,
    )
    if thr > 0:
        mask_r2    = np.nan_to_num(layers["r2"],    nan=0.0) < thr
        mask_alpha = np.nan_to_num(layers["alpha"], nan=0.0) < thr
        for k, mask in [
            ("r2",        mask_r2),
            ("mov_r2",    mask_r2),
            ("lag_r2",    mask_r2),
            ("alpha",     mask_alpha),
            ("mov_alpha", mask_alpha),
            ("lag_alpha", mask_alpha),
            ("diff",      mask_r2 & mask_alpha),
        ]:
            if layers.get(k) is not None:
                layers[k] = np.where(mask, np.nan, layers[k])

    mov_names = layers["mov_names"]
    mov_scale = _mov_colorscale(mov_names)

    n_movs = len(mov_names)
    if n_movs > 0:
        mov_tickvals = [i + 0.5 for i in range(n_movs)]
        mov_ticktext = mov_names
    else:
        mov_tickvals = []
        mov_ticktext = []

    alpha_scale = _alpha_colorscale()

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            "R² Máximo (todos os MoVs)", "PREDEP Máximo (todos os MoVs)",
            "MoV Vencedor (R²)", "MoV Vencedor (PREDEP)",
            "Lag Ótimo do MoV Vencedor (R²)", "Lag Ótimo do MoV Vencedor (PREDEP)",
        ],
        horizontal_spacing=0.06,
        vertical_spacing=0.1,
    )
    
    _gray = [[0, "#dcdcdc"], [1, "#dcdcdc"]]

    # Row 1: R² max e PREDEP max (com fundo cinza para footprint)
    for col, (z_data, lbl) in enumerate(
        [(layers["r2"], "R²"), (layers["alpha"], "PREDEP")], start=1
    ):
        fig.add_trace(go.Heatmap(
            x=lons, y=lats, z=footprint_gray,
            colorscale=_gray, showscale=False, hoverinfo="skip",
        ), row=1, col=col)
        fig.add_trace(go.Heatmap(
            x=lons, y=lats, z=z_data,
            colorscale=alpha_scale, zmin=0.0, zmax=1.0,
            colorbar=dict(title="Valor", thickness=14, y=0.866, len=0.26) if col == 2 else None,
            showscale=(col == 2),
            hovertemplate=(
                "Lon: %{x:.3f}<br>Lat: %{y:.3f}"
                f"<br>{lbl}: %{{z:.4f}}<extra></extra>"
            ),
        ), row=1, col=col)

    # Row 2: MoV Vencedor
    for col, (z_data, lbl) in enumerate(
        [(layers["mov_r2"], "R²"), (layers["mov_alpha"], "PREDEP")], start=1
    ):
        fig.add_trace(go.Heatmap(
            x=lons, y=lats, z=footprint_gray,
            colorscale=_gray, showscale=False, hoverinfo="skip",
        ), row=2, col=col)

        custom_data = np.empty(z_data.shape, dtype=object)
        for i, name in enumerate(mov_names):
            custom_data[z_data == i] = name

        fig.add_trace(go.Heatmap(
            x=lons, y=lats, z=z_data,
            customdata=custom_data,
            colorscale=mov_scale, zmin=0, zmax=n_movs,
            colorbar=dict(
                title="MoV", thickness=14,
                tickvals=mov_tickvals, ticktext=mov_ticktext,
                y=0.5, len=0.26,
            ) if col == 2 else None,
            showscale=(col == 2),
            hovertemplate=(
                "Lon: %{x:.3f}<br>Lat: %{y:.3f}"
                f"<br>MoV Vencedor ({lbl}): %{{customdata}}<extra></extra>"
            ),
        ), row=2, col=col)

    # Row 3: Lag Ótimo
    all_lags = sorted(list(set(
        layers["lag_r2"][~np.isnan(layers["lag_r2"])].tolist()
        + layers["lag_alpha"][~np.isnan(layers["lag_alpha"])].tolist()
    )))
    if all_lags:
        lag_scale = _lag_colorscale(all_lags)
        lmin, lmax = min(all_lags), max(all_lags)
    else:
        lag_scale = _lag_colorscale([0, 12])
        lmin, lmax = 0, 12

    for col, (z_data, lbl) in enumerate(
        [(layers["lag_r2"], "R²"), (layers["lag_alpha"], "PREDEP")], start=1
    ):
        fig.add_trace(go.Heatmap(
            x=lons, y=lats, z=footprint_gray,
            colorscale=_gray, showscale=False, hoverinfo="skip",
        ), row=3, col=col)
        
        fig.add_trace(go.Heatmap(
            x=lons, y=lats, z=z_data,
            colorscale=lag_scale, zmin=lmin, zmax=lmax,
            colorbar=dict(
                title="Lag (meses)", thickness=14,
                tickvals=all_lags,
                y=0.133, len=0.26,
            ) if col == 2 else None,
            showscale=(col == 2),
            hovertemplate=(
                "Lon: %{x:.3f}<br>Lat: %{y:.3f}"
                f"<br>Lag Ótimo ({lbl}): %{{z}}<extra></extra>"
            ),
        ), row=3, col=col)
        
    fig.update_layout(
        height=1400, dragmode="zoom",
        margin=dict(l=60, r=60, t=60, b=50),
        title=dict(text=f"Overview Global (Todos os MoVs) — {season} | {basin}", font=dict(size=14)),
    )
    for i in range(1, 4):
        for j in range(1, 3):
            suffix = f"{j if i==1 and j==2 else (i-1)*2+j}"
            if suffix == "1": suffix = ""
            fig.update_layout(**{
                f"xaxis{suffix}": dict(showgrid=False, scaleanchor=f"y{suffix}", scaleratio=1),
                f"yaxis{suffix}": dict(showgrid=False),
            })
            
    # Figure 2: Métrica Vencedora (PREDEP vs R²)
    diff = layers["diff"]
    low = (layers["alpha"] < 0.1) & (layers["r2"] < 0.1)
    
    # 0 para R², 1 para PREDEP
    winner = np.where(diff > 0, 1.0, 0.0)
    winner = np.where(low | np.isnan(layers["alpha"]) | np.isnan(layers["r2"]),
                      np.nan, winner)
    gray = np.where(low, 0.0, np.nan)
    
    fig_diff = make_subplots(
        rows=1, cols=1,
        subplot_titles=[f"Métrica Vencedora (Qual métrica obteve o maior valor máximo?)"],
    )
    fig_diff.add_trace(go.Heatmap(
        x=lons, y=lats, z=gray,
        colorscale=[[0, "#bfbfbf"], [1, "#bfbfbf"]],
        showscale=False, hoverinfo="skip",
    ), row=1, col=1)
    
    custom_data = np.empty(winner.shape, dtype=object)
    custom_data[winner == 0] = "R²"
    custom_data[winner == 1] = "PREDEP"
    
    # Escala discreta com 2 cores: Vermelho (R²) e Azul (PREDEP)
    cat_scale = [
        [0.0, "#d62728"], [0.5, "#d62728"],
        [0.5, "#1f77b4"], [1.0, "#1f77b4"]
    ]
    
    fig_diff.add_trace(go.Heatmap(
        x=lons, y=lats, z=winner,
        customdata=custom_data,
        colorscale=cat_scale, zmin=-0.5, zmax=1.5,
        colorbar=dict(
            title="Métrica Vencedora", thickness=14,
            tickvals=[0, 1],
            ticktext=["R²", "PREDEP"],
        ),
        hovertemplate=(
            "Lon: %{x:.3f}<br>Lat: %{y:.3f}"
            "<br>Vencedor: %{customdata}<extra></extra>"
        ),
    ), row=1, col=1)
    
    fig_diff.update_layout(
        title=dict(text=f"Comparação de Máximos — {season} | {basin}", font=dict(size=13), x=0),
        xaxis=dict(showgrid=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(showgrid=False),
        margin=dict(l=60, r=60, t=60, b=50),
        height=520, dragmode="zoom",
    )

    return html.Div([
        dcc.Graph(figure=fig, config={"displayModeBar": False},
                  style={"minHeight": "1400px"}),
        html.Hr(style={"margin": "40px 0"}),
        dcc.Graph(figure=fig_diff, config={"displayModeBar": False},
                  style={"minHeight": "520px"}),
    ])


def _build_overview_alt_map(layers: dict, family: str, metric: str) -> dcc.Graph:
    """
    Monta o mapa de uma coluna da aba "Overview (alternativo)".
    family: "r2" ou "alpha" — qual família de métrica esta coluna mostra.
    metric: "max" | "mov" | "lag" — visão selecionada no dropdown.
    O tooltip sempre mostra as 3 facetas da família (Máximo, MoV Vencedor,
    Lag Ótimo), independente de `metric`.
    """
    lons, lats = layers["lons"], layers["lats"]
    mov_names = layers["mov_names"]
    n_movs = len(mov_names)

    val = layers["r2"] if family == "r2" else layers["alpha"]
    mov_idx = layers["mov_r2"] if family == "r2" else layers["mov_alpha"]
    lag = layers["lag_r2"] if family == "r2" else layers["lag_alpha"]
    basin_grid = layers.get("basin")

    lbl = "R²" if family == "r2" else "PREDEP"

    basin_cd = (
        np.where(pd.isna(basin_grid), "—", basin_grid.astype(object))
        if basin_grid is not None else np.full(val.shape, "—", dtype=object)
    )
    mov_name_cd = np.full(val.shape, "—", dtype=object)
    for i, name in enumerate(mov_names):
        mov_name_cd[mov_idx == i] = name

    customdata = np.dstack([basin_cd, val, mov_name_cd, lag])

    hovertemplate = (
        "Bacia: %{customdata[0]}<br>"
        "Lon: %{x:.3f}<br>Lat: %{y:.3f}<br>"
        f"{lbl} Máximo: " + "%{customdata[1]:.4f}<br>"
        f"MoV Vencedor ({lbl}): " + "%{customdata[2]}<br>"
        f"Lag Ótimo ({lbl}): " + "%{customdata[3]}"
        "<extra></extra>"
    )

    if metric == "max":
        z = val
        colorscale, zmin, zmax = _alpha_colorscale(), 0.0, 1.0
        cb_title, cb_kwargs = "Valor", {}
    elif metric == "mov":
        z = mov_idx
        colorscale, zmin, zmax = _mov_colorscale(mov_names), 0, n_movs
        tickvals = [i + 0.5 for i in range(n_movs)] if n_movs else []
        cb_title, cb_kwargs = "MoV", dict(tickvals=tickvals, ticktext=mov_names)
    else:  # "lag"
        valid_lags = sorted(set(
            int(x) for x in lag[~np.isnan(lag)].tolist()
        )) or [0, 12]
        colorscale = _lag_colorscale(valid_lags)
        zmin, zmax = min(valid_lags), max(valid_lags)
        z = lag
        cb_title, cb_kwargs = "Lag (meses)", dict(tickvals=valid_lags)

    fig = go.Figure()

    gray_ref = val if metric == "max" else z
    gray = np.where(~np.isnan(gray_ref), 0.0, np.nan)
    fig.add_trace(go.Heatmap(
        x=lons, y=lats, z=gray,
        colorscale=[[0, "#dcdcdc"], [1, "#dcdcdc"]],
        showscale=False, hoverinfo="skip",
    ))

    fig.add_trace(go.Heatmap(
        x=lons, y=lats, z=z, customdata=customdata,
        colorscale=colorscale, zmin=zmin, zmax=zmax,
        showscale=True,
        colorbar=dict(
            title=cb_title, orientation="h",
            x=0.5, xanchor="center",
            y=-0.18, yanchor="top",
            len=0.8, thickness=14,
            **cb_kwargs,
        ),
        hovertemplate=hovertemplate,
    ))

    lon_min, lon_max = float(np.min(lons)), float(np.max(lons))
    lat_min, lat_max = float(np.min(lats)), float(np.max(lats))
    lon_pad = (lon_max - lon_min) * 0.02 or 0.5
    lat_pad = (lat_max - lat_min) * 0.02 or 0.5

    titles = {
        "max": f"{lbl} Máximo",
        "mov": f"MoV Vencedor ({lbl})",
        "lag": f"Lag Ótimo ({lbl})",
    }
    fig.update_layout(
        height=560, dragmode="zoom",
        margin=dict(l=50, r=50, t=50, b=90),
        title=dict(text=titles[metric], font=dict(size=14)),
        xaxis=dict(
            range=[lon_min - lon_pad, lon_max + lon_pad],
            showgrid=False, scaleanchor="y", scaleratio=1,
        ),
        yaxis=dict(
            range=[lat_min - lat_pad, lat_max + lat_pad],
            showgrid=False,
        ),
    )
    return dcc.Graph(figure=fig, config={"displayModeBar": False})


def _single_map_graph(
    lons: np.ndarray,
    lats: np.ndarray,
    z: np.ndarray,
    title: str,
    colorscale=None,
    zmin: float = 0.0,
    zmax: float = 1.0,
    cb_title: str = "Valor",
    cb_extra: dict | None = None,
    height: int = 480,
    gray_bg: bool = False,
    hovertemplate: str | None = None,
    customdata=None,
    z_footprint: np.ndarray | None = None,
) -> dcc.Graph:
    """Heatmap geográfico individual — padrão estável usado no Overview (alternativo).

    z_footprint: máscara de footprint pré-threshold para o fundo cinza.
    Se None, usa z para determinar quais pixels têm fundo cinza.
    """
    if colorscale is None:
        colorscale = _alpha_colorscale()

    lon_min, lon_max = float(np.nanmin(lons)), float(np.nanmax(lons))
    lat_min, lat_max = float(np.nanmin(lats)), float(np.nanmax(lats))
    lon_pad = (lon_max - lon_min) * 0.02 or 0.5
    lat_pad = (lat_max - lat_min) * 0.02 or 0.5

    fig = go.Figure()
    if gray_bg:
        z_base = z_footprint if z_footprint is not None else z
        fig.add_trace(go.Heatmap(
            x=lons, y=lats,
            z=np.where(~np.isnan(z_base), 0.0, np.nan),
            colorscale=[[0, "#dcdcdc"], [1, "#dcdcdc"]],
            showscale=False, hoverinfo="skip",
        ))
    fig.add_trace(go.Heatmap(
        x=lons, y=lats, z=z,
        colorscale=colorscale, zmin=zmin, zmax=zmax,
        colorbar=dict(
            title=cb_title, thickness=12, len=0.85,
            orientation="h", x=0.5, xanchor="center",
            y=-0.18, yanchor="top",
            **(cb_extra or {}),
        ),
        customdata=customdata,
        hovertemplate=(
            hovertemplate or
            f"Lon: %{{x:.3f}}<br>Lat: %{{y:.3f}}<br>{cb_title}: %{{z:.4f}}<extra></extra>"
        ),
    ))
    fig.update_layout(
        height=height, dragmode="zoom",
        margin=dict(l=40, r=40, t=40, b=80),
        title=dict(text=title, font=dict(size=12), x=0),
        xaxis=dict(
            range=[lon_min - lon_pad, lon_max + lon_pad],
            showgrid=False, scaleanchor="y", scaleratio=1,
        ),
        yaxis=dict(
            range=[lat_min - lat_pad, lat_max + lat_pad],
            showgrid=False,
        ),
    )
    return dcc.Graph(
        figure=fig,
        config={"scrollZoom": True, "displayModeBar": False},
        style={"flex": "1", "minWidth": "300px", "height": f"{height}px"},
    )


@app.callback(
    Output("overview-alt-perlags-content", "children"),
    Input("ri-season-overview-alt", "value"),
    Input("sl-threshold-overview-alt", "value"),
)
def cb_overview_alt_perlags(season: str, threshold: float):
    per_lag = _get_brasil_per_lag(season or "Todas")
    if not per_lag:
        return html.P("Nenhum dado encontrado.")

    thr = threshold or 0.0

    rows = []
    for lag in sorted(per_lag.keys()):
        d = per_lag[lag]
        r2_orig = d["r2"]
        al_orig = d["alpha"]

        if thr > 0:
            r2_z = np.where(np.nan_to_num(r2_orig, nan=0.0) < thr, np.nan, r2_orig)
            al_z = np.where(np.nan_to_num(al_orig, nan=0.0) < thr, np.nan, al_orig)
        else:
            r2_z, al_z = r2_orig, al_orig
            r2_orig = al_orig = None  # sem footprint separado necessário

        lag_lbl = f"lag = {lag} {'mês' if lag == 1 else 'meses'}"
        rows.append(html.Div([
            _single_map_graph(d["lons"], d["lats"], r2_z,
                              f"R²  —  {lag_lbl}", cb_title="R²", height=600,
                              gray_bg=True, z_footprint=r2_orig),
            _single_map_graph(d["lons"], d["lats"], al_z,
                              f"PREDEP  —  {lag_lbl}", cb_title="PREDEP", height=600,
                              gray_bg=True, z_footprint=al_orig),
        ], style={"display": "flex", "gap": "8px", "marginBottom": "8px"}))

    return html.Div(rows)


@app.callback(
    Output("dd-cluster-lag0", "options"),
    Output("dd-cluster-lag0", "value"),
    Input("dd-exp-lag0", "value"),
)
def cb_clusters_lag0(exp: str):
    if not exp:
        return [], None
    index = _get_results_index(RESULTS_DIR)
    basins = sorted({
        basin
        for mov_map in index.get(exp, {}).values()
        for basin in mov_map
    })
    if not basins:
        return [], None
    opts = ([{"label": "Brasil", "value": "Brasil"}]
            + _basin_opts(basins))
    return opts, "Brasil"


@app.callback(
    Output("lag0-content", "children"),
    Input("dd-exp-lag0", "value"),
    Input("dd-cluster-lag0", "value"),
    Input("ri-season-lag0", "value"),
)
def cb_lag0_content(exp: str, basin: str, season: str):
    if not exp or not basin:
        return html.P("Selecione um experimento e um cluster.")

    layers = _compute_lag0_layers(RESULTS_DIR, exp, basin, season)
    if not layers:
        return html.P("Sem dados de lag=0 para esta seleção.")

    lons, lats = layers["lons"], layers["lats"]

    if basin == "Brasil" and len(lats) > 50 and len(lons) > 50:
        lons = lons[::2]
        lats = lats[::2]
        for k in ["r2", "alpha", "mov_r2", "mov_alpha"]:
            v = layers.get(k)
            if v is not None and hasattr(v, "__getitem__"):
                layers[k] = v[::2, ::2]

    mov_names = layers["mov_names"]
    mov_scale = _mov_colorscale(mov_names)
    n_movs = len(mov_names)
    mov_tickvals = [i + 0.5 for i in range(n_movs)]

    pfx = f"{season} | {basin}"

    # Linha 1: valores máximos
    row1 = html.Div([
        _single_map_graph(lons, lats, layers["r2"],
                          f"R² Máximo (lag=0) — {pfx}", cb_title="R²"),
        _single_map_graph(lons, lats, layers["alpha"],
                          f"PREDEP Máximo (lag=0) — {pfx}", cb_title="PREDEP"),
    ], style={"display": "flex", "gap": "8px", "marginBottom": "8px"})

    # Linha 2: MoV vencedor
    def _mov_graph(z_data, lbl):
        cd = np.empty(z_data.shape, dtype=object)
        for i, name in enumerate(mov_names):
            cd[z_data == i] = name
        return _single_map_graph(
            lons, lats, z_data,
            title=f"MoV Vencedor ({lbl}, lag=0) — {pfx}",
            colorscale=mov_scale, zmin=0, zmax=n_movs,
            cb_title="MoV",
            cb_extra={"tickvals": mov_tickvals, "ticktext": mov_names},
            gray_bg=True, customdata=cd,
            hovertemplate=(
                f"Lon: %{{x:.3f}}<br>Lat: %{{y:.3f}}"
                f"<br>MoV Vencedor ({lbl}): %{{customdata}}<extra></extra>"
            ),
        )

    row2 = html.Div([
        _mov_graph(layers["mov_r2"], "R²"),
        _mov_graph(layers["mov_alpha"], "PREDEP"),
    ], style={"display": "flex", "gap": "8px", "marginBottom": "8px"})

    return html.Div([
        html.H4(f"Lag 0 — MoV Vencedor por Pixel — {pfx}",
                style={"marginBottom": "12px", "fontSize": "14px", "fontWeight": "500"}),
        row1, row2,
    ])


@app.callback(
    Output("dd-cluster-destaque", "options"),
    Output("dd-cluster-destaque", "value"),
    Input("dd-exp-destaque", "value"),
)
def cb_clusters_destaque(exp: str):
    if not exp:
        return [], None
    index = _get_results_index(RESULTS_DIR)
    basins = sorted({
        basin
        for mov_map in index.get(exp, {}).values()
        for basin in mov_map
    })
    if not basins:
        return [], None
    opts = ([{"label": "Brasil", "value": "Brasil"}]
            + _basin_opts(basins))
    return opts, "Brasil"


def _destaque_scores(exp: str, basin: str) -> tuple:
    """Returns (scores_r2, scores_alpha) DataFrames sorted desc by score.
    Score penalizes MoVs with fewer pixels: Media × (N_validos / N_total_brasil).
    Missing pixels count as zero (Brasil-wide denominator).
    """
    df = compute_mov_stats(RESULTS_DIR, exp, basin, threshold=0.2)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    # Brazil-wide pixel count = max N_validos per season across all MoVs
    n_total = df.groupby("Season")["N_validos"].max()
    df = df.copy()
    df["_w"] = df.apply(lambda r: r["N_validos"] / max(n_total.get(r["Season"], 1), 1), axis=1)
    df["_media_alpha_brasil"] = df["Media_alpha"] * df["_w"]
    df["_media_r2_brasil"]    = df["Media_R2"]    * df["_w"]

    scores_alpha = (
        df.groupby("MoV")[["Max_alpha", "_media_alpha_brasil"]]
        .max()
        .assign(score=lambda d: d["Max_alpha"] * d["_media_alpha_brasil"])
        .sort_values("score", ascending=False)
    )
    scores_r2 = (
        df.groupby("MoV")[["Max_R2", "_media_r2_brasil"]]
        .max()
        .assign(score=lambda d: d["Max_R2"] * d["_media_r2_brasil"])
        .sort_values("score", ascending=False)
    )
    return scores_r2, scores_alpha


@app.callback(
    Output("dd-mov-destaque-r2",    "options"),
    Output("dd-mov-destaque-r2",    "value"),
    Output("dd-mov-destaque-alpha", "options"),
    Output("dd-mov-destaque-alpha", "value"),
    Input("dd-exp-destaque",      "value"),
    Input("dd-cluster-destaque",  "value"),
    Input("ri-season-destaque",   "value"),
)
def cb_mov_opts_destaque(exp: str, basin: str, season: str):
    empty = ([], None, [], None)
    if not exp or not basin:
        return empty
    scores_r2, scores_alpha = _destaque_scores(exp, basin)
    if scores_r2.empty:
        return empty

    def _opts(scores, metric):
        return [
            {"label": f"{mov}  ({metric} score {row['score']:.3f})", "value": mov}
            for mov, row in scores.iterrows()
        ]

    best_r2    = scores_r2.index[0]    if not scores_r2.empty    else None
    best_alpha = scores_alpha.index[0] if not scores_alpha.empty else None
    return _opts(scores_r2, "R²"), best_r2, _opts(scores_alpha, "PREDEP"), best_alpha


@app.callback(
    Output("destaque-content", "children"),
    Input("dd-exp-destaque",          "value"),
    Input("dd-cluster-destaque",      "value"),
    Input("ri-season-destaque",       "value"),
    Input("dd-mov-destaque-r2",       "value"),
    Input("dd-mov-destaque-alpha",    "value"),
)
def cb_destaque_content(exp: str, basin: str, season: str,
                        mov_r2: str, mov_alpha: str):
    if not exp or not basin or not mov_r2 or not mov_alpha:
        return html.P("Selecione experimento e bacia.")

    index = _get_results_index(RESULTS_DIR)

    def _src(mov):
        bmap = index.get(exp, {}).get(mov, {})
        return next(iter(bmap.values()), None) if basin == "Brasil" else bmap.get(basin)

    src_r2    = _src(mov_r2)
    src_alpha = _src(mov_alpha)
    if not src_r2 or not src_alpha:
        return html.P("Dados não encontrados.")

    # Union of available lags from both MoVs
    lags = sorted(
        set(_available_lags(src_r2, season)) | set(_available_lags(src_alpha, season))
    )
    if not lags:
        return html.P("Sem lags disponíveis.")

    same_mov = (mov_r2 == mov_alpha)
    title_txt = (
        f"{mov_r2} — R² e PREDEP por Lag — {season} | {basin}"
        if same_mov else
        f"R²: {mov_r2}  |  PREDEP: {mov_alpha} — por Lag — {season} | {basin}"
    )

    rows = []
    for lag in lags:
        layers_r2    = _map_layers(src_r2,    basin, season, lag)
        layers_alpha = _map_layers(src_alpha, basin, season, lag)

        graphs = []
        for lay, z_key, lbl, mov_lbl in [
            (layers_r2,    "r2",    "R²",    mov_r2),
            (layers_alpha, "alpha", "PREDEP", mov_alpha),
        ]:
            if lay is None:
                graphs.append(html.Div(style={"flex": "1"}))
                continue
            lons, lats = lay["lons"], lay["lats"]
            z = lay[z_key]
            if basin == "Brasil" and len(lats) > 50 and len(lons) > 50:
                lons, lats, z = lons[::2], lats[::2], z[::2, ::2]
            graphs.append(_single_map_graph(
                lons, lats, z,
                title=f"{lbl} — {mov_lbl}  lag={lag}",
                cb_title=lbl,
            ))

        rows.append(html.Div(
            graphs,
            style={"display": "flex", "gap": "8px", "marginBottom": "8px"},
        ))

    return html.Div([
        html.H4(title_txt, style={
            "marginBottom": "12px", "fontSize": "14px", "fontWeight": "500",
        }),
    ] + rows)


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
    results_index = _get_results_index(RESULTS_DIR)
    app.layout = _build_layout(results_index)

    exps_r = sorted(results_index.keys())
    print(f"Experimentos disponíveis: {exps_r}")
    print(f"Abrindo em   http://localhost:{args.port}/")
    app.run(debug=False, port=args.port, host=args.host)


if __name__ == "__main__":
    main()
