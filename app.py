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
import plotly.basedatatypes as _plotly_basedatatypes
from plotly.subplots import make_subplots
import xarray as xr
from dash import Dash, Input, Output, State, dash_table, dcc, html

# plotly>=6 base64-encodes homogeneous arrays (numpy) into a compact
# {"dtype","bdata"} spec unconditionally in Figure.to_dict() — but the
# plotly.js bundled by dash==4.1.0 (pinned in requirements.txt) predates
# support for decoding that spec, so every go.Heatmap/etc. renders blank in
# the browser even though the server-side callback succeeds. Disable it here
# instead of touching every trace call site.
_plotly_basedatatypes.convert_to_base64 = lambda obj: None

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
            "`modal run src/modal/som_insights_modal.py --exp-n 1 --dual "
            "--run-id exp01_full` (e variantes --mov-alpha-threshold) "
            "(repo irc_predep_bootstrap) para popular results/predep_som/.",
            style={"color": "#999", "fontStyle": "italic", "padding": "16px"},
        )
    return html.Div([
        html.Div([
            html.Div([
                html.Label(
                    ["Limiar de MoVs (PREDEP)", _hi(
                        "Filtra quais MoVs entram no SOM, com base no score de α "
                        "(PREDEP; média entre pixels do melhor lag×estação por "
                        "pixel de cada MoV). O mesmo subconjunto de MoVs é usado "
                        "para treinar os dois SOMs (PREDEP e R²) exibidos lado a "
                        "lado, garantindo uma comparação sobre os mesmos dados. "
                        "'Todos' usa os 23 MoVs sem filtro."
                    )],
                    style={"fontWeight": "500"},
                ),
                dcc.RadioItems(
                    id="ri-som-threshold",
                    options=[
                        {"label": "α > 0.01", "value": "gt001"},
                        {"label": "α > 0.05", "value": "gt005"},
                        {"label": "α > 0.1", "value": "gt01"},
                        {"label": "α > 0.15", "value": "gt015"},
                    ],
                    value="gt001", inline=True,
                    labelStyle={"marginRight": "12px"},
                ),
            ], style=_RADIO_DIV),
            html.Div([
                html.Label(
                    ["Visão SOM", _hi(
                        "Tipo de mapa a exibir, para os dois SOMs treinados sobre "
                        "o mesmo subconjunto de MoVs: à esquerda, sobre α (PREDEP, "
                        "∈ [0,1]); à direita, sobre r² (regressão linear, "
                        "baseline). 1 amostra = 1 pixel. O SOM preserva topologia: "
                        "pixels semelhantes caem em neurônios vizinhos. Os regimes "
                        "são agrupamentos de neurônios (cotovelo do WSS)."
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


_METODOLOGIA_STAGES = [
    {
        "letra": "A",
        "titulo": "Pré-processamento dos dados",
        "resumo": (
            "Chuva e índices de MoV são convertidos em anomalias mensais "
            "e o cálculo é sempre feito por bacia hidrográfica — nunca "
            "sobre o Brasil como uma população única de pixels."
        ),
        "detalhes": [
            "Anomalia: de cada série (X e Y) é subtraída sua própria "
            "climatologia mensal (média de cada mês-calendário) antes de "
            "qualquer cálculo — remove a componente sazonal determinística "
            "comum às duas séries, para que a dependência medida não seja "
            "artefato desse ciclo compartilhado.",
            "Segmentação por bacia: o α e o r² são estimados "
            "separadamente dentro de cada bacia hidrográfica, nunca sobre "
            "o conjunto nacional de pixels — evita agregar populações "
            "estatisticamente heterogêneas (distribuições marginais "
            "distintas) num único teste de dependência.",
            "Resolução temporal: agregação para série mensal, alinhando "
            "a granularidade de X (índice do MoV) e Y (precipitação) "
            "antes de entrarem no estimador.",
            "X e Y recebem exatamente o mesmo tratamento de "
            "pré-processamento (mesma transformação em anomalia), "
            "garantindo simetria entre as duas séries antes do cálculo "
            "assimétrico da etapa seguinte.",
        ],
    },
    {
        "letra": "B",
        "titulo": "Cálculo do PREDEP",
        "resumo": (
            "Para cada combinação de MoV, bacia, estação do ano e "
            "defasagem, estimamos α = (S_Y|X − S_Y)/S_Y|X por bootstrap, "
            "e computamos r² de uma regressão linear no mesmo par (X, Y) "
            "como referência direta de comparação."
        ),
        "detalhes": [
            "S_Y (dispersão marginal de Y) e S_Y|X (dispersão de Y "
            "condicionada a faixas de X) são estimadas pelo mesmo "
            "procedimento bootstrap em ambos os casos: reamostra-se pares "
            "da variável (Y inteira, ou Y restrita a uma faixa de X), "
            "calcula-se a diferença entre as duas reamostragens, e "
            "estima-se via KDE a densidade dessa diferença avaliada em "
            "zero.",
            "As faixas (bins) de X usadas no condicionamento não têm "
            "largura fixa nem número pré-definido — são obtidas por um "
            "particionamento adaptativo à distribuição empírica de X, "
            "evitando a escolha arbitrária de cortes.",
            "O resultado α fica bounded em [0,1]: 0 se e somente se X e "
            "Y forem independentes, 1 no limite de previsão perfeita de Y "
            "a partir de X.",
            "Estação do ano e defasagem (lag 0–12 meses) são tratadas "
            "como dimensões independentes da varredura de parâmetros — "
            "cada combinação (MoV, bacia, estação, lag) gera sua própria "
            "estimativa de α e r², não uma agregação única.",
            "α_{Y|X} é assimétrico por construção — mede especificamente "
            "quanto X reduz a incerteza sobre Y, não o inverso — e o r² "
            "calculado ao lado usa exatamente o mesmo par (X, Y), mesma "
            "estação e mesmo lag, servindo de baseline direto e "
            "comparável ponto a ponto.",
        ],
    },
    {
        "letra": "C",
        "titulo": "Organização dos resultados",
        "resumo": (
            "O resultado final é granular: para cada ponto do grid, "
            "guardamos o α e o r² de cada MoV, em cada estação do ano e "
            "cada defasagem testada — não um único número agregado por "
            "bacia."
        ),
        "detalhes": [
            "Mantemos as duas métricas (PREDEP e r²) lado a lado para "
            "todo ponto, em vez de guardar só a 'vencedora' — isso "
            "preserva a possibilidade de comparar os dois métodos "
            "depois, e de auditar onde eles concordam ou divergem.",
            "Os resultados calculados bacia por bacia são reunidos num "
            "mosaico único cobrindo o Brasil inteiro, permitindo tanto a "
            "leitura regional (uma bacia específica) quanto a leitura "
            "nacional (padrões que atravessam bacias).",
            "Para as visualizações e para o agrupamento por regime "
            "(aba SOM), derivamos resumos mais compactos a partir desse "
            "atlas completo — por exemplo, o 'melhor' MoV e lag por "
            "ponto — mas o dado granular original é sempre preservado "
            "como fonte de verdade.",
        ],
    },
    {
        "letra": "D",
        "titulo": "Plotagem",
        "resumo": (
            "Os mapas de α e r² são exibidos na mesma escala/grid, ponto "
            "a ponto, para leitura direta da divergência entre os dois "
            "métodos; um agrupamento não supervisionado (SOM) resume o "
            "perfil de dependência de cada pixel em regimes discretos."
        ),
        "detalhes": [
            "PREDEP e r² são plotados sobre o mesmo grid de pixels, com "
            "escalas comparáveis, especificamente para tornar legível a "
            "diferença ponto a ponto entre os dois métodos — não só o "
            "valor absoluto de cada um isoladamente.",
            "O SOM é treinado sobre o vetor de α de cada pixel (por MoV × "
            "lag × estação, ou uma redução desse vetor), sem usar "
            "latitude/longitude como feature — o agrupamento é por "
            "similaridade estatística do perfil de dependência, não por "
            "proximidade geográfica.",
            "Os regimes finais vêm de um KMeans aplicado sobre os "
            "neurônios do SOM já treinado — uma segunda etapa de "
            "clusterização sobre a topologia reduzida pelo SOM, não "
            "diretamente sobre os pixels.",
            "O número de regimes (k) é escolhido por inspeção do cotovelo "
            "da soma de quadrados intra-cluster (WSS), não fixado a "
            "priori.",
        ],
    },
]


def _metodologia_tab_layout() -> html.Div:
    """Aba 'Metodologia': resumo sempre visível de cada etapa do pipeline
    (pré-processamento, cálculo, salvamento, plotagem) + botão 'Ver
    detalhes completos' (<details>/<summary> nativos, sem callback) que
    expande a explicação técnica de cada uma."""
    cards = []
    for stage in _METODOLOGIA_STAGES:
        cards.append(html.Div([
            html.Div([
                html.Span(stage["letra"], style={
                    "display": "inline-block", "width": "28px", "height": "28px",
                    "borderRadius": "50%", "background": "#e07b39", "color": "#fff",
                    "textAlign": "center", "lineHeight": "28px", "fontWeight": "700",
                    "marginRight": "10px", "flexShrink": "0",
                }),
                html.H4(stage["titulo"], style={"margin": "0", "flex": "1"}),
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "8px"}),
            html.P(stage["resumo"], style={"margin": "0 0 10px 38px", "color": "#333"}),
            html.Details([
                html.Summary("Ver detalhes completos", style={
                    "cursor": "pointer", "color": "#e07b39", "fontWeight": "600",
                    "fontSize": "13px", "marginLeft": "38px",
                }),
                html.Ul([
                    html.Li(d, style={"marginBottom": "6px"})
                    for d in stage["detalhes"]
                ], style={
                    "margin": "8px 0 0 38px", "paddingLeft": "18px",
                    "fontSize": "13px", "color": "#555", "lineHeight": "1.5",
                }),
            ]),
        ], style=_CARD))
    return html.Div(cards, style={"maxWidth": "900px"})


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
    first_som = "exp_brasil_gt001" if "exp_brasil_gt001" in som_index else (
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
                        {"label": "Intensidade Vencedora", "value": "diff"},
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
            html.Div([
                html.Label("Threshold R²",
                           style={"fontWeight": "500", "marginBottom": "4px"}),
                dcc.Slider(
                    id="sl-threshold-overview-alt-r2",
                    min=0.0, max=1.0, step=0.05, value=0.0,
                    marks={v: f"{v:.2g}" for v in [0, 0.2, 0.4, 0.6, 0.8, 1.0]},
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
            ], style={"flex": "1", "minWidth": "300px", "padding": "0 8px"}),
            html.Div([
                html.Label("Threshold PREDEP",
                           style={"fontWeight": "500", "marginBottom": "4px"}),
                dcc.Slider(
                    id="sl-threshold-overview-alt-predep",
                    min=0.0, max=1.0, step=0.05, value=0.0,
                    marks={v: f"{v:.2g}" for v in [0, 0.2, 0.4, 0.6, 0.8, 1.0]},
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
            ], style={"flex": "1", "minWidth": "300px", "padding": "0 8px"}),
        ], style={"display": "flex", "flexWrap": "wrap", "marginBottom": "16px"}),
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
            dcc.Tab(label="Metodologia", children=[
                html.Div(
                    _metodologia_tab_layout(),
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
            # Intensidade do vencedor + qual métrica venceu por pixel
            r2_same = layers["r2_albest"]
            alpha_v = layers["alpha"]
            _a = np.nan_to_num(alpha_v, nan=0.0)
            _r = np.nan_to_num(r2_same, nan=0.0)
            both_nan = np.isnan(alpha_v) & np.isnan(r2_same)
            winner_val = np.where(both_nan, np.nan, np.abs(_a - _r))
            # Categoria: 0 = R² venceu, 1 = PREDEP venceu
            low = (_a < 0.1) & (_r < 0.1)
            winner_cat = np.where(both_nan | low, np.nan,
                                  np.where(_a >= _r, 1.0, 0.0))
            winner_cd = np.empty(winner_cat.shape, dtype=object)
            winner_cd[winner_cat == 1] = "PREDEP"
            winner_cd[winner_cat == 0] = "R²"
            cat_scale = [
                [0.0, "#d62728"], [0.5, "#d62728"],
                [0.5, "#1f77b4"], [1.0, "#1f77b4"],
            ]
            _fp = np.where(~np.isnan(winner_val), 0.0, np.nan)
            fig_map = make_subplots(
                rows=1, cols=2,
                subplot_titles=[
                    f"Margem de Vitória — |PREDEP − R²|  ({lag_txt})",
                    f"Métrica Vencedora  ({lag_txt})",
                ],
                horizontal_spacing=0.16,
            )
            # Col 1 — intensidade (escala laranja)
            fig_map.add_trace(go.Heatmap(
                x=lons, y=lats, z=_fp,
                colorscale=[[0, "#bfbfbf"], [1, "#bfbfbf"]],
                showscale=False, hoverinfo="skip",
            ), row=1, col=1)
            fig_map.add_trace(go.Heatmap(
                x=lons, y=lats, z=winner_val,
                colorscale=alpha_scale, zmin=0.0, zmax=1.0,
                colorbar=dict(
                    title="Margem", thickness=12, len=0.85,
                    x=0.46, xanchor="left",
                ),
                hovertemplate=(
                    "Lon: %{x:.3f}<br>Lat: %{y:.3f}"
                    "<br>|PREDEP−R²|: %{z:.4f}<extra></extra>"
                ),
            ), row=1, col=1)
            # Col 2 — categoria (R² vs PREDEP)
            fig_map.add_trace(go.Heatmap(
                x=lons, y=lats, z=_fp,
                colorscale=[[0, "#bfbfbf"], [1, "#bfbfbf"]],
                showscale=False, hoverinfo="skip",
            ), row=1, col=2)
            fig_map.add_trace(go.Heatmap(
                x=lons, y=lats, z=winner_cat,
                customdata=winner_cd,
                colorscale=cat_scale, zmin=-0.5, zmax=1.5,
                colorbar=dict(
                    title="Métrica", thickness=12, len=0.85,
                    tickvals=[0, 1], ticktext=["R²", "PREDEP"],
                    x=1.02, xanchor="left",
                ),
                hovertemplate=(
                    "Lon: %{x:.3f}<br>Lat: %{y:.3f}"
                    "<br>Vencedora: %{customdata}<extra></extra>"
                ),
            ), row=1, col=2)
            map_title = f"Margem de Vitória — {mov_map} | {season_used} | {basin}"
            fig_map.update_layout(
                title=dict(text=map_title, font=dict(size=13), x=0),
                xaxis=dict(showgrid=False, scaleanchor="y", scaleratio=1),
                yaxis=dict(showgrid=False),
                xaxis2=dict(showgrid=False, scaleanchor="y2", scaleratio=1),
                yaxis2=dict(showgrid=False),
                margin=dict(l=60, r=80, t=60, b=50),
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


def _build_k_diagnostics_chart(k_diag: dict):
    """Mini-gráfico da curva de silhouette (K=2..15, média±desvio entre
    sementes) usada para escolher o nº de regimes — permite auditar se o K
    escolhido venceu com folga ou empatou tecnicamente com outro K (nesse
    caso o maior K empatado é escolhido; ver k_margin_note)."""
    if not k_diag or not k_diag.get("k_values"):
        return None
    k_values = k_diag["k_values"]
    means = k_diag.get("silhouette_means") or []
    stds = k_diag.get("silhouette_stds") or [0] * len(means)
    chosen_k = k_diag.get("chosen_k")
    tied = set(k_diag.get("k_candidates_tied") or [])

    colors = ["#d62728" if k == chosen_k else
              ("#ff9896" if k in tied else "#8899aa") for k in k_values]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=k_values, y=means, mode="markers+lines",
        error_y=dict(type="data", array=stds, visible=True, color="#ccc", thickness=1),
        marker=dict(color=colors, size=7), line=dict(color="#ccc", width=1),
        hovertemplate="K=%{x}<br>silhouette=%{y:.4f}<extra></extra>",
    ))
    fig.update_layout(
        height=110, margin=dict(l=30, r=8, t=4, b=20),
        xaxis=dict(title=None, tickmode="linear", tick0=k_values[0], dtick=1,
                   tickfont=dict(size=8)),
        yaxis=dict(title=None, tickfont=dict(size=8)),
        showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return html.Div([
        dcc.Graph(figure=fig, config={"displayModeBar": False},
                  style={"height": "110px"}),
        html.P(k_diag.get("k_margin_note", ""), style={
            "fontSize": "8px", "color": "#888", "margin": "0 0 4px 2px",
            "fontStyle": "italic",
        }),
    ])


def _build_regime_legend(regimes_meta: list) -> html.Div:
    """Painel abaixo do mapa: 1 card por regime (grid multi-coluna), com a
    lista COMPLETA de MoVs do combo (sem truncar — isso fica só no tick da
    barra de cor, que tem pouco espaço)."""
    has_lag_info = any(rm.get("combo_best_lag") for rm in regimes_meta)
    cards = []
    for rm in regimes_meta:
        combo      = rm.get("combo", [])
        vals       = rm.get("combo_alpha")      or [None] * len(combo)
        lags       = rm.get("combo_best_lag")   or [None] * len(combo)
        seasons    = rm.get("combo_best_season") or [None] * len(combo)
        dom        = rm.get("dominant_season")
        dom_movs   = rm.get("dominant_movs") or []
        dom_vals   = rm.get("dominant_values") or [None] * len(dom_movs)

        # Chips em destaque = MoVs DOMINANTES (maior valor bruto/absoluto
        # aqui) — é o que de fato mais prediz nesse regime, mesmo que seja
        # parecido em todos os outros (ex.: um índice ENSO forte no Brasil
        # inteiro). O "distintivo" (anomalia vs. outros regimes) vira nota
        # secundária: útil para ver o que DIFERE entre regiões, mas pode
        # ser um valor quase-ruído quando o sinal dominante é uniforme.
        chips: list = []
        for mv, val in zip(dom_movs, dom_vals):
            val_str = f" {val:.3f}" if val is not None else ""
            chips.append(html.Span([
                html.B(mv, style={"fontSize": "9px"}),
                html.Span(val_str, style={"color": "#888", "fontSize": "8px"}),
            ], style={
                "background": "#fff", "border": "1px solid #e2e4e8",
                "borderRadius": "3px", "padding": "1px 4px",
                "marginRight": "3px", "marginBottom": "2px",
                "display": "inline-block",
            }))

        combo_parts = []
        for mv, lg, ss, val in zip(combo, lags, seasons, vals):
            val_str = f"{val:.3f}" if val is not None else ""
            extra = f" ({val_str}, {lg}m, {ss})" if lg is not None and ss is not None \
                else (f" ({val_str})" if val_str else "")
            combo_parts.append(f"{mv}{extra}")

        cards.append(html.Div([
            html.Div([
                html.Span(style={
                    "display": "inline-block", "width": "9px", "height": "9px",
                    "background": rm["color_hex"], "borderRadius": "2px",
                    "verticalAlign": "middle", "marginRight": "4px",
                }),
                html.Span(f"R{rm['id']}", style={
                    "fontWeight": "700", "fontSize": "10px", "marginRight": "6px",
                }),
                html.Span(dom or "—", style={
                    "background": "#e0e8f8", "borderRadius": "3px",
                    "padding": "0px 4px", "fontSize": "8px",
                    "fontWeight": "600", "letterSpacing": "0.02em",
                    "marginRight": "6px",
                }),
                html.Span(f"n={rm['size']:,}", style={
                    "color": "#999", "fontSize": "8px",
                }),
            ], style={"marginBottom": "3px"}),
            html.Div(chips, style={"lineHeight": "1.5"}),
            (html.Div([
                html.Span("distintivo: ", style={
                    "color": "#aaa", "fontSize": "8px",
                }),
                html.Span(
                    ", ".join(combo_parts),
                    style={"color": "#666", "fontSize": "8px", "fontStyle": "italic"},
                ),
            ], style={"marginTop": "2px"}) if combo_parts else None),
        ], style={
            "background": "#fff", "border": "1px solid #e8e9ec",
            "borderRadius": "5px", "padding": "5px 7px",
        }, title=(
            "Chips em destaque = MoVs DOMINANTES (maior valor absoluto neste "
            "regime), mesmo que sejam parecidos em todos os regimes (ex.: um "
            "índice ENSO forte no Brasil inteiro). 'Distintivo' = MoV com "
            "maior DIFERENÇA vs. média dos outros regimes — pode ser um "
            "valor pequeno/quase-ruído quando o sinal dominante é uniforme "
            "entre regiões, mas indica o que muda de um regime pro outro."
        )))

    note = (
        "" if has_lag_info
        else " (regenere o parquet com som_to_parquet.py para ver lag e trimestre)"
    )
    return html.Div([
        html.P(
            f"Detalhes por regime{note}:",
            style={"fontSize": "10px", "color": "#666", "margin": "0 0 4px 0"},
        ),
        html.Div(cards, style={
            "display": "grid",
            "gridTemplateColumns": "repeat(auto-fill, minmax(190px, 1fr))",
            "gap": "5px",
        }),
    ], style={
        "background": "#f7f8fa",
        "borderRadius": "5px",
        "padding": "6px 10px",
        "marginTop": "4px",
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


_SOM_THRESHOLD_RUN_ID = {
    "gt001": "exp_brasil_gt001",
    "gt005": "exp_brasil_gt005",
    "gt01":  "exp_brasil_gt01",
    "gt015": "exp_brasil_gt015",
}
_SOM_THRESHOLD_CMD = {
    "gt001": "--mov-alpha-threshold 0.01 --run-id exp_brasil_gt001",
    "gt005": "--mov-alpha-threshold 0.05 --run-id exp_brasil_gt005",
    "gt01":  "--mov-alpha-threshold 0.1 --run-id exp_brasil_gt01",
    "gt015": "--mov-alpha-threshold 0.15 --run-id exp_brasil_gt015",
}


def _som_panel_pivot(meta_metric, tag, view):
    """(colorscale, cmin, cmax, cbar, hov, reverse) para uma métrica (predep/r2)."""
    reverse = False
    if view == "component":
        colorscale = "ylorrd"
        cmin, cmax = meta_metric["value_min"], meta_metric["value_max"]
        lbl = "α médio (lag×estação)" if tag == "predep" else "r² médio (lag×estação)"
        cbar = dict(title=lbl, thickness=14)
        hov = lbl
    elif view == "atypicality":
        colorscale, reverse = "magma", True
        cmin, cmax = meta_metric["atypicality_min"], meta_metric["atypicality_max"]
        cbar = dict(title="atipicidade", thickness=14)
        hov = "atipicidade"
    elif view == "boundary":
        colorscale, reverse = "ice", True
        cmin, cmax = meta_metric["boundary_min"], meta_metric["boundary_max"]
        cbar = dict(title="U-matrix", thickness=14)
        hov = "fronteira"
    else:  # regime
        n = meta_metric["n_regimes"]
        cols_hex = [r["color_hex"] for r in meta_metric["regimes"]]
        colorscale = []
        for i, c in enumerate(cols_hex):
            colorscale += [[i / n, c], [(i + 1) / n, c]]
        cmin, cmax = -0.5, n - 0.5
        # Só "R0, R1, ..." na barra — a lista completa de MoVs (com valor,
        # lag e estação) fica no grid de cards abaixo do mapa.
        cbar = dict(
            title="Regime", thickness=12, tickmode="array",
            tickvals=list(range(n)),
            ticktext=[f"R{r['id']}" for r in meta_metric["regimes"]],
        )
        hov = "regime"
    return colorscale, cmin, cmax, cbar, hov, reverse


@app.callback(
    Output("som-content", "children"),
    Input("ri-som-view", "value"),
    Input("ri-som-threshold", "value"),
)
def cb_som_content(view: str, threshold: str):
    run_id = _SOM_THRESHOLD_RUN_ID.get(threshold or "gt001", "exp_brasil_gt001")
    loaded = load_som(RESULTS_DIR, run_id, n_regimes=7)
    if loaded is None:
        cmd = (
            "modal run src/modal/som_insights_modal.py --exp-n 1 "
            '--lags "0,1,3,6,9,12" --dual --no-upload '
            f"{_SOM_THRESHOLD_CMD.get(threshold or 'all', '')}"
        )
        return html.Div([
            html.P(
                f"Artefato SOM dual não encontrado para '{run_id}'. "
                "Gere no repo irc_predep_bootstrap com:",
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
    df, meta = loaded

    def _piv(col):
        return df.pivot(
            index="latitude", columns="longitude", values=col
        ).sort_index()

    regime_view = (view == "regime")
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["PREDEP (α)", "R² (regressão linear)"],
        horizontal_spacing=0.09,
    )
    legends = []
    for i, tag in enumerate(("predep", "r2")):
        meta_metric = meta[tag]
        if view == "component":
            mov = meta_metric.get("mov_names", meta.get("mov_names", []))[0]
            col = f"{mov}_{tag}"
        else:
            col = f"{tag}_{view}"
        colorscale, cmin, cmax, cbar, hov, reverse = _som_panel_pivot(
            meta_metric, tag, view
        )
        p = _piv(col)
        lons, lats, z = p.columns.values, p.index.values, p.values
        cbar = {**cbar, "x": 0.44 if i == 0 else 1.0, "len": 0.9}
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
        fig.add_trace(go.Heatmap(**heat), row=1, col=i + 1)
        if regime_view:
            k_chart = _build_k_diagnostics_chart(meta_metric.get("k_diagnostics"))
            legends.append(html.Div([
                html.H5("PREDEP" if tag == "predep" else "R²",
                        style={"margin": "8px 0 4px 0", "fontSize": "13px"}),
                k_chart if k_chart is not None else None,
                _build_regime_legend(meta_metric["regimes"]),
            ], style={"flex": "1 1 0"}))

    thr_lbl = {"gt001": "α > 0.01", "gt005": "α > 0.05", "gt01": "α > 0.1",
               "gt015": "α > 0.15"}.get(threshold or "gt001", "α > 0.01")
    n_movs_predep = len(meta["predep"].get("mov_names", meta.get("mov_names", [])))
    n_movs_r2 = len(meta["r2"].get("mov_names", meta.get("mov_names", [])))
    title = (
        f"PREDEP vs R² — {run_id} ({thr_lbl}, "
        f"PREDEP: {n_movs_predep} MoVs | R²: {n_movs_r2} MoVs)"
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=13), x=0),
        xaxis=dict(showgrid=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(showgrid=False),
        xaxis2=dict(showgrid=False, scaleanchor="y2", scaleratio=1),
        yaxis2=dict(showgrid=False),
        margin=dict(l=60, r=60, t=70, b=50),
        height=750, dragmode="zoom",
    )
    graph = dcc.Graph(
        figure=fig,
        config={
            "scrollZoom": True, "displayModeBar": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            "toImageButtonOptions": {"filename": title},
        },
        style={"width": "100%", "marginBottom": "12px", "height": "750px"},
    )
    if regime_view:
        return html.Div([
            graph,
            html.Div(legends, style={"display": "flex", "gap": "16px",
                                     "flexWrap": "wrap"}),
        ])
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

    # Range explícito (mesmo padrão de _single_map_graph) — sem isso, o
    # autorange do Plotly em subplots com scaleanchor pode sobrar espaço em
    # branco enorme ao redor do mapa e não se realinha bem ao redimensionar.
    lon_min, lon_max = float(np.nanmin(lons)), float(np.nanmax(lons))
    lat_min, lat_max = float(np.nanmin(lats)), float(np.nanmax(lats))
    lon_pad = (lon_max - lon_min) * 0.02 or 0.5
    lat_pad = (lat_max - lat_min) * 0.02 or 0.5
    _map_xrange = [lon_min - lon_pad, lon_max + lon_pad]
    _map_yrange = [lat_min - lat_pad, lat_max + lat_pad]

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

    # Pré-computar grids de nomes de MoV para customdata dos mapas
    _ov_mov_r2_names = np.empty(layers["mov_r2"].shape, dtype=object)
    _ov_mov_al_names = np.empty(layers["mov_alpha"].shape, dtype=object)
    for _oi, _on in enumerate(mov_names):
        _ov_mov_r2_names[layers["mov_r2"] == _oi] = _on
        _ov_mov_al_names[layers["mov_alpha"] == _oi] = _on

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
    for col, (z_data, lbl, other_z, mov_names_arr, lag_arr, other_lbl) in enumerate(
        [
            (layers["r2"],    "R²",    layers["alpha"], _ov_mov_r2_names, layers["lag_r2"],    "PREDEP"),
            (layers["alpha"], "PREDEP", layers["r2"],   _ov_mov_al_names, layers["lag_alpha"], "R²"),
        ],
        start=1,
    ):
        fig.add_trace(go.Heatmap(
            x=lons, y=lats, z=footprint_gray,
            colorscale=_gray, showscale=False, hoverinfo="skip",
        ), row=1, col=col)
        _cd1 = np.dstack([other_z.astype(object), mov_names_arr, lag_arr.astype(object)])
        fig.add_trace(go.Heatmap(
            x=lons, y=lats, z=z_data,
            colorscale=alpha_scale, zmin=0.0, zmax=1.0,
            colorbar=dict(title="Valor", thickness=14, y=0.866, len=0.26) if col == 2 else None,
            showscale=(col == 2),
            customdata=_cd1,
            hovertemplate=(
                "Lon: %{x:.3f}<br>Lat: %{y:.3f}"
                f"<br>{lbl}: %{{z:.4f}}"
                f"<br>{other_lbl}: %{{customdata[0]:.4f}}"
                "<br>MoV: %{customdata[1]}"
                "<br>Lag: %{customdata[2]:.0f}m"
                "<extra></extra>"
            ),
        ), row=1, col=col)

    # Row 2: MoV Vencedor
    for col, (z_data, lbl, val_layer, lag_layer, mov_names_arr) in enumerate(
        [
            (layers["mov_r2"],    "R²",    layers["r2"],    layers["lag_r2"],    _ov_mov_r2_names),
            (layers["mov_alpha"], "PREDEP", layers["alpha"], layers["lag_alpha"], _ov_mov_al_names),
        ],
        start=1,
    ):
        fig.add_trace(go.Heatmap(
            x=lons, y=lats, z=footprint_gray,
            colorscale=_gray, showscale=False, hoverinfo="skip",
        ), row=2, col=col)

        _cd2 = np.dstack([mov_names_arr, val_layer.astype(object), lag_layer.astype(object)])

        fig.add_trace(go.Heatmap(
            x=lons, y=lats, z=z_data,
            customdata=_cd2,
            colorscale=mov_scale, zmin=0, zmax=n_movs,
            colorbar=dict(
                title="MoV", thickness=14,
                tickvals=mov_tickvals, ticktext=mov_ticktext,
                y=0.5, len=0.26,
            ) if col == 2 else None,
            showscale=(col == 2),
            hovertemplate=(
                "Lon: %{x:.3f}<br>Lat: %{y:.3f}"
                f"<br>MoV Vencedor ({lbl}): %{{customdata[0]}}"
                f"<br>{lbl}: %{{customdata[1]:.4f}}"
                "<br>Lag: %{customdata[2]:.0f}m"
                "<extra></extra>"
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

    for col, (z_data, lbl, val_layer, mov_names_arr) in enumerate(
        [
            (layers["lag_r2"],    "R²",    layers["r2"],    _ov_mov_r2_names),
            (layers["lag_alpha"], "PREDEP", layers["alpha"], _ov_mov_al_names),
        ],
        start=1,
    ):
        fig.add_trace(go.Heatmap(
            x=lons, y=lats, z=footprint_gray,
            colorscale=_gray, showscale=False, hoverinfo="skip",
        ), row=3, col=col)

        _cd3 = np.dstack([val_layer.astype(object), mov_names_arr])

        fig.add_trace(go.Heatmap(
            x=lons, y=lats, z=z_data,
            colorscale=lag_scale, zmin=lmin, zmax=lmax,
            colorbar=dict(
                title="Lag (meses)", thickness=14,
                tickvals=all_lags,
                y=0.133, len=0.26,
            ) if col == 2 else None,
            showscale=(col == 2),
            customdata=_cd3,
            hovertemplate=(
                "Lon: %{x:.3f}<br>Lat: %{y:.3f}"
                f"<br>Lag Ótimo ({lbl}): %{{z:.0f}}m"
                f"<br>{lbl}: %{{customdata[0]:.4f}}"
                "<br>MoV: %{customdata[1]}"
                "<extra></extra>"
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
                f"xaxis{suffix}": dict(
                    showgrid=False, scaleanchor=f"y{suffix}", scaleratio=1,
                    constrain="domain", range=_map_xrange,
                ),
                f"yaxis{suffix}": dict(showgrid=False, range=_map_yrange),
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
        xaxis=dict(
            showgrid=False, scaleanchor="y", scaleratio=1,
            constrain="domain", range=_map_xrange,
        ),
        yaxis=dict(showgrid=False, range=_map_yrange),
        margin=dict(l=60, r=60, t=60, b=50),
        height=520, dragmode="zoom",
    )

    return html.Div([
        dcc.Graph(
            figure=fig,
            config={
                "displayModeBar": True,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                "toImageButtonOptions": {
                    "filename": f"overview_global_{season}_{basin}"
                },
            },
            style={"width": "100%", "minHeight": "1400px"},
        ),
        html.Hr(style={"margin": "40px 0"}),
        dcc.Graph(
            figure=fig_diff,
            config={
                "displayModeBar": True,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                "toImageButtonOptions": {
                    "filename": f"metrica_vencedora_{season}_{basin}"
                },
            },
            style={"width": "100%", "minHeight": "520px"},
        ),
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
    return dcc.Graph(
        figure=fig,
        config={
            "displayModeBar": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            "toImageButtonOptions": {"filename": titles[metric]},
        },
    )


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
        config={
            "scrollZoom": True,
            "displayModeBar": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            "toImageButtonOptions": {"filename": title},
        },
        style={"flex": "1", "minWidth": "300px", "height": f"{height}px"},
    )


@app.callback(
    Output("overview-alt-perlags-content", "children"),
    Input("ri-season-overview-alt", "value"),
    Input("sl-threshold-overview-alt-r2", "value"),
    Input("sl-threshold-overview-alt-predep", "value"),
)
def cb_overview_alt_perlags(season: str, threshold_r2: float, threshold_predep: float):
    per_lag = _get_brasil_per_lag(season or "Todas")
    if not per_lag:
        return html.P("Nenhum dado encontrado.")

    thr_r2 = threshold_r2 or 0.0
    thr_al = threshold_predep or 0.0

    rows = []
    for lag in sorted(per_lag.keys()):
        d = per_lag[lag]
        r2_orig = d["r2"]
        al_orig = d["alpha"]

        if thr_r2 > 0:
            r2_z = np.where(np.nan_to_num(r2_orig, nan=0.0) < thr_r2, np.nan, r2_orig)
        else:
            r2_z = r2_orig
            r2_orig = None  # sem footprint separado necessário

        if thr_al > 0:
            al_z = np.where(np.nan_to_num(al_orig, nan=0.0) < thr_al, np.nan, al_orig)
        else:
            al_z = al_orig
            al_orig = None  # sem footprint separado necessário

        lag_lbl = f"lag = {lag} {'mês' if lag == 1 else 'meses'}"
        rows.append(html.Div([
            _single_map_graph(d["lons"], d["lats"], r2_z,
                              f"R²  —  {lag_lbl}", cb_title="R²", height=600,
                              gray_bg=True, z_footprint=r2_orig,
                              customdata=d["alpha"],
                              hovertemplate=(
                                  "Lon: %{x:.3f}<br>Lat: %{y:.3f}"
                                  "<br>R²: %{z:.4f}"
                                  "<br>PREDEP: %{customdata:.4f}"
                                  "<extra></extra>"
                              )),
            _single_map_graph(d["lons"], d["lats"], al_z,
                              f"PREDEP  —  {lag_lbl}", cb_title="PREDEP", height=600,
                              gray_bg=True, z_footprint=al_orig,
                              customdata=d["r2"],
                              hovertemplate=(
                                  "Lon: %{x:.3f}<br>Lat: %{y:.3f}"
                                  "<br>PREDEP: %{z:.4f}"
                                  "<br>R²: %{customdata:.4f}"
                                  "<extra></extra>"
                              )),
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

    # Linha 1: valores máximos — hover cross-métrica + MoV
    mov_r2_names = np.empty(layers["mov_r2"].shape, dtype=object)
    mov_al_names = np.empty(layers["mov_alpha"].shape, dtype=object)
    for _i, _n in enumerate(mov_names):
        mov_r2_names[layers["mov_r2"] == _i] = _n
        mov_al_names[layers["mov_alpha"] == _i] = _n

    cd_r2 = np.dstack([layers["alpha"].astype(object), mov_r2_names])
    cd_al = np.dstack([layers["r2"].astype(object), mov_al_names])

    row1 = html.Div([
        _single_map_graph(lons, lats, layers["r2"],
                          f"R² Máximo (lag=0) — {pfx}", cb_title="R²",
                          customdata=cd_r2,
                          hovertemplate=(
                              "Lon: %{x:.3f}<br>Lat: %{y:.3f}"
                              "<br>R²: %{z:.4f}"
                              "<br>PREDEP: %{customdata[0]:.4f}"
                              "<br>MoV: %{customdata[1]}"
                              "<extra></extra>"
                          )),
        _single_map_graph(lons, lats, layers["alpha"],
                          f"PREDEP Máximo (lag=0) — {pfx}", cb_title="PREDEP",
                          customdata=cd_al,
                          hovertemplate=(
                              "Lon: %{x:.3f}<br>Lat: %{y:.3f}"
                              "<br>PREDEP: %{z:.4f}"
                              "<br>R²: %{customdata[0]:.4f}"
                              "<br>MoV: %{customdata[1]}"
                              "<extra></extra>"
                          )),
    ], style={"display": "flex", "gap": "8px", "marginBottom": "8px"})

    # Linha 2: MoV vencedor — hover com valor da métrica
    def _mov_graph(z_data, lbl, val_layer):
        cd = np.empty(z_data.shape, dtype=object)
        for i, name in enumerate(mov_names):
            cd[z_data == i] = name
        cd_full = np.dstack([cd, val_layer.astype(object)])
        return _single_map_graph(
            lons, lats, z_data,
            title=f"MoV Vencedor ({lbl}, lag=0) — {pfx}",
            colorscale=mov_scale, zmin=0, zmax=n_movs,
            cb_title="MoV",
            cb_extra={"tickvals": mov_tickvals, "ticktext": mov_names},
            gray_bg=True, customdata=cd_full,
            hovertemplate=(
                f"Lon: %{{x:.3f}}<br>Lat: %{{y:.3f}}"
                f"<br>MoV Vencedor ({lbl}): %{{customdata[0]}}"
                f"<br>{lbl}: %{{customdata[1]:.4f}}"
                "<extra></extra>"
            ),
        )

    row2 = html.Div([
        _mov_graph(layers["mov_r2"], "R²", layers["r2"]),
        _mov_graph(layers["mov_alpha"], "PREDEP", layers["alpha"]),
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
        for lay, z_key, other_key, lbl, other_lbl, mov_lbl in [
            (layers_r2,    "r2",    "alpha", "R²",    "PREDEP", mov_r2),
            (layers_alpha, "alpha", "r2",    "PREDEP", "R²",    mov_alpha),
        ]:
            if lay is None:
                graphs.append(html.Div(style={"flex": "1"}))
                continue
            lons, lats = lay["lons"], lay["lats"]
            z = lay[z_key]
            z_other = lay[other_key]
            if basin == "Brasil" and len(lats) > 50 and len(lons) > 50:
                lons, lats, z = lons[::2], lats[::2], z[::2, ::2]
                z_other = z_other[::2, ::2]
            graphs.append(_single_map_graph(
                lons, lats, z,
                title=f"{lbl} — {mov_lbl}  lag={lag}",
                cb_title=lbl,
                customdata=z_other,
                hovertemplate=(
                    f"Lon: %{{x:.3f}}<br>Lat: %{{y:.3f}}"
                    f"<br>{lbl}: %{{z:.4f}}"
                    f"<br>{other_lbl}: %{{customdata:.4f}}"
                    "<extra></extra>"
                ),
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
