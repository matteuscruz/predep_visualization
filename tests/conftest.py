from __future__ import annotations

import struct
import zlib
from pathlib import Path

import pandas as pd
import pytest

# Importing app.py triggers a real scan of ./results and ./plots and builds the
# full Dash layout at module scope (see app.py:1719-1720) — expensive, so it must
# happen exactly once per test session, not once per test module.
import app as app_module  # noqa: E402


@pytest.fixture(scope="session")
def app_mod():
    return app_module


@pytest.fixture(scope="session")
def flask_client(app_mod):
    app_mod.server.testing = True
    with app_mod.server.test_client() as client:
        yield client


def write_min_png(path: Path) -> None:
    """Writes a minimal valid 1x1 transparent PNG to `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0)
    raw = b"\x00\x00\x00\x00\x00"  # filter byte + 1 RGBA pixel
    idat = zlib.compress(raw)
    png = sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")
    path.write_bytes(png)


def write_min_parquet(path: Path, basin: str, lag: int = 0) -> None:
    """Writes a minimal parquet file with the columns scan/stat helpers expect."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "basin": [basin],
            "latitude": [-10.0],
            "longitude": [-50.0],
            "lag": [lag],
            "r2": [0.5],
            "alpha": [0.5],
        }
    )
    df.to_parquet(path)


@pytest.fixture
def synthetic_results_dir(tmp_path):
    """results/predep_granular_brazil/expX/movY/(Season.parquet dir) with a fake basin."""
    base = tmp_path / "results"
    gran = base / "predep_granular_brazil" / "exp_test01" / "mov_test"
    write_min_parquet(gran / "DJF.parquet", basin="bacia_teste")
    return base


@pytest.fixture
def synthetic_plots_dir(tmp_path):
    """plots/expX/movY/{PREDEP,...}/{BRAZIL,BACIAS}/... PNG tree."""
    base = tmp_path / "plots"
    mov_dir = base / "exp_test01" / "mov_test"
    write_min_png(mov_dir / "PREDEP" / "BACIAS" / "bacia_teste" / "predep_optimal.png")
    write_min_png(mov_dir / "PREDEP" / "BRAZIL" / "predep_optimal.png")
    return base
