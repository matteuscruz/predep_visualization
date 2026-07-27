from __future__ import annotations

from .conftest import write_min_png


def test_scan_results_finds_synthetic_parquet(app_mod, synthetic_results_dir):
    index = app_mod.scan_results(synthetic_results_dir)
    assert index["exp_test01"]["mov_test"]["bacia_teste"] == (
        synthetic_results_dir
        / "predep_granular_brazil"
        / "exp_test01"
        / "mov_test"
    )


def test_scan_results_empty_when_dir_missing(app_mod, tmp_path):
    index = app_mod.scan_results(tmp_path / "no_results")
    assert index == {}


def test_scan_plots_bacias(app_mod, synthetic_plots_dir):
    index = app_mod.scan_plots(synthetic_plots_dir)
    entry = index["mov_test"]["bacia_teste"]["PREDEP"]["Lag ótimo"]["exp_test01"]
    assert entry.name == "predep_optimal.png"


def test_scan_plots_brazil_suppressed_when_not_in_valid_brasil(
    app_mod, synthetic_plots_dir, monkeypatch
):
    # exp_test01/mov_test isn't a member of the real _valid_brasil set, so the
    # BRAZIL-wide plot must be suppressed unless the global is explicitly relaxed.
    monkeypatch.setattr(app_mod, "_valid_brasil", {("some_other_exp", "some_mov")})
    index = app_mod.scan_plots(synthetic_plots_dir)
    assert "Brasil" not in index.get("mov_test", {})


def test_scan_plots_brazil_included_when_valid_brasil_empty(
    app_mod, synthetic_plots_dir, monkeypatch
):
    monkeypatch.setattr(app_mod, "_valid_brasil", set())
    index = app_mod.scan_plots(synthetic_plots_dir)
    entry = index["mov_test"]["Brasil"]["PREDEP"]["Lag ótimo"]["exp_test01"]
    assert entry.name == "predep_optimal.png"


def test_scan_best_mov_requires_at_least_two_movs(app_mod, tmp_path):
    plots_dir = tmp_path / "plots"
    exp_dir = plots_dir / "exp_test01"
    write_min_png(exp_dir / "best_mov" / "best_mov_lag_0.png")
    write_min_png(exp_dir / "mov_a" / "PREDEP" / "BRAZIL" / "x.png")

    # Only one MoV dir besides best_mov/ -> should be skipped.
    assert app_mod.scan_best_mov(plots_dir) == {}

    write_min_png(exp_dir / "mov_b" / "PREDEP" / "BRAZIL" / "x.png")
    index = app_mod.scan_best_mov(plots_dir)
    assert index["lag 0"]["exp_test01"].name == "best_mov_lag_0.png"


def test_scan_som_new_layout(app_mod, tmp_path):
    results_dir = tmp_path / "results"
    som_dir = results_dir / "predep_som" / "exp_test01" / "n07"
    som_dir.mkdir(parents=True)
    (som_dir / "som_pixels.parquet").write_bytes(b"")
    (som_dir / "som_meta.json").write_text("{}")

    index = app_mod.scan_som(results_dir)
    assert index["exp_test01"]["n_regimes"] == [7]


def test_scan_som_legacy_flat_layout(app_mod, tmp_path):
    results_dir = tmp_path / "results"
    exp_dir = results_dir / "predep_som" / "exp_test01"
    exp_dir.mkdir(parents=True)
    (exp_dir / "som_pixels.parquet").write_bytes(b"")
    (exp_dir / "som_meta.json").write_text("{}")

    index = app_mod.scan_som(results_dir)
    assert index["exp_test01"] == {"n_regimes": [7], "_flat": True}


def test_compute_valid_brasil_requires_two_basins(app_mod, tmp_path):
    results_dir = tmp_path / "results"
    mov_dir = results_dir / "predep_granular_brazil" / "exp_test01" / "mov_test"
    mov_dir.mkdir(parents=True)
    (mov_dir / "mov_test_predep_granular_seasonal.nc").write_bytes(b"")
    (mov_dir / "bacia_a_predep_granular_seasonal.nc").write_bytes(b"")

    # Only one basin besides the Brasil-wide file -> not valid yet.
    assert app_mod.compute_valid_brasil(results_dir) == set()

    (mov_dir / "bacia_b_predep_granular_seasonal.nc").write_bytes(b"")
    assert app_mod.compute_valid_brasil(results_dir) == {("exp_test01", "mov_test")}
