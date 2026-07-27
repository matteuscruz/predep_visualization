from __future__ import annotations


def test_app_layout_builds_from_real_data(app_mod):
    assert app_mod.app.layout is not None


def test_index_route_returns_ok(flask_client):
    resp = flask_client.get("/")
    assert resp.status_code == 200
    assert b"PREDEP Viewer" in resp.data


def test_results_index_not_empty_for_real_data(app_mod):
    index = app_mod._get_results_index(app_mod.RESULTS_DIR)
    assert index, "results/predep_granular_brazil parece vazio ou corrompido"


def test_compute_valid_brasil_runs_on_real_data(app_mod):
    # Dados reais do repo hoje são só Parquet (o .nc é o formato legado que este
    # helper inspeciona), então um set vazio é esperado e não indica quebra —
    # scan_plots só filtra quando o set vem não-vazio (ver app.py:96).
    valid = app_mod.compute_valid_brasil(app_mod.RESULTS_DIR)
    assert isinstance(valid, set)


def test_serve_plot_route_for_a_real_png(app_mod, flask_client):
    plots_index = app_mod.scan_plots(app_mod.PLOTS_DIR)
    real_png = None
    for movs in plots_index.values():
        for areas in movs.values():
            for viz_map in areas.values():
                for exp_map in viz_map.values():
                    real_png = next(iter(exp_map.values()), None)
                    if real_png:
                        break
                if real_png:
                    break
            if real_png:
                break
        if real_png:
            break
    assert real_png is not None, "nenhum PNG encontrado em plots/ para testar a rota"

    rel_path = real_png.relative_to(app_mod.PLOTS_DIR).as_posix()
    resp = flask_client.get(f"/plots/{rel_path}")
    assert resp.status_code == 200
    assert resp.content_type == "image/png"
