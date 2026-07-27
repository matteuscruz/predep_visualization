from __future__ import annotations


def test_viz_label_optimal(app_mod):
    assert app_mod._viz_label("mov_predep_optimal.png") == "Lag ótimo"


def test_viz_label_winner(app_mod):
    assert app_mod._viz_label("mov_predep_winner.png") == "Lag ótimo"


def test_viz_label_default(app_mod):
    assert app_mod._viz_label("mov_predep_all_lags.png") == "Todos os lags"


def test_alpha_colorscale_bounds(app_mod):
    scale = app_mod._alpha_colorscale()
    assert scale[0][0] == 0.0
    assert scale[-1][0] == 1.0
    for pos, color in scale:
        assert 0.0 <= pos <= 1.0
        assert color.startswith("rgb(")


def test_lag_colorscale_empty(app_mod):
    scale = app_mod._lag_colorscale([])
    assert scale[0][0] == 0.0
    assert scale[-1][0] == 1.0


def test_lag_colorscale_single_lag(app_mod):
    scale = app_mod._lag_colorscale([3])
    assert scale == [[0.0, scale[0][1]], [1.0, scale[0][1]]]


def test_lag_colorscale_multiple_lags(app_mod):
    scale = app_mod._lag_colorscale([0, 1, 2, 3])
    assert scale[0][0] == 0.0
    assert scale[-1][0] == 1.0
    positions = [p for p, _ in scale]
    assert positions == sorted(positions)


def test_pretty_basin(app_mod):
    assert app_mod._pretty_basin("bacia_teste") == "Bacia Teste"


def test_basin_opts(app_mod):
    opts = app_mod._basin_opts(["bacia_um", "bacia_dois"])
    assert opts == [
        {"label": "Bacia Um", "value": "bacia_um"},
        {"label": "Bacia Dois", "value": "bacia_dois"},
    ]
