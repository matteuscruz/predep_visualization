"""Garante que cada aba do dashboard carrega dados reais (não cai no branch de
"nenhum dado encontrado"/"selecione..."). Chama os callbacks do Dash como
funções Python normais — são apenas `def`s decorados, dá pra chamar direto
sem precisar de um app rodando ou de um browser.
"""
from __future__ import annotations

import pytest

_NO_DATA_MARKERS = (
    "nenhum dado",
    "sem dados",
    "selecione",
    "dados não encontrados",
    "dados nao encontrados",
)


def _assert_real_content(component, where: str):
    assert component is not None, f"{where} retornou None"
    text = getattr(component, "children", None)
    if isinstance(text, str):
        lowered = text.lower()
        for marker in _NO_DATA_MARKERS:
            assert marker not in lowered, (
                f"{where} caiu no fallback de 'sem dados': {text!r}"
            )


@pytest.fixture(scope="module")
def first_exp_basin(app_mod):
    index = app_mod._get_results_index(app_mod.RESULTS_DIR)
    assert index, "results/predep_granular_brazil está vazio — nada pra testar"
    # Mesmo critério de _build_layout (app.py:1394): a UI real só exibe
    # _FIXED_EXP quando ele existe — outros exps podem ser legados/parciais.
    exp = (
        app_mod._FIXED_EXP if app_mod._FIXED_EXP in index
        else sorted(index.keys())[0]
    )
    basins = sorted({
        basin
        for mov_map in index.get(exp, {}).values()
        for basin in mov_map
    })
    assert basins, f"experimento {exp} não tem nenhuma bacia indexada"
    return exp, basins[0]


def test_overview_tab_loads_data(app_mod, first_exp_basin):
    exp, _ = first_exp_basin
    opts, basin = app_mod.cb_clusters_overview(exp)
    assert opts and basin, "aba Overview: dropdown de bacias veio vazio"

    content = app_mod.cb_overview_content(exp, basin, "Todas", 0.0)
    _assert_real_content(content, "aba Overview")


def test_exploracao_tab_loads_data(app_mod, first_exp_basin):
    exp, _ = first_exp_basin
    _, basin = app_mod.cb_clusters_explore(exp)
    assert basin, "aba Exploração: dropdown de bacias veio vazio"

    mov_opts, mov = app_mod.cb_mov_map_opts(exp, basin)
    assert mov_opts and mov, "aba Exploração: dropdown de MoVs veio vazio"

    lag_opts, lag = app_mod.cb_lag_map_opts(exp, basin, mov, "DJF")
    assert lag_opts

    stats_json = app_mod.cb_load_stats(exp, basin)
    assert stats_json, "aba Exploração: stats-store veio vazio"

    content = app_mod.cb_explore_content(
        stats_json, "DJF", mov, lag, "ralpha", exp, basin,
    )
    _assert_real_content(content, "aba Exploração")


def test_lag0_tab_loads_data(app_mod, first_exp_basin):
    exp, _ = first_exp_basin
    opts, basin = app_mod.cb_clusters_lag0(exp)
    assert opts and basin, "aba Lag 0: dropdown de bacias veio vazio"

    content = app_mod.cb_lag0_content(exp, basin, "Todas")
    _assert_real_content(content, "aba Lag 0")


def test_lags_tab_loads_data(app_mod):
    content = app_mod.cb_overview_alt_perlags("Todas", 0.0, 0.0)
    _assert_real_content(content, "aba Lag's")


def test_mov_vencedor_tab_loads_data(app_mod, first_exp_basin):
    exp, _ = first_exp_basin
    opts, basin = app_mod.cb_clusters_destaque(exp)
    assert opts and basin, "aba MoV Vencedor: dropdown de bacias veio vazio"

    r2_opts, mov_r2, alpha_opts, mov_alpha = app_mod.cb_mov_opts_destaque(
        exp, basin, "Todas",
    )
    assert r2_opts and mov_r2, "aba MoV Vencedor: melhor MoV (R²) não calculado"
    assert alpha_opts and mov_alpha, (
        "aba MoV Vencedor: melhor MoV (PREDEP) não calculado"
    )

    content = app_mod.cb_destaque_content(
        exp, basin, "Todas", mov_r2, mov_alpha,
    )
    _assert_real_content(content, "aba MoV Vencedor")


def test_som_tab_loads_data(app_mod):
    content = app_mod.cb_som_content("regime", "gt001")
    _assert_real_content(content, "aba SOM")
