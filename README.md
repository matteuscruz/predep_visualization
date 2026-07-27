# PREDEP Visualization

Viewer interativo (Dash/Plotly) para explorar os resultados do projeto **PREDEP** — sinais de
previsibilidade de precipitação associados a diferentes Modos de Variabilidade (MoV: ENSO, AMO,
TNA, ATL3, ONI, AAO, etc.) sobre as bacias hidrográficas do Brasil.

A aplicação lê:
- **plots/** — imagens PNG já geradas pelo pipeline de análise (mapas de PREDEP, regressão e
  comparação, por MoV/bacia/experimento);
- **results/** — arquivos Parquet (e, em experimentos legados, NetCDF `.nc`) com os dados
  granulares, usados para o mapa interativo pixel a pixel;
- **data/clusters/** — shapefiles com os contornos das bacias hidrográficas.

Nenhum desses dados é gerado pelo próprio viewer — eles vêm de um pipeline de processamento externo.
`plots/`, os Parquet de `results/` e `data/clusters/` já ficam versionados neste repositório, então
um `git clone`/`git pull` traz tudo que a aplicação precisa para rodar, sem passos manuais de cópia.
Só os `.nc` (formato antigo, substituído por Parquet) continuam fora do git por serem pesados — veja
`.gitignore`.

## Estrutura esperada

```
predep_visualization/
├── app.py
├── requirements.txt
├── plots/                          # plots PNG gerados pelo pipeline
│   └── expNN/
│       └── <MOV>/
│           ├── PREDEP/
│           ├── REGRESSAO/
│           └── COMPARACAO/
├── results/
│   └── predep_granular_brazil/     # arquivos Parquet (/ .nc legado)
│       └── expNN/
└── data/
    └── clusters/                   # shapefiles das bacias
        ├── amazonica/
        ├── parana/
        └── ...
```

`results/*.nc` e os PNGs residuais dentro de `results/**/plots/` não são versionados no git (veja
`.gitignore`) por serem arquivos pesados/obsoletos. Se algum experimento só existir em NetCDF, copie
o `.nc` para dentro de `results/` (ou aponte `--results-dir` para onde ele está).

## Pré-requisitos

- Python 3.10+
- pip

## Configuração local

1. Clone o repositório e entre na pasta:
   ```bash
   git clone <url-do-repo>
   cd predep_visualization
   ```

2. (Recomendado) crie um ambiente virtual:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Instale as dependências (versões fixadas — usar exatamente estas evita divergência de
   comportamento entre máquinas):
   ```bash
   pip install -r requirements.txt
   ```

`plots/`, `results/` (Parquet) e `data/clusters/` já vêm populados pelo próprio `git clone`/`git
pull` — nenhum passo adicional de cópia de dados é necessário para rodar localmente.

## Execução

```bash
python app.py
```

Por padrão sobe em `http://localhost:8050`.

Opções úteis:

```bash
python app.py --port 8051
python app.py --host 127.0.0.1
python app.py --plots-dir /caminho/para/plots --results-dir /caminho/para/results
```

| Flag | Padrão | Descrição |
|------|--------|-----------|
| `--port` | `8050` (ou env `PORT`) | Porta HTTP |
| `--host` | `0.0.0.0` (ou env `HOST`) | Endereço de bind |
| `--plots-dir` | `./plots` | Diretório com os PNGs |
| `--results-dir` | `./results` | Diretório com os NetCDF/Parquet |

Acesse a URL impressa no terminal (ex.: `http://localhost:8050`) no navegador.

## Deploy no Render

1) No Render, crie um **Web Service** a partir do repo do GitHub.
2) Configure:

```text
Build Command: pip install -r requirements.txt
Start Command: python app.py
```

3) O Render define a porta via a variável `PORT` automaticamente.
4) Se precisar, ajuste `--plots-dir` e `--results-dir` no Start Command.

## Abas disponíveis

| Aba | Descrição |
|-----|-----------|
| **Overview** | Mapa por bacia/season, com slider de threshold, mostrando R² e PREDEP (máximo, MoV vencedor e lag ótimo) por pixel |
| **Exploração** | Mapa interativo pixel a pixel (R²+PREDEP, lag ótimo ou intensidade vencedora) + tabela de estatísticas por MoV |
| **Lag 0** | Mapa do MoV vencedor por pixel fixando o lag em 0 |
| **Lag's** | Comparação por lag das famílias R² e PREDEP, com threshold — mostra máximo, MoV vencedor e lag ótimo lado a lado |
| **MoV Vencedor** | Comparação, por lag, entre o MoV com melhor R² e o MoV com melhor PREDEP (podem ser MoVs diferentes) |
| **SOM** | Regimes de previsibilidade obtidos via Self-Organizing Map, treinado de forma geral ou por season (DJF/MAM/JJA/SON) |
