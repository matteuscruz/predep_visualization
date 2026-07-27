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

## Testes

O projeto tem testes unitários (funções puras de formatação/colorscale, funções de
scan de `plots/`/`results/` com dados sintéticos), um smoke test que sobe a app com
os dados reais já versionados no repo e confere as rotas principais via Flask test
client, e — o mais importante — `tests/test_tabs.py`, que exercita o carregamento
de dados de cada uma das 6 abas (Overview, Exploração, Lag 0, Lag's, MoV Vencedor,
SOM) chamando os callbacks do Dash diretamente com o experimento/bacia reais que a
UI usa, garantindo que nenhuma aba caia no fallback de "nenhum dado encontrado".

```bash
pip install -r requirements.txt
pytest -q
```

Um workflow do GitHub Actions (`.github/workflows/ci.yml`) roda esses testes e um
build do Docker a cada push/PR para `main`. Para que PRs com testes quebrados não
possam ser mergeados (e, por consequência, não cheguem ao deploy automático do
Render), configure em **Settings → Branches** do repositório no GitHub uma regra de
proteção para `main` exigindo que os checks `test` e `docker-build` passem antes do
merge.

## Executando via Docker

Alternativa recomendada se você tem tido problemas para rodar o projeto localmente em outras
máquinas (versões de Python, libs de sistema para netCDF4/pyarrow, etc.) — o container isola
tudo isso.

Pré-requisito: Docker instalado (Docker Desktop ou Docker Engine + Compose plugin).

```bash
docker compose up --build
```

Acesse `http://localhost:8050`. Para rodar em segundo plano, use `docker compose up --build -d`
e `docker compose down` para parar.

Sem `docker compose`, também dá pra usar `docker` puro:

```bash
docker build -t predep-viewer .
docker run --rm -p 8050:8050 predep-viewer
```

O `Dockerfile` já inclui `plots/`, `results/` e `data/` (versionados no git) dentro da imagem,
então nenhuma cópia manual de dados é necessária. Se preferir editar esses diretórios sem
reconstruir a imagem a cada mudança, descomente os `volumes` no `docker-compose.yml`.

Dentro do container o app roda com `gunicorn` (produção) em vez do servidor de desenvolvimento
do Dash; porta e host continuam configuráveis via as variáveis de ambiente `PORT`/`HOST`.

## Deploy no Render

1) No Render, crie um **Web Service** a partir do repo do GitHub.
2) Configure:

```text
Build Command: pip install -r requirements.txt
Start Command: python app.py
```

3) O Render define a porta via a variável `PORT` automaticamente.
4) Se precisar, ajuste `--plots-dir` e `--results-dir` no Start Command.
5) O `runtime.txt` na raiz fixa a versão do Python em `3.11.9` — sem ele, o Render
   pode escolher um Python mais novo que ainda não tem wheel pré-compilada para
   `pandas==2.2.0`, forçando compilação from source (que falha no build). Se
   trocar a versão do Python aqui, confira se ainda existe wheel para todas as
   deps do `requirements.txt` antes de fazer deploy.

## Abas disponíveis

| Aba | Descrição |
|-----|-----------|
| **Overview** | Mapa por bacia/season, com slider de threshold, mostrando R² e PREDEP (máximo, MoV vencedor e lag ótimo) por pixel |
| **Exploração** | Mapa interativo pixel a pixel (R²+PREDEP, lag ótimo ou intensidade vencedora) + tabela de estatísticas por MoV |
| **Lag 0** | Mapa do MoV vencedor por pixel fixando o lag em 0 |
| **Lag's** | Comparação por lag das famílias R² e PREDEP, com threshold — mostra máximo, MoV vencedor e lag ótimo lado a lado |
| **MoV Vencedor** | Comparação, por lag, entre o MoV com melhor R² e o MoV com melhor PREDEP (podem ser MoVs diferentes) |
| **SOM** | Regimes de previsibilidade obtidos via Self-Organizing Map, treinado de forma geral ou por season (DJF/MAM/JJA/SON) |
