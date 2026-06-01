# PREDEP Visualization

Viewer interativo para resultados do projeto PREDEP.

## Estrutura esperada

```
predep_visualization/
├── app.py
├── requirements.txt
├── plots/                          # plots PNG gerados pelo pipeline
│   └── expNN/
├── results/
│   └── predep_granular_brazil/     # arquivos NetCDF (.nc)
│       └── expNN/
└── data/
    └── clusters/                   # shapefiles das bacias
        ├── amazonica/
        ├── parana/
        └── ...
```

## Instalação

```bash
pip install -r requirements.txt
```

## Execução

```bash
python app.py
python app.py --port 8051
python app.py --plots-dir /caminho/para/plots --results-dir /caminho/para/results
```

Acesse em `http://localhost:8050`

## Modos disponíveis

| Modo | Descrição |
|------|-----------|
| **Por MoV** | Navega os plots PNG por MoV, área e tipo |
| **Melhor MoV** | Comparação do MoV com maior sinal (só exps com ≥ 2 MoVs) |
| **Exploração** | Mapa interativo pixel a pixel + tabela de estatísticas por MoV |
