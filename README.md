# Avaliação automática de redações do ENEM

Este projeto avalia se modelos de linguagem conseguem atribuir notas a redações do ENEM de forma próxima às avaliações humanas. O estudo usa o subconjunto `sourceAWithGraders` do dataset `kamel-usp/aes_enem_dataset` e compara dois aspectos principais:

- precisão: proximidade entre as notas do modelo e as notas humanas;
- consistência: estabilidade das notas em execuções repetidas da mesma redação.

## Desenho experimental

- técnicas: `zero-shot`
- estruturas: `estruturada`
- consistência: `redações avaliadas 3 vezes` para medir a variabilidade na saída dos modelos
- objetivo geral: medir desempenho em correção automática de redações do ENEM em termos de precisão e consistência

## Arquivos principais

- `preprocessing.py` — validação e agrupamento do corpus
- `prompts.py` — construção dos prompts para a condição experimental
- `models.py` — modelos usados via Groq
- `run_experiments.py` — execução do experimento
- `compute_metrics_resumo_expandido.py` — cálculo de QWK e consistência
- `explore_corpus.py` — análise exploratória do corpus

## Fluxo do projeto

### 1. Pré-processamento

```bash
pip install -r requirements.txt
python preprocessing.py
```

Gera `data/processed_essays.jsonl`.

### 2. Validar prompts

```bash
python run_experiments.py --dry-run
```

Verifica a montagem dos prompts sem chamar a API.

### 3. Rodar experimento

```bash
export GROQ_API_KEY="sk-or-..."

python run_experiments.py --models qwen3.8-27b --techniques zero-shot --structures estruturada --phase consistencia --repeats 3 --n 100

python run_experiments.py --models gpt-oss-120b --techniques zero-shot --structures estruturada --phase consistencia --repeats 3 --n 100

```

Os resultados são salvos em `data/results/`.

### 4. Calcular métricas

```bash
python compute_metrics_resumo_expandido.py --results data/results/zero-shot_estruturada.jsonl --structure estruturada --model gpt-oss-120b --diagnose-discarded

python compute_metrics_resumo_expandido.py --results data/results/zero-shot_estruturada.jsonl --structure estruturada --model qwen3.8-27b --diagnose-discarded
```

## Observações

- a chave real da redação é formada por `id_prompt + id`;
- o conjunto inclui múltiplas avaliações humanas por redação;
- o projeto foi pensado para permitir execução local, teste sem rede e análise de resultados em conjunto.
