# Pipeline de pré-processamento e teste piloto — AES ENEM

Implementa a Seção 3 (Metodologia) sobre o subconjunto `sourceAWithGraders`
do dataset `kamel-usp/aes_enem_dataset` (Silveira, Barbosa e Mauá, 2024).

## Arquivos

- `preprocessing.py` — verificação de integridade, agrupamento para
  inferência (uma redação → múltiplas notas humanas) e limpeza textual.
- `prompts.py` — construção dos prompts para as 6 condições do desenho
  fatorial 3x2 (zero-shot/few-shot/CoT × holística/estruturada).
- `models.py` — identificadores dos 5 modelos selecionados no OpenRouter.
- `run_experiment.py` — roda um teste piloto (N redações, 1 modelo,
  1 condição) via OpenRouter.
- `data/sample_raw.jsonl` — 7 linhas de amostra (5 reais e truncadas + 2
  sintéticas inválidas) para testar o pipeline **sem rede e sem gastar
  cota de API**. Inclui um caso real de uma redação com 2 avaliadores e
  um caso real do bug de `id` duplicado entre temas diferentes.

## Passo 1 — Pré-processamento

Com o dataset completo (requer rede):

```bash
pip install -r requirements.txt
python preprocessing.py
# gera data/processed_essays.jsonl
```

Para testar a lógica sem rede, usando a amostra:

```python
from preprocessing import load_rows_from_jsonl, group_rows, save_processed

rows = load_rows_from_jsonl("data/sample_raw.jsonl")
essays, discarded = group_rows(rows)
save_processed(essays, "data/processed_essays.jsonl")
```

## Passo 2 — Validar os prompts (sem custo de API)

```bash
python run_experiment.py --technique zero-shot --structure estruturada \
    --n 2 --data data/processed_essays.jsonl --dry-run
```

Isso monta os prompts reais e imprime um trecho de cada um, sem chamar a
API — útil para revisar a redação do prompt antes de gastar cota.

## Passo 3 — Teste piloto real

```bash
export OPENROUTER_API_KEY="sk-or-..."
python run_experiment.py --model llama-3.2-3b --technique zero-shot \
    --structure holistica --n 3 --temperature 0 --seed 42
# gera data/pilot_results.jsonl
```

## Pendências antes de rodar os experimentos completos

1. **Confirmar os slugs exatos no OpenRouter** para `gemma-4-31b` e
   `ministral-3b` em <https://openrouter.ai/models> — são modelos recentes
   e o identificador usado aqui pode estar desatualizado.
2. **Definir os 2 exemplos fixos de few-shot** em formato compatível com
   `prompts.build_prompt` (ver `NotImplementedError` em `run_experiment.py`).
   Os exemplos devem ser os mesmos para todos os modelos e todas as
   redações (Seção 3.4).
3. **Confirmar se o provedor escolhido no OpenRouter respeita `seed`** para
   cada um dos 5 modelos — nem todo provedor de inferência o faz; isso
   afeta a reprodutibilidade discutida na Seção 3.5.
4. Este projeto ainda não calcula QWK/MAE/desvio padrão/CV (Seção 3.6) —
   ele cobre pré-processamento + geração de respostas. O cálculo de
   métricas é um próximo passo natural, uma vez que `pilot_results.jsonl`
   esteja populado com saídas reais dos modelos.
