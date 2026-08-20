"""
Teste piloto: roda N redações pré-processadas através de UM modelo, sob UMA
das 6 condições experimentais, via OpenRouter (Seção 3.5 da metodologia).

Uso:
    export OPENROUTER_API_KEY="sk-or-..."
    python run_experiment.py --model llama-3.2-3b --technique zero-shot \
        --structure holistica --n 3

Pré-requisito: já ter rodado `python preprocessing.py` (ou usado o arquivo
de amostra `data/sample_raw.jsonl` via `group_rows`) para gerar
`data/processed_essays.jsonl`.
"""

from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

import requests

from projeto_tcc.models import MODELS
from projeto_tcc.prompts import build_prompt

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
PILOT_MODEL = "llama-3.2-3b"  # modelo  para teste piloto
load_dotenv()  # carrega variáveis de ambiente do arquivo .env

MODELS_WITH_JSON_FORMAT_SUPPORT: set[str] = {
    "qwen/qwen3-next-80b-a3b",
    "google/gemma-4-31b",
}


# chama a API do OpenRouter com retry em caso de falha temporária (timeout, 429, 5xx)
def call_with_retry(func, max_retries=5, backoff_factor=2):
    for attempt in range(max_retries):
        try:
            return func()
        except (requests.exceptions.RequestException, requests.exceptions.Timeout) as e:
            if attempt < max_retries - 1:
                wait_time = backoff_factor ** attempt
                print(f"[retry] tentativa {attempt + 1}/{max_retries} falhou: {e}. Aguardando {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"[error] todas as tentativas falharam: {e}")
                raise
def call_openrouter(
    model_id: str,
    system: str,
    user: str,
    temperature: float,
    seed: int | None,
) -> dict[str, Any]:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Defina a variável de ambiente OPENROUTER_API_KEY antes de rodar."
        )

    payload: dict[str, Any] = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
    }
    if model_id in MODELS_WITH_JSON_FORMAT_SUPPORT:
        payload["response_format"] = {"type": "json_object"}

    if seed is not None:
        payload["seed"] = seed  # nem todo provedor respeita; ver Seção 3.5

    response = call_with_retry(lambda: requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=120,
    ))
    response.raise_for_status()
    return response.json()


def parse_model_output(
    raw_response: dict[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    """Extrai e faz parsing do JSON retornado pelo modelo.

    Retorna (parsed, failure_reason):
      - parsed: dict com as notas, ou None em caso de falha.
      - failure_reason: "" se ok, ou uma descrição da falha para registro.

    Falhas são registradas e excluídas das métricas (Seção 3.4).
    Causas comuns:
      - "tool_calls": modelo respondeu via tool use em vez de conteúdo inline
        (response_format=json_object não suportado pelo provedor).
      - "content_null": conteúdo vazio sem tool_calls.
      - "json_decode_error": conteúdo presente mas não é JSON válido.
    """
    try:
        choice = raw_response["choices"][0]
        finish_reason = choice.get("finish_reason") or choice.get("native_finish_reason", "")

        if finish_reason == "tool_calls" or choice["message"].get("tool_calls"):
            return None, "tool_calls"

        content = choice["message"]["content"]
        if content is None:
            return None, "content_null"

        # Remove delimitadores de bloco de código que alguns modelos incluem
        # mesmo quando instruídos a retornar JSON puro.
        content = content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        return json.loads(content), ""

    except (KeyError, IndexError, TypeError):
        return None, "response_structure_error"
    except json.JSONDecodeError as e:
        return None, f"json_decode_error: {e.msg}"

def load_processed_essays(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_pilot(
    model_key: str,
    technique: str,
    structure: str,
    n: int,
    temperature: float,
    seed: int | None,
    data_path: str,
    out_path: str,
    few_shot_examples=None,
    dry_run: bool = False,
    phase: str = "precisao",
    repeats: int = 1,
) -> None:
    essays = load_processed_essays(data_path)[:n]
    model_id = MODELS[model_key]

    results: list[dict[str, Any]] = []
    for essay in essays:
        for run_index in range(repeats):
            prompt = build_prompt(essay, technique, structure, few_shot_examples)

            if dry_run:
                # Não chama a API: apenas valida que o prompt é montado
                # corretamente. Útil para testar o pipeline sem gastar cota.
                print(f"\n=== prompt para {essay['essay_key']} (run {run_index}) ===")
                print("--- system ---")
                print(prompt["system"][:500] + ("..." if len(prompt["system"]) > 500 else ""))
                print("--- user (trecho) ---")
                print(prompt["user"][:300] + ("..." if len(prompt["user"]) > 300 else ""))
                results.append(
                    {
                        "essay_key": essay["essay_key"],
                        "model": model_key,
                        "technique": technique,
                        "structure": structure,
                        "phase": phase,
                        "run_index": run_index,
                        "dry_run": True,
                        "system_chars": len(prompt["system"]),
                        "user_chars": len(prompt["user"]),
                    }
                )
                continue

            raw = call_openrouter(model_id, prompt["system"], prompt["user"], temperature, seed)
            parsed, failure_reason = parse_model_output(raw)

            results.append(
                {
                    "essay_key": essay["essay_key"],
                    "model": model_key,
                    "technique": technique,
                    "structure": structure,
                    "phase": phase,
                    "run_index": run_index,
                    "temperature": temperature,
                    "seed": seed,
                    "n_human_graders": len(essay.get("annotations", [])),
                    "parsed_output": parsed,
                    "parse_failed": parsed is None,
                    "failure_reason": failure_reason,
                    "raw_response": raw,
                }
            )
            status = "ok" if parsed else f"FALHA ({failure_reason})"
            print(f"[{essay['essay_key']} run={run_index}] {status}")
            time.sleep(1)  # gentileza com limites de taxa do provedor

    if dry_run:
        print(f"\n[dry-run ok] {len(results)} prompts validados, nenhuma chamada de API realizada.")
        return

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_failed = sum(r["parse_failed"] for r in results)
    print(f"\n[ok] {len(results)} redações processadas | falhas de parsing: {n_failed}")
    print(f"Resultados salvos em {out_path}")

def main() -> None:
    parser = argparse.ArgumentParser(...)
    parser.add_argument("--model",     choices=MODELS.keys(), default=PILOT_MODEL)
    parser.add_argument("--technique", choices=[...],         default="zero-shot")
    parser.add_argument("--structure", choices=[...],         default="holistica")
    parser.add_argument("--n",         type=int,              default=3)
    parser.add_argument("--out",                              default="data/pilot_results.jsonl")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--data", default="data/processed_essays.jsonl")
    # Bug 1 — adicionar os dois que faltam:
    parser.add_argument("--phase", choices=["precisao", "consistencia"], default="precisao")
    parser.add_argument("--repeats", type=int, default=1,
                        help="Execuções por redação. Use 5 com --phase consistencia.")
    # Bug 2 — descomentar:
    parser.add_argument("--dry-run", action="store_true",
                        help="Valida prompts sem chamar a API.")
    args = parser.parse_args()

    few_shot_examples = None
    if args.technique == "few-shot":
        raise NotImplementedError(
            "Defina aqui os 2 exemplos fixos (redação + notas) antes de usar "
            "a condição few-shot — ver comentário no topo de prompts.py e a "
            "Seção 3.4 da metodologia (os exemplos devem ser idênticos para "
            "todos os modelos e todas as redações)."
        )

    run_pilot(
        model_key=args.model,
        technique=args.technique,
        structure=args.structure,
        n=args.n,
        temperature=args.temperature,
        seed=args.seed,
        data_path=args.data,
        out_path=args.out,
        few_shot_examples=few_shot_examples,
        phase=args.phase,
        repeats=args.repeats,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
