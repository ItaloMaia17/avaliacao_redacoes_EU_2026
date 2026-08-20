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

import requests

from projeto_tcc.models import MODELS
from projeto_tcc.prompts import build_prompt

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Modelos que suportam response_format=json_object nativamente.
# Os demais recebem apenas instrução de formato no prompt (já presente em
# prompts.py/_output_schema_instructions). Isso evita o comportamento de
# tool_calls observado no Llama 3.2 3B via Cloudflare, onde o modelo
# "escapa" para o mecanismo de tool use em vez de gerar conteúdo inline
# quando recebe um response_format que não reconhece.
MODELS_WITH_JSON_FORMAT_SUPPORT: set[str] = {
    "qwen/qwen3-next-80b-a3b",
    "google/gemma-4-31b",
}


def call_openrouter(
    model_id: str,
    system: str,
    user: str,
    temperature: float,
    seed: int | None,
) -> dict[str, Any]:
    api_key = os.environ.get("OPENROUTER_API_KEY")
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

    response = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=120,
    )
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
    parser = argparse.ArgumentParser(description="Teste piloto de uma condição experimental.")
    parser.add_argument("--model", choices=MODELS.keys(), default="llama-3.2-3b")
    parser.add_argument("--technique", choices=["zero-shot", "few-shot", "cot"], default="zero-shot")
    parser.add_argument("--structure", choices=["holistica", "estruturada"], default="holistica")
    parser.add_argument("--n", type=int, default=3, help="Número de redações no teste piloto.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", default="data/processed_essays.jsonl")
    parser.add_argument("--out", default="data/pilot_results.jsonl")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Monta os prompts mas não chama a API (valida o pipeline sem custo).",
    )
    parser.add_argument(
        "--phase",
        choices=["precisao", "consistencia"],
        default="precisao",
        help="precisao: 1 execução/redação, temp=0. consistencia: N execuções, temp>0 (Seção 3.5).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Número de execuções independentes por redação (use 5 para a fase de consistência).",
    )
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
        dry_run=args.dry_run,
        phase=args.phase,
        repeats=args.repeats,
    )


if __name__ == "__main__":
    main()
