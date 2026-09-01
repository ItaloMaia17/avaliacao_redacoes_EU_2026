"""
Roda os experimentos de avaliação automática de redações via OpenRouter,
cobrindo o desenho fatorial 3x2 (técnica x estrutura) para um ou mais
modelos (Seção 3.4 da metodologia).

Uso:
    export OPENROUTER_API_KEY="sk-or-..."

    # tudo: todos os modelos, todas as técnicas/estruturas, fase precisão
    python run_experiments.py

    # um recorte específico
    python run_experiments.py --models llama-3.2-3b gpt-4o-mini \
        --techniques zero-shot cot --structures holistica

    # fase de consistência (5 execuções por redação)
    python run_experiments.py --phase consistencia --repeats 5

    # validar prompts sem gastar cota
    python run_experiments.py --dry-run

Cada condição (técnica, estrutura) grava em data/results/{tecnica}_{estrutura}.jsonl,
acumulando os resultados de todos os modelos rodados (campo "model" em cada
linha). O script retoma de onde parou automaticamente: resultados já
concluídos com sucesso são pulados se você rodar de novo.

Pré-requisito: `data/processed_essays.jsonl` já gerado (ver preprocessing.py).
"""

from __future__ import annotations
import argparse
import itertools
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
import requests

from models import MODELS
from prompts import build_prompt, TECHNIQUES, STRUCTURES

load_dotenv()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
JSON_FORMAT_MODELS = {"qwen/qwen3-next-80b-a3b", "google/gemma-4-31b"}
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
NON_RETRYABLE_STATUS = {400, 401, 403, 404, 422}
REASONING_BY_TECHNIQUE: dict[str, dict[str, Any]] = {
    "zero-shot": {"effort": "none"},
    "few-shot": {"effort": "none"},
    "cot": {"effort": "high"},
}


# --------------------------------------------------------------------------
# Chamada à API com retry (rate limit e erros transitórios)
# --------------------------------------------------------------------------

class RateLimitError(Exception):
    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class TransientAPIError(Exception):
    """5xx / rede — vale tentar de novo."""


class NonRetryableAPIError(Exception):
    """4xx que não é rate limit — não adianta repetir."""


def call_with_retry(func: Callable[[], Any], max_retries: int = 5,
                     base_delay: float = 1.0, max_delay: float = 60.0) -> Any:
    for attempt in range(max_retries):
        try:
            return func()
        except NonRetryableAPIError:
            raise
        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise
            wait = e.retry_after if e.retry_after is not None else min(max_delay, base_delay * 2**attempt)
            wait += random.uniform(0, wait * 0.2)
            print(f"[rate-limit] {attempt + 1}/{max_retries}: {e}. Aguardando {wait:.1f}s...")
            time.sleep(wait)
        except (TransientAPIError, requests.exceptions.RequestException) as e:
            if attempt == max_retries - 1:
                print(f"[error] todas as tentativas falharam: {e}")
                raise
            wait = min(max_delay, base_delay * 2**attempt) + random.uniform(0, 1)
            print(f"[retry] {attempt + 1}/{max_retries} falhou: {e}. Aguardando {wait:.1f}s...")
            time.sleep(wait)


def _post_openrouter(payload: dict[str, Any], api_key: str) -> dict[str, Any]:
    resp = requests.post(
        OPENROUTER_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=120,
    )
    if resp.status_code == 429:
        retry_after = resp.headers.get("Retry-After")
        raise RateLimitError(f"429 do OpenRouter", float(retry_after) if retry_after else None)
    if resp.status_code in NON_RETRYABLE_STATUS:
        raise NonRetryableAPIError(f"{resp.status_code}: {resp.text[:300]}")
    if resp.status_code in RETRYABLE_STATUS:
        raise TransientAPIError(f"{resp.status_code}: {resp.text[:300]}")
    resp.raise_for_status()
    return resp.json()


def call_openrouter(model_id: str, system: str, user: str, technique: str, temperature: float,
                     seed: int | None, max_retries: int = 5) -> dict[str, Any]:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Defina a variável de ambiente OPENROUTER_API_KEY antes de rodar.")

    payload: dict[str, Any] = {
        "model": model_id,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": temperature,
    }

    reasoning = REASONING_BY_TECHNIQUE.get(technique)
    if reasoning:
        payload["reasoning"] = reasoning
            

    if model_id in JSON_FORMAT_MODELS:
        payload["response_format"] = {"type": "json_object"}
    if seed is not None:
        payload["seed"] = seed  # nem todo provedor respeita

    return call_with_retry(lambda: _post_openrouter(payload, api_key), max_retries=max_retries)


def parse_model_output(raw: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    """(parsed, failure_reason). Falhas são excluídas das métricas (Seção 3.4)."""
    try:
        choice = raw["choices"][0]
        finish_reason = choice.get("finish_reason") or choice.get("native_finish_reason", "")
        if finish_reason == "tool_calls" or choice["message"].get("tool_calls"):
            return None, "tool_calls"

        content = choice["message"]["content"]
        if content is None:
            return None, "content_null"

        content = content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            content = content[4:] if content.startswith("json") else content
            content = content.strip()

        return json.loads(content), ""
    except (KeyError, IndexError, TypeError):
        return None, "response_structure_error"
    except json.JSONDecodeError as e:
        return None, f"json_decode_error: {e.msg}"


# --------------------------------------------------------------------------
# Execução de uma condição (modelo x técnica x estrutura)
# --------------------------------------------------------------------------

def load_processed_essays(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_completed_keys(out_path: str) -> set[tuple[str, str, int]]:
    """(essay_key, model, run_index) já concluídos com sucesso em `out_path`."""
    completed: set[tuple[str, str, int]] = set()
    p = Path(out_path)
    if not p.exists():
        return completed
    with open(p, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not row.get("parse_failed", True):
                completed.add((row["essay_key"], row["model"], row["run_index"]))
    return completed


def run_condition(
    model_key: str, technique: str, structure: str, essays: list[dict[str, Any]],
    out_path: str, temperature: float, seed: int | None, phase: str, repeats: int,
    max_retries: int, request_delay: float, dry_run: bool, few_shot_examples=None,
    resume: bool = True, truncate: bool = False,
) -> None:
    model_id = MODELS[model_key]
    tag = f"{model_key} | {technique}/{structure}"

    if dry_run:
        n_run = 0
        for essay, run_index in itertools.product(essays, range(repeats)):
            prompt = build_prompt(essay, technique, structure, few_shot_examples)
            print(f"[dry-run] {tag} | {essay['essay_key']} run={run_index} "
                  f"(system={len(prompt['system'])} chars, user={len(prompt['user'])} chars)")
            n_run += 1
        print(f"[dry-run ok] {tag}: {n_run} prompts validados.")
        return

    completed: set[tuple[str, str, int]] = load_completed_keys(out_path) if resume else set()
    if completed:
        print(f"[resume] {len(completed)} execuções já concluídas em {out_path} — pulando.")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if truncate else "a"
    if mode == "w":
        print(f"[overwrite] recriando {out_path} do zero.")

    n_run = n_failed = n_skipped = 0
    with open(out_path, mode, encoding="utf-8") as out_file:
        for essay, run_index in itertools.product(essays, range(repeats)):
            key = (essay["essay_key"], model_key, run_index)
            if key in completed:
                n_skipped += 1
                continue

            prompt = build_prompt(essay, technique, structure, few_shot_examples)
            try:
                raw = call_openrouter(model_id, prompt["system"], prompt["user"],
                                       technique, temperature, seed, max_retries=max_retries)
                parsed, failure_reason = parse_model_output(raw)
            except NonRetryableAPIError as e:
                raw, parsed, failure_reason = None, None, f"api_error: {e}"
            except Exception as e:
                raw, parsed, failure_reason = None, None, f"retries_exhausted: {e}"

            result = {
                "essay_key": essay["essay_key"], "model": model_key, "technique": technique,
                "structure": structure, "phase": phase, "run_index": run_index,
                "temperature": temperature, "seed": seed,
                "n_human_graders": len(essay.get("annotations", [])),
                "parsed_output": parsed, "parse_failed": parsed is None,
                "failure_reason": failure_reason, "raw_response": raw,
            }
            out_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_file.flush()
            os.fsync(out_file.fileno())

            n_run += 1
            n_failed += result["parse_failed"]
            print(f"[{tag} | {essay['essay_key']} run={run_index}] "
                  f"{'ok' if parsed else f'FALHA ({failure_reason})'}")
            time.sleep(request_delay)

    print(f"[ok] {tag}: {n_run} rodadas ({n_skipped} puladas por resume), "
          f"{n_failed} falhas de parsing -> {out_path}")


# --------------------------------------------------------------------------
# Driver: roda o desenho fatorial completo
# --------------------------------------------------------------------------

def run_experiments(
    models: list[str], techniques: list[str], structures: list[str],
    data_path: str, out_dir: str, temperature: float, seed: int | None,
    n: int | None, phase: str, repeats: int, max_retries: int,
    request_delay: float, dry_run: bool, few_shot_examples=None,
    resume: bool = True,
) -> None:
    essays = load_processed_essays(data_path)[:n]
    print(f"{len(essays)} redações | {len(models)} modelo(s) x "
          f"{len(techniques)} técnica(s) x {len(structures)} estrutura(s)\n")

    touched_paths: set[str] = set()  # controla truncamento só na 1a vez por arquivo
    for technique, structure in itertools.product(techniques, structures):
        out_path = str(Path(out_dir) / f"{technique}_{structure}.jsonl")
        for model_key in models:
            if technique == "few-shot" and not few_shot_examples:
                print(f"[skip] few-shot exige few_shot_examples (Seção 3.4) — "
                      f"pulando {model_key} | few-shot/{structure}.")
                continue
            truncate = (not resume) and out_path not in touched_paths
            touched_paths.add(out_path)
            run_condition(
                model_key, technique, structure, essays, out_path, temperature, seed,
                phase, repeats, max_retries, request_delay, dry_run, few_shot_examples,
                resume=resume, truncate=truncate,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Roda os experimentos de correção automática via OpenRouter.")
    parser.add_argument("--models", nargs="+", choices=MODELS.keys(), default=list(MODELS.keys()))
    parser.add_argument("--techniques", nargs="+", choices=TECHNIQUES, default=list(TECHNIQUES))
    parser.add_argument("--structures", nargs="+", choices=STRUCTURES, default=list(STRUCTURES))
    parser.add_argument("--n", type=int, default=None, help="Limita o nº de redações (padrão: todas).")
    parser.add_argument("--out-dir", default="data/results")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--data", default="data/processed_essays.jsonl")
    parser.add_argument("--phase", choices=["precisao", "consistencia"], default="precisao")
    parser.add_argument("--repeats", type=int, default=1,
                         help="Execuções por redação. Use 5 com --phase consistencia.")
    parser.add_argument("--dry-run", action="store_true", help="Valida prompts sem chamar a API.")
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--request-delay", type=float, default=1.0,
                         help="Segundos entre chamadas bem-sucedidas.")
    parser.add_argument("--no-resume", action="store_true",
                         help="Ignora resultados já salvos e recomeça cada arquivo do zero.")
    args = parser.parse_args()

    run_experiments(
        models=args.models, techniques=args.techniques, structures=args.structures,
        data_path=args.data, out_dir=args.out_dir, temperature=args.temperature,
        seed=args.seed, n=args.n, phase=args.phase, repeats=args.repeats,
        max_retries=args.max_retries, request_delay=args.request_delay,
        dry_run=args.dry_run, few_shot_examples=None, resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()