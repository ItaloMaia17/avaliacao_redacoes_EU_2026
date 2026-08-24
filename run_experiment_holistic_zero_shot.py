"""
Teste piloto: roda N redações pré-processadas através de UM modelo, sob UMA
das 6 condições experimentais, via OpenRouter.

Uso:
    export OPENROUTER_API_KEY="sk-or-..."
    python run_experiment.py --model llama-3.2-3b --technique zero-shot \
        --structure holistica --n 3

Pré-requisito: já ter rodado `python preprocessing.py` (ou usado o arquivo
de amostra `data/sample_raw.jsonl` via `group_rows`) para gerar
`data/processed_essays.jsonl`.

Ajustes desta versão (lidar com limites de taxa da API):
  - Retry diferencia erros retentáveis (429/5xx/timeout/conexão) de erros
    não-retentáveis (400/401/403/404) — estes últimos falham na hora, sem
    queimar as tentativas.
  - Em 429, respeita o header Retry-After quando presente; senão usa backoff
    exponencial com jitter.
  - Escreve cada resultado no arquivo de saída assim que é produzido
    (append + flush), em vez de guardar tudo em memória até o fim.
  - Suporta retomada: se `--out` já existir, pula combinações
    (essay_key, run_index) já registradas com sucesso.
  - Delay entre chamadas configurável via --request-delay (gentileza com o
    provedor), e --max-retries configurável.
"""

from __future__ import annotations
import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Callable
from dotenv import load_dotenv

import requests

from projeto_tcc.models import MODELS
from projeto_tcc.prompts import build_prompt, TECHNIQUES, STRUCTURES

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
PILOT_MODEL = "llama-3.2-3b"  # modelo para teste piloto
load_dotenv()  # carrega as variáveis de ambiente do arquivo .env

MODELS_WITH_JSON_FORMAT_SUPPORT: set[str] = {
    "qwen/qwen3-next-80b-a3b",
    "google/gemma-4-31b",
}

# Status HTTP que valem retry (problema transitório do lado do provedor).
RETRYABLE_STATUS: set[int] = {429, 500, 502, 503, 504}
# Status HTTP que NUNCA devem ser retentados (erro do request em si).
NON_RETRYABLE_STATUS: set[int] = {400, 401, 403, 404, 422}


class RateLimitError(Exception):
    """429 — respeita Retry-After quando o provedor manda."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class TransientAPIError(Exception):
    """5xx — provavelmente vale a pena tentar de novo."""


class NonRetryableAPIError(Exception):
    """4xx que não é rate limit — erro de request, não adianta repetir."""


def call_with_retry(
    func: Callable[[], Any],
    max_retries: int = 7,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
) -> Any:
    """Executa `func`, retentando com backoff exponencial + jitter.

    - RateLimitError: usa retry_after do header se disponível.
    - TransientAPIError / erros de rede (timeout, conexão): backoff exponencial.
    - NonRetryableAPIError: propaga imediatamente, sem retry.
    """
    for attempt in range(max_retries):
        try:
            return func()
        except NonRetryableAPIError:
            raise
        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise
            if e.retry_after is not None:
                wait_time = e.retry_after
            else:
                wait_time = min(max_delay, base_delay * (2**attempt))
            wait_time += random.uniform(0, wait_time * 0.2)  # jitter
            print(
                f"[rate-limit] tentativa {attempt + 1}/{max_retries}: {e}. "
                f"Aguardando {wait_time:.1f}s..."
            )
            time.sleep(wait_time)
        except (
            TransientAPIError,
            requests.exceptions.RequestException,
            requests.exceptions.Timeout,
        ) as e:
            if attempt == max_retries - 1:
                print(f"[error] todas as tentativas falharam: {e}")
                raise
            wait_time = min(max_delay, base_delay * (2**attempt))
            wait_time += random.uniform(0, 1)
            print(
                f"[retry] tentativa {attempt + 1}/{max_retries} falhou: {e}. "
                f"Aguardando {wait_time:.1f}s..."
            )
            time.sleep(wait_time)


def _post_openrouter(payload: dict[str, Any], api_key: str) -> dict[str, Any]:
    """Faz a chamada HTTP e classifica o erro (se houver) em retentável ou não."""
    response = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=120,
    )

    if response.status_code == 429:
        retry_after_header = response.headers.get("Retry-After")
        retry_after = float(retry_after_header) if retry_after_header else None
        raise RateLimitError(
            f"429 recebido do OpenRouter (retry-after={retry_after})",
            retry_after=retry_after,
        )

    if response.status_code in NON_RETRYABLE_STATUS:
        raise NonRetryableAPIError(
            f"{response.status_code} do OpenRouter (não-retentável): "
            f"{response.text[:300]}"
        )

    if response.status_code in RETRYABLE_STATUS:
        raise TransientAPIError(
            f"{response.status_code} do OpenRouter (transitório): "
            f"{response.text[:300]}"
        )

    response.raise_for_status()  # captura qualquer outro status inesperado
    return response.json()


def call_openrouter(
    model_id: str,
    system: str,
    user: str,
    temperature: float,
    seed: int | None,
    max_retries: int = 7,
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
        payload["seed"] = seed  # nem todo provedor respeita

    return call_with_retry(
        lambda: _post_openrouter(payload, api_key),
        max_retries=max_retries,
    )


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


def load_completed_keys(out_path: str) -> set[tuple[str, int]]:
    """Lê resultados já salvos em `out_path` (se existir) para permitir retomada.

    Só considera "completo" o par (essay_key, run_index) cujo resultado NÃO
    falhou no parsing — assim, falhas antigas são automaticamente
    re-executadas se você rodar o script de novo.
    """
    completed: set[tuple[str, int]] = set()
    p = Path(out_path)
    if not p.exists():
        return completed
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # linha corrompida (ex: escrita interrompida) — ignora
            if not row.get("parse_failed", True):
                completed.add((row["essay_key"], row["run_index"]))
    return completed


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
    max_retries: int = 7,
    request_delay: float = 1.0,
    resume: bool = True,
) -> None:
    essays = load_processed_essays(data_path)[:n]
    model_id = MODELS[model_key]

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    completed_keys: set[tuple[str, int]] = set()
    if resume and not dry_run:
        completed_keys = load_completed_keys(out_path)
        if completed_keys:
            print(f"[resume] {len(completed_keys)} execuções já concluídas serão puladas.")

    # Append: preserva resultados de execuções anteriores (usado para resume).
    out_mode = "a" if (resume and not dry_run) else "w"
    n_written = 0
    n_failed = 0
    n_skipped = 0

    out_file = None if dry_run else open(out_path, out_mode, encoding="utf-8")
    try:
        for essay in essays:
            for run_index in range(repeats):
                if (essay["essay_key"], run_index) in completed_keys:
                    n_skipped += 1
                    continue

                prompt = build_prompt(essay, technique, structure, few_shot_examples)

                if dry_run:
                    print(f"\n=== prompt para {essay['essay_key']} (run {run_index}) ===")
                    print("--- system ---")
                    print(prompt["system"][:500] + ("..." if len(prompt["system"]) > 500 else ""))
                    print("--- user (trecho) ---")
                    print(prompt["user"][:300] + ("..." if len(prompt["user"]) > 300 else ""))
                    n_written += 1
                    continue

                try:
                    raw = call_openrouter(
                        model_id,
                        prompt["system"],
                        prompt["user"],
                        temperature,
                        seed,
                        max_retries=max_retries,
                    )
                    parsed, failure_reason = parse_model_output(raw)
                except NonRetryableAPIError as e:
                    # Erro do request em si (ex: 400/401) — registra e segue,
                    # em vez de derrubar o experimento inteiro.
                    raw = None
                    parsed = None
                    failure_reason = f"api_error: {e}"
                except Exception as e:
                    # Retries esgotados em erro transitório (429/5xx/rede).
                    raw = None
                    parsed = None
                    failure_reason = f"retries_exhausted: {e}"

                result = {
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

                # Escreve e força para o disco imediatamente: se o processo
                # cair no meio (ex: rate limit persistente), nada se perde.
                out_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_file.flush()
                os.fsync(out_file.fileno())

                n_written += 1
                n_failed += result["parse_failed"]

                status = "ok" if parsed else f"FALHA ({failure_reason})"
                print(f"[{essay['essay_key']} run={run_index}] {status}")
                time.sleep(request_delay)  # gentileza com limites de taxa do provedor
    finally:
        if out_file is not None:
            out_file.close()

    if dry_run:
        print(f"\n[dry-run ok] {n_written} prompts validados, nenhuma chamada de API realizada.")
        return

    print(
        f"\n[ok] {n_written} redações processadas nesta execução "
        f"({n_skipped} já estavam concluídas e foram puladas) | "
        f"falhas de parsing: {n_failed}"
    )
    print(f"Resultados salvos em {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Roda teste piloto de correção automática de redações via OpenRouter."
    )
    parser.add_argument("--model", choices=MODELS.keys(), default=PILOT_MODEL)
    parser.add_argument("--technique", choices=TECHNIQUES, default="zero-shot")
    parser.add_argument("--structure", choices=STRUCTURES, default="holistica")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--out", default="data/pilot_results.jsonl")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--data", default="data/processed_essays.jsonl")
    parser.add_argument("--phase", choices=["precisao", "consistencia"], default="precisao")
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Execuções por redação. Use 5 com --phase consistencia.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Valida prompts sem chamar a API."
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=7,
        help="Tentativas por chamada em caso de erro retentável (429/5xx/rede).",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=1.0,
        help="Segundos de espera entre chamadas bem-sucedidas (gentileza com o provedor).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignora resultados já salvos em --out e recomeça do zero (sobrescreve o arquivo).",
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
        phase=args.phase,
        repeats=args.repeats,
        dry_run=args.dry_run,
        max_retries=args.max_retries,
        request_delay=args.request_delay,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()