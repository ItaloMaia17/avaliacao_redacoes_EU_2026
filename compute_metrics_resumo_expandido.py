"""
Avaliação do resumo expandido.

Desenho do experimento:
- Precisão: QWK da nota geral modelo × humana
- Consistência: desvio padrão e coeficiente de variação (CV) da nota geral
  entre 3 execuções independentes da mesma redação
- Amostra: 100 redações selecionadas aleatoriamente
- Técnica: zero-shot
- Estrutura: estruturada por competência do ENEM

A nota geral do modelo é calculada pela soma das cinco competências.

Uso:
    python compute_metrics_resumo_expandido.py \
        --results data/results/zero-shot_estruturada.jsonl \
        --structure estruturada
"""

from __future__ import annotations

import argparse
import json
import random
from typing import Any

from sklearn.metrics import cohen_kappa_score

from metrics2 import consistency_metrics


def load_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def qwk_from_pairs(pairs: list[tuple[float, float]]) -> float | None:
    if len(pairs) < 2:
        return None
    y1 = [a for a, _ in pairs]
    y2 = [b for _, b in pairs]
    return float(cohen_kappa_score(y1, y2, weights="quadratic"))


def model_total_from_parsed(parsed_output: dict[str, Any] | None, structure: str) -> float | None:
    if parsed_output is None:
        return None

    if structure == "holistica":
        total = parsed_output.get("nota_final")
        return float(total) if isinstance(total, (int, float)) else None

    if structure != "estruturada":
        raise ValueError(f"Estrutura inválida: {structure}")

    soma = 0.0
    for i in range(1, 6):
        key = f"competencia_{i}"
        value = parsed_output.get(key)
        if isinstance(value, dict):
            value = value.get("nota")
        if not isinstance(value, (int, float)):
            return None
        soma += float(value)
    return soma


def compute_human_human_reference(essays_by_key: dict[str, dict[str, Any]]) -> dict[str, Any]:
    pairs: list[tuple[float, float]] = []
    for essay in essays_by_key.values():
        annotations = essay.get("annotations") or []
        refs = {a.get("reference"): a for a in annotations if isinstance(a, dict) and a.get("reference")}
        a = refs.get("grader_a")
        b = refs.get("grader_b")
        if a is None or b is None:
            continue
        total_a = a.get("total")
        total_b = b.get("total")
        if total_a is None or total_b is None:
            continue
        pairs.append((float(total_a), float(total_b)))

    return {
        "n_pares": len(pairs),
        "qwk": qwk_from_pairs(pairs),
    }


def compute_precision_from_first_consistency_run(
    results: list[dict[str, Any]],
    essays_by_key: dict[str, dict[str, Any]],
    structure: str,
    reference: str | None = None,
) -> dict[str, Any]:
    pairs: list[tuple[float, float]] = []
    n_redacoes = set()

    for result in results:
        if result.get("parse_failed") or result.get("dry_run"):
            continue
        if result.get("phase") != "consistencia":
            continue
        if result.get("run_index") != 0:
            continue

        essay_key = result.get("essay_key")
        if essay_key is None:
            continue

        essay = essays_by_key.get(essay_key)
        if essay is None:
            continue

        model_total = model_total_from_parsed(result.get("parsed_output"), structure)
        if model_total is None:
            continue

        annotations = essay.get("annotations") or []
        refs = {a.get("reference"): a for a in annotations if isinstance(a, dict) and a.get("reference")}

        if reference is not None:
            annotation = refs.get(reference)
            if annotation is None:
                continue
            human_total = annotation.get("total")
            if human_total is None:
                continue
            pairs.append((float(model_total), float(human_total)))
            n_redacoes.add(essay_key)
            continue

        for annotation in annotations:
            human_total = annotation.get("total")
            if human_total is None:
                continue
            pairs.append((float(model_total), float(human_total)))
            n_redacoes.add(essay_key)

    return {
        "n_redacoes": len(n_redacoes),
        "n_pares": len(pairs),
        "qwk": qwk_from_pairs(pairs),
    }


def print_precision(summary: dict[str, Any], titulo: str) -> None:
    print(f"\n=== {titulo} ===")
    if not summary or summary.get("n_pares", 0) == 0:
        print("  Nenhum par válido encontrado.")
        return

    print(
        f"  n_redacoes={summary.get('n_redacoes', 0):3d} | "
        f"n_pares={summary.get('n_pares', 0):4d} | "
        f"QWK={summary.get('qwk', float('nan')):.3f}"
    )


def print_consistency(result: dict[str, Any]) -> None:
    print("\n=== Consistência — nota geral ===")
    summary = result.get("summary", {})
    total_summary = summary.get("total") or summary
    if not total_summary or not isinstance(total_summary, dict):
        print("  Nenhuma redação com >=2 execuções válidas encontrada.")
        return

    std_medio = total_summary.get("std_medio")
    cv_medio = total_summary.get("cv_medio")
    print(
        f"  n_redacoes={total_summary.get('n_redacoes', 0):3d} | "
        f"n_runs_esperado={3:d} | "
        f"std_medio={std_medio if std_medio is not None else 'N/A'} | "
        f"cv_medio={cv_medio if cv_medio is not None else 'N/A'}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Avalia precisão (QWK total) e consistência (std/CV total) para o resumo expandido."
    )
    parser.add_argument(
        "--essays",
        default="data/processed_essays.jsonl",
        help="Arquivo de redações processadas (humano).",
    )
    parser.add_argument(
        "--results",
        required=True,
        help="Arquivo JSONL dos resultados do modelo. Ex.: data/results/zero-shot_estruturada.jsonl",
    )
    parser.add_argument(
        "--structure",
        choices=["estruturada", "holistica"],
        default="estruturada",
        help="Estrutura da resposta do modelo. Para o resumo expandido, normalmente é 'estruturada'.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Filtra o arquivo de resultados para um modelo específico. Ex.: gpt-oss-120b",
    )
    args = parser.parse_args()

    essays_raw = load_jsonl(args.essays)
    essays_by_key = {essay["essay_key"]: essay for essay in essays_raw if "essay_key" in essay}

    rng = random.Random(42)
    selected_essays = rng.sample(essays_raw, 100)
    selected_keys = {essay["essay_key"] for essay in selected_essays if "essay_key" in essay}
    sampled_essays_by_key = {k: essays_by_key[k] for k in selected_keys if k in essays_by_key}

    print(f"[info] amostra selecionada com seed 42: {len(selected_keys)} redações")

    results = load_jsonl(args.results)
    if args.model:
        results = [r for r in results if r.get("model") == args.model]
        print(f"[info] avaliando apenas o modelo: {args.model}")
    results_sampled = [r for r in results if r.get("essay_key") in selected_keys]
    n_parse_failed = sum(1 for r in results_sampled if r.get("parse_failed"))
    print(
        f"[info] {len(results_sampled)} registros da amostra carregados de {args.results} | "
        f"falhas de parsing: {n_parse_failed} (excluídas)"
    )

    consistency_results = [r for r in results_sampled if r.get("phase") == "consistencia"]
    if consistency_results:
        filtered_consistency = [
            r for r in consistency_results if not r.get("parse_failed") and r.get("essay_key") in selected_keys
        ]
        if filtered_consistency:
            consistency_summary = consistency_metrics(filtered_consistency, args.structure)
            print_consistency(consistency_summary)
        else:
            print("\n[info] nenhum registro de consistência válido na amostra seed 42.")
    else:
        print("\n[info] nenhum registro com phase='consistencia' encontrado no arquivo de resultados.")

    precision_model_any = compute_precision_from_first_consistency_run(
        results_sampled,
        sampled_essays_by_key,
        args.structure,
    )
    print_precision(
        precision_model_any,
        titulo="Precisão agregada — modelo x humanos (run_index=0, amostra seed 42)",
    )

    precision_model_a = compute_precision_from_first_consistency_run(
        results_sampled,
        sampled_essays_by_key,
        args.structure,
        reference="grader_a",
    )
    print_precision(
        precision_model_a,
        titulo="Precisão — modelo x grader_a (run_index=0, amostra seed 42)",
    )

    precision_model_b = compute_precision_from_first_consistency_run(
        results_sampled,
        sampled_essays_by_key,
        args.structure,
        reference="grader_b",
    )
    print_precision(
        precision_model_b,
        titulo="Precisão — modelo x grader_b (run_index=0, amostra seed 42)",
    )

    human_reference = compute_human_human_reference(sampled_essays_by_key)
    print_precision(
        {**human_reference, "n_redacoes": len(sampled_essays_by_key)},
        titulo="Teto humano-humano — grader_a x grader_b (mesma amostra)",
    )

    for label, qwk_value in [
        ("modelo x humanos", precision_model_any.get("qwk")),
        ("modelo x grader_a", precision_model_a.get("qwk")),
        ("modelo x grader_b", precision_model_b.get("qwk")),
    ]:
        if qwk_value is not None and human_reference.get("qwk") is not None:
            gap = human_reference["qwk"] - qwk_value
            print(f"\n  gap [{label}] = {gap:+.3f} (humano-humano - {label})")


if __name__ == "__main__":
    main()
