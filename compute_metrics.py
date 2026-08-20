"""
Aplica as métricas da Seção 3.6 sobre os resultados gerados por
run_experiment.py.

Uso:
    python compute_metrics.py --results data/pilot_results.jsonl --structure estruturada
    python compute_metrics.py --results data/consistency_results.jsonl --structure holistica
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from projeto_tcc.metrics import COMPETENCY_KEYS, consistency_metrics, human_human_agreement, precision_metrics


def load_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def print_precision(summary: dict[str, dict[str, Any]], titulo: str = "Precisão (QWK, MAE)") -> None:
    print(f"\n=== {titulo} ===")
    if not summary:
        print("  Nenhum par previsão/referência válido encontrado.")
        return
    ordered_keys = [*COMPETENCY_KEYS, "total"]
    for key in ordered_keys:
        if key not in summary:
            continue
        v = summary[key]
        print(f"  {key:>15s} | n_pares={v['n_pares']:4d} | QWK={v['qwk']:.3f} | MAE={v['mae']:.2f}")


def print_consistency(result: dict[str, Any]) -> None:
    print("\n=== Consistência (desvio padrão, coeficiente de variação) ===")
    summary = result["summary"]
    if not summary:
        print("  Nenhuma redação com >=2 execuções válidas encontrada.")
        return
    ordered_keys = [*COMPETENCY_KEYS, "total"]
    for key in ordered_keys:
        if key not in summary:
            continue
        v = summary[key]
        cv_str = f"{v['cv_medio']:.3f}" if v["cv_medio"] is not None else "N/A"
        print(
            f"  {key:>15s} | n_redacoes={v['n_redacoes']:3d} | "
            f"std_médio={v['std_medio']:.2f} | CV_médio={cv_str}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Calcula QWK, MAE, desvio padrão e CV.")
    parser.add_argument("--essays", default="data/processed_essays.jsonl")
    parser.add_argument("--results", required=True, help="Arquivo .jsonl gerado por run_experiment.py")
    parser.add_argument("--structure", choices=["holistica", "estruturada"], required=True)
    args = parser.parse_args()

    essays_raw = load_jsonl(args.essays)
    essays_by_key = {e["essay_key"]: e for e in essays_raw}

    results = load_jsonl(args.results)
    n_parse_failed = sum(1 for r in results if r.get("parse_failed"))
    print(
        f"[info] {len(results)} registros carregados de {args.results} "
        f"| falhas de parsing: {n_parse_failed} (excluídas das métricas)"
    )
    if n_parse_failed:
        from collections import Counter
        reasons = Counter(
            r.get("failure_reason", "desconhecido")
            for r in results if r.get("parse_failed")
        )
        for reason, count in reasons.most_common():
            print(f"  [{count}x] {reason}")

    precision_results = [r for r in results if r.get("phase", "precisao") == "precisao"]
    consistency_results = [r for r in results if r.get("phase") == "consistencia"]

    if precision_results:
        precision_summary = precision_metrics(precision_results, essays_by_key, args.structure)
        print_precision(precision_summary, titulo="Precisão — modelo vs. humano")
    else:
        precision_summary = {}
        print("\n[info] nenhum registro com phase='precisao' neste arquivo.")

    hh_summary = human_human_agreement(essays_by_key)
    n_multi = hh_summary.pop("_meta", {}).get("n_redacoes_com_multiplos_avaliadores", 0)
    if n_multi:
        print_precision(hh_summary, titulo="Teto de referência — humano vs. humano")
        print(f"  ({n_multi} redação(ões) com 2+ avaliadores usada(s) neste cálculo)")

        if precision_results:
            print("\n=== QWK: modelo vs. teto humano-humano ===")
            for key in (*COMPETENCY_KEYS, "total"):
                if key in precision_summary and key in hh_summary:
                    modelo_qwk = precision_summary[key]["qwk"]
                    humano_qwk = hh_summary[key]["qwk"]
                    print(
                        f"  {key:>15s} | modelo={modelo_qwk:.3f} | "
                        f"humano-humano={humano_qwk:.3f} | "
                        f"gap={humano_qwk - modelo_qwk:+.3f}"
                    )
    else:
        print(
            "\n[info] nenhuma redação com 2+ avaliadores em "
            f"{args.essays} — teto humano-humano não pôde ser calculado."
        )

    if consistency_results:
        print_consistency(consistency_metrics(consistency_results, args.structure))
    else:
        print(
            "\n[info] nenhum registro com phase='consistencia' neste arquivo — "
            "rode run_experiment.py com --phase consistencia --repeats 5 "
            "--temperature 0.3 para gerar."
        )


if __name__ == "__main__":
    main()
