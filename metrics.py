"""
Métricas de avaliação (Seção 3.6 da metodologia):

  Precisão:     QWK e MAE, comparando notas geradas pelos modelos com as
                notas de referência humanas.
  Consistência: desvio padrão e coeficiente de variação (CV) entre N=5
                execuções independentes por redação.

Ambas calculadas por competência individualmente e para a nota geral, nos
casos em que isso é aplicável.

Decisão metodológica importante (ver discussão da Seção 3.1/3.6): como o
subconjunto sourceAWithGraders fornece mais de uma anotação humana por
redação, cada anotação é tratada como uma instância independente de
comparação (protocolo de Silveira et al., 2024). Ou seja, uma redação com
k avaliadores gera k pares (previsão, referência) na etapa de precisão,
em vez de 1 — embora o modelo tenha sido executado uma única vez por
redação (ver preprocessing.py e a discussão sobre "agrupamento para
inferência").
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any

from sklearn.metrics import cohen_kappa_score, mean_absolute_error

from projeto_tcc.competencies import VALID_SCORES
COMPETENCY_KEYS = [f"competencia_{i}" for i in range(1, 6)]


def snap_to_valid_scale(
    value: float, valid: list[int] = VALID_SCORES
) -> int:
    """Arredonda uma nota de competência prevista pelo modelo para o valor
    válido mais próximo do conjunto discreto {0,40,80,120,160,200}.

    Necessário porque o QWK do scikit-learn baseia o peso quadrático na
    posição relativa dos valores únicos observados, não na distância
    numérica real entre eles — uma única nota fora da escala (ex.: 90)
    distorce a ponderação de todas as demais comparações.
    """
    return min(valid, key=lambda v: abs(v - value))


def extract_model_scores(
    parsed_output: dict[str, Any] | None, structure: str
) -> dict[str, float | None]:
    """Normaliza a saída do modelo (JSON já parseado) para o formato
    {"competencia_1": ..., ..., "competencia_5": ..., "total": ...}.

    Campos não aplicáveis à estrutura (ex.: competências na condição
    holística) ficam como None.
    """
    scores: dict[str, float | None] = {k: None for k in COMPETENCY_KEYS}
    scores["total"] = None

    if parsed_output is None:
        return scores

    if structure == "holistica":
        total = parsed_output.get("nota_final")
        scores["total"] = float(total) if isinstance(total, (int, float)) else None
        return scores

    # estruturada
    competencias: list[float | None] = []
    for key in COMPETENCY_KEYS:
        campo = parsed_output.get(key)
        nota = campo.get("nota") if isinstance(campo, dict) else None
        nota = float(nota) if isinstance(nota, (int, float)) else None
        scores[key] = nota
        competencias.append(nota)

    if all(c is not None for c in competencias):
        scores["total"] = sum(competencias)  # type: ignore[arg-type]

    return scores


def _summarize_pairs(pairs: dict[str, list[tuple[float, float]]]) -> dict[str, dict[str, Any]]:
    """QWK e MAE a partir de um dict {dimensão: [(valor_a, valor_b), ...]}.

    Convenção de nomeação dos parâmetros em `cohen_kappa_score`/`mean_absolute_error`
    (y1, y2) é simétrica para QWK e MAE, então a ordem dos elementos de cada
    tupla não afeta o resultado — mas mantemos (modelo, humano) em
    `precision_metrics` e (avaliador_base, outro_avaliador) em
    `human_human_agreement` por clareza de leitura no código de origem.
    """
    summary: dict[str, dict[str, Any]] = {}
    for key, pair_list in pairs.items():
        if len(pair_list) < 2:
            continue
        y1 = [a for a, _ in pair_list]
        y2 = [b for _, b in pair_list]
        summary[key] = {
            "n_pares": len(pair_list),
            "qwk": cohen_kappa_score(y1, y2, weights="quadratic"),
            "mae": mean_absolute_error(y1, y2),
        }
    return summary


def precision_metrics(
    results: list[dict[str, Any]],
    essays_by_key: dict[str, dict[str, Any]],
    structure: str,
) -> dict[str, dict[str, Any]]:
    """QWK e MAE, comparando cada previsão do modelo a TODAS as anotações
    humanas disponíveis para a redação correspondente.

    `results`: lista de registros gerados por run_experiment.py (fase
    'precisao', 1 execução por redação).
    `essays_by_key`: dict essay_key -> Essay (como salvo por preprocessing.py),
    cada um com `annotations`: lista de {"competencies": [...], "total": ...}.
    """
    pairs: dict[str, list[tuple[float, float]]] = defaultdict(list)

    for r in results:
        if r.get("parse_failed") or r.get("dry_run"):
            continue
        essay = essays_by_key.get(r["essay_key"])
        if essay is None or not essay.get("annotations"):
            continue

        model_scores = extract_model_scores(r["parsed_output"], structure)

        for annotation in essay["annotations"]:
            if model_scores["total"] is not None and annotation.get("total") is not None:
                pairs["total"].append((model_scores["total"], annotation["total"]))

            if structure == "estruturada":
                for i, key in enumerate(COMPETENCY_KEYS):
                    pred = model_scores[key]
                    true = annotation["competencies"][i] if annotation.get("competencies") else None
                    if pred is not None and true is not None:
                        pairs[key].append((snap_to_valid_scale(pred), true))

    return _summarize_pairs(pairs)


def human_human_agreement(
    essays_by_key: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """QWK e MAE entre avaliadores humanos, para redações com 2+ avaliadores.

    Serve como TETO DE REFERÊNCIA para `precision_metrics`: a concordância
    modelo-humano não deveria, em princípio, superar a concordância
    observada entre os próprios avaliadores humanos no mesmo subconjunto
    (sourceAWithGraders), já que ambos avaliam a mesma redação sob a mesma
    matriz de competências.

    Cada redação com k avaliadores (k >= 2) contribui com k-1 pares,
    comparando o primeiro avaliador disponível contra cada um dos demais
    — evitando contar a mesma comparação duas vezes (avaliador A vs. B e
    B vs. A não são pares distintos para QWK/MAE, que são simétricos).
    """
    pairs: dict[str, list[tuple[float, float]]] = defaultdict(list)
    n_redacoes_com_multiplos = 0

    for essay in essays_by_key.values():
        annotations = essay.get("annotations") or []
        if len(annotations) < 2:
            continue
        n_redacoes_com_multiplos += 1

        base = annotations[0]
        for other in annotations[1:]:
            if base.get("total") is not None and other.get("total") is not None:
                pairs["total"].append((base["total"], other["total"]))

            base_comp = base.get("competencies")
            other_comp = other.get("competencies")
            if base_comp and other_comp:
                for i, key in enumerate(COMPETENCY_KEYS):
                    pairs[key].append((base_comp[i], other_comp[i]))

    summary = _summarize_pairs(pairs)
    summary["_meta"] = {"n_redacoes_com_multiplos_avaliadores": n_redacoes_com_multiplos}
    return summary


def consistency_metrics(
    results: list[dict[str, Any]], structure: str
) -> dict[str, Any]:
    """Desvio padrão e coeficiente de variação entre N execuções
    independentes da mesma redação (fase 'consistencia', Seção 3.6).

    Calcula primeiro por redação, depois agrega (média) entre redações
    para um resumo único por modelo/condição.
    """
    by_essay: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for r in results:
        if r.get("parse_failed") or r.get("dry_run"):
            continue
        scores = extract_model_scores(r["parsed_output"], structure)
        for key, value in scores.items():
            if value is not None:
                by_essay[r["essay_key"]][key].append(value)

    per_essay: dict[str, dict[str, dict[str, Any]]] = {}
    for essay_key, score_lists in by_essay.items():
        per_essay[essay_key] = {}
        for key, values in score_lists.items():
            if len(values) < 2:
                continue  # desvio padrão exige ao menos 2 execuções
            mean = statistics.mean(values)
            std = statistics.stdev(values)
            per_essay[essay_key][key] = {
                "n_runs": len(values),
                "mean": mean,
                "std": std,
                "cv": (std / mean) if mean else None,
            }

    summary: dict[str, dict[str, Any]] = {}
    for key in (*COMPETENCY_KEYS, "total"):
        stds = [e[key]["std"] for e in per_essay.values() if key in e]
        cvs = [e[key]["cv"] for e in per_essay.values() if key in e and e[key]["cv"] is not None]
        if stds:
            summary[key] = {
                "n_redacoes": len(stds),
                "std_medio": statistics.mean(stds),
                "cv_medio": statistics.mean(cvs) if cvs else None,
            }

    return {"per_essay": per_essay, "summary": summary}
