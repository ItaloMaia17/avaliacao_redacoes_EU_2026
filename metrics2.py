"""
Métricas de avaliação para o resumo expandido.

Desenho experimental:
- Precisão: QWK entre a nota geral gerada pelo modelo e a nota geral humana.
- Consistência: desvio padrão e coeficiente de variação (CV) da nota geral
  entre 3 execuções independentes da mesma redação.
- Amostra: 100 redações selecionadas aleatoriamente de um conjunto de
  384 redações.
- Técnica: zero-shot.
- Estrutura: estruturada por competência do ENEM.
- Modelos: Qwen e GPT-OSS.

Na precisão, quando uma redação possui mais de uma anotação humana, cada
anotação é tratada como uma comparação independente. Assim, uma redação com
k avaliadores gera k pares (modelo, humano).

A saída estruturada do modelo contém as cinco notas das competências. A nota
geral do modelo é calculada pela soma das cinco competências.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any

from sklearn.metrics import cohen_kappa_score

from competencies import VALID_SCORES


COMPETENCY_KEYS = [f"competencia_{i}" for i in range(1, 6)]


# Configuração fixa do resumo expandido.
SAMPLE_SIZE = 100
N_CONSISTENCY_RUNS = 3
STRUCTURE = "estruturada"
TECHNIQUE = "zero-shot"



def snap_to_valid_scale(
    value: float, valid: set[int] = VALID_SCORES
) -> int:
    """Arredonda uma nota para o valor válido mais próximo.

    A escala válida do ENEM é {0, 40, 80, 120, 160, 200} por competência.
    A função é mantida para normalizar eventuais valores numéricos fora da
    escala antes de calcular métricas que dependam da escala discreta.
    """
    return min(valid, key=lambda v: abs(v - value))



def extract_model_scores(
    parsed_output: dict[str, Any] | None,
    structure: str = STRUCTURE,
) -> dict[str, float | None]:
    """Extrai as notas do modelo e calcula a nota geral.

    Para a configuração do resumo expandido (estrutura estruturada), espera:

        competencia_1, competencia_2, ..., competencia_5

    A nota geral é a soma das cinco notas.
    """
    scores: dict[str, float | None] = {key: None for key in COMPETENCY_KEYS}
    scores["total"] = None

    if parsed_output is None:
        return scores

    if structure == "holistica":
        total = parsed_output.get("nota_final")
        if isinstance(total, (int, float)):
            scores["total"] = float(total)
        return scores

    if structure != STRUCTURE:
        raise ValueError(
            f"Estrutura '{structure}' não faz parte do desenho do resumo expandido. "
            f"Use '{STRUCTURE}'."
        )

    competencias: list[float] = []

    for key in COMPETENCY_KEYS:
        campo = parsed_output.get(key)

        if not isinstance(campo, (int, float)):
            return scores

        nota = float(campo)
        scores[key] = nota
        competencias.append(nota)

    # A nota geral é a soma das cinco competências.
    scores["total"] = sum(competencias)
    return scores



def _qwk_from_pairs(pairs: list[tuple[float, float]]) -> float | None:
    """Calcula QWK para uma lista de pares (modelo, humano)."""
    if len(pairs) < 2:
        return None

    y_model = [model for model, _ in pairs]
    y_human = [human for _, human in pairs]

    return float(cohen_kappa_score(y_model, y_human, weights="quadratic"))



def precision_metrics(
    results: list[dict[str, Any]],
    essays_by_key: dict[str, dict[str, Any]],
    structure: str = STRUCTURE,
) -> dict[str, Any]:
    """Calcula somente o QWK da nota geral modelo × humano.

    Não calcula MAE e não calcula métricas por competência, pois a precisão
    do resumo expandido é avaliada exclusivamente pela nota geral.

    Cada anotação humana disponível para uma redação gera um par independente:

        (nota_geral_modelo, nota_geral_humana)
    """
    pairs: list[tuple[float, float]] = []
    n_redacoes = set()

    for result in results:
        if result.get("parse_failed") or result.get("dry_run"):
            continue

        essay_key = result.get("essay_key")
        if essay_key is None:
            continue

        essay = essays_by_key.get(essay_key)
        if essay is None:
            continue

        annotations = essay.get("annotations") or []
        if not annotations:
            continue

        scores = extract_model_scores(result.get("parsed_output"), structure)
        model_total = scores["total"]

        if model_total is None:
            continue

        for annotation in annotations:
            human_total = annotation.get("total")

            if human_total is None:
                continue

            pairs.append((model_total, float(human_total)))
            n_redacoes.add(essay_key)

    qwk = _qwk_from_pairs(pairs)

    if qwk is None:
        return {}

    return {
        "n_redacoes": len(n_redacoes),
        "n_pares": len(pairs),
        "qwk": qwk,
    }



def human_human_agreement(
    essays_by_key: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Calcula QWK da nota geral entre avaliadores humanos.

    Esta métrica é mantida apenas como referência contextual da concordância
    humana no conjunto utilizado. Não há cálculo por competência nem MAE.

    Para cada redação com dois ou mais avaliadores, o primeiro avaliador é
    comparado aos demais, evitando duplicar comparações simétricas.
    """
    pairs: list[tuple[float, float]] = []
    n_redacoes_com_multiplos = 0

    for essay in essays_by_key.values():
        annotations = essay.get("annotations") or []
        if len(annotations) < 2:
            continue

        base = annotations[0]
        base_total = base.get("total")
        if base_total is None:
            continue

        outros_validos = 0
        for other in annotations[1:]:
            other_total = other.get("total")
            if other_total is None:
                continue

            pairs.append((float(base_total), float(other_total)))
            outros_validos += 1

        if outros_validos:
            n_redacoes_com_multiplos += 1

    qwk = _qwk_from_pairs(pairs)

    result: dict[str, Any] = {
        "n_redacoes_com_multiplos_avaliadores": n_redacoes_com_multiplos,
        "n_pares": len(pairs),
    }

    if qwk is not None:
        result["qwk"] = qwk

    return result



def consistency_metrics(
    results: list[dict[str, Any]],
    structure: str = STRUCTURE,
) -> dict[str, Any]:
    """Calcula consistência somente para a nota geral.

    Para cada redação, são reunidas as notas gerais das execuções
    independentes. O desvio padrão amostral e o CV são calculados por redação.
    Depois, os valores são agregados pela média entre as redações.

    O desenho do resumo expandido prevê N=3 execuções por redação.
    """
    by_essay: dict[str, list[float]] = defaultdict(list)

    for result in results:
        if result.get("parse_failed") or result.get("dry_run"):
            continue

        essay_key = result.get("essay_key")
        if essay_key is None:
            continue

        scores = extract_model_scores(result.get("parsed_output"), structure)
        total = scores["total"]

        if total is not None:
            by_essay[essay_key].append(float(total))

    per_essay: dict[str, dict[str, Any]] = {}

    for essay_key, values in by_essay.items():
        if len(values) < 2:
            continue

        mean = statistics.mean(values)
        std = statistics.stdev(values)
        cv = (100 * std / mean) if mean else None

        per_essay[essay_key] = {
            "n_runs": len(values),
            "mean": mean,
            "std": std,
            "cv": cv,
        }

    stds = [item["std"] for item in per_essay.values()]
    cvs = [item["cv"] for item in per_essay.values() if item["cv"] is not None]

    summary: dict[str, Any] = {
        "n_redacoes": len(per_essay),
        "n_runs_esperado": N_CONSISTENCY_RUNS,
        "std_medio": statistics.mean(stds) if stds else None,
        "cv_medio": statistics.mean(cvs) if cvs else None,
    }

    # Compatibilidade com scripts que esperam um resumo em "summary.total".
    total_summary = dict(summary)
    return {
        "per_essay": per_essay,
        "summary": {
            "total": total_summary,
            **total_summary,
        },
    }