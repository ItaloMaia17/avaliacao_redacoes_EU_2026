"""
Construção dos prompts para as 6 condições do desenho fatorial 3x2
(Seção 3.4 da metodologia).

Inclui opcionalmente as rubricas de cada competência — recomendado para
melhorar a qualidade da avaliação, especialmente em modo estruturado e CoT.
"""

from __future__ import annotations

from typing import Any

from competencies import (
    N_COMPETENCIES,
    VALID_SCORES,
    format_rubrics_for_prompt,
)

TECHNIQUES = ("zero-shot", "few-shot", "cot")
STRUCTURES = ("holistica", "estruturada")


def _essay_block(essay: dict[str, Any]) -> str:
    """Monta o bloco com texto motivador, proposta e redação a avaliar."""
    return (
        f"Texto motivador:\n{essay['supporting_text']}\n\n"
        f"Proposta de redação:\n{essay['prompt']}\n\n"
        f"Redação a ser avaliada:\n{essay['essay_text']}\n"
    )


def _output_schema_instructions(structure: str) -> str:
    """Instruções sobre o formato esperado da resposta (JSON)."""
    if structure == "holistica":
        return (
            "Responda SOMENTE em JSON válido, sem texto antes ou depois, no formato:\n"
            '{"nota_final": <inteiro de 0 a 1000>}'
        )

    # campos = ",\n".join(
    #     f'  "competencia_{i}": {{"nota": <um dos valores {sorted(VALID_SCORES)}>, '
    #     f'"justificativa": "<texto>"}}'
    #     for i in range(1, N_COMPETENCIES + 1)
    # )
    campos = ",\n".join(
        f'  "competencia_{i}": <um dos valores {sorted(VALID_SCORES)}>'
        for i in range(1, N_COMPETENCIES + 1)
    )
    return (
        "Responda SOMENTE em JSON válido, sem texto antes ou depois, "
        "contendo apenas a nota de cada competência, no formato:\n"
        "{\n" + campos + "\n}"
    )


def build_prompt(
    essay: dict[str, Any],
    technique: str,
    structure: str,
    few_shot_examples: list[tuple[dict[str, Any], dict[str, Any]]] | None = None,
    include_rubrics: bool = True,
) -> dict[str, str]:
    """Monta o prompt (system + user) para uma das 6 condições experimentais.

    Args:
        essay: redação a avaliar, com campos: essay_text, supporting_text, prompt, etc.
        technique: "zero-shot", "few-shot", ou "cot"
        structure: "holistica" ou "estruturada"
        few_shot_examples: lista de tuplas (redação_exemplo, notas_exemplo).
                          Fixas e idênticas para todos os modelos.
        include_rubrics: se True (padrão), inclui as rubricas completas no system prompt.
                        Melhora a qualidade, mas adiciona ~1.200 tokens.

    Returns:
        dict com chaves "system" e "user" — pronto para chamar a API.
    """
    if technique not in TECHNIQUES:
        raise ValueError(f"technique deve ser um de {TECHNIQUES}")
    if structure not in STRUCTURES:
        raise ValueError(f"structure deve ser um de {STRUCTURES}")

    system_parts = [
        "Você é um avaliador especializado em correção de redações do ENEM.",
        "Avalie a redação segundo a matriz de referência do ENEM.",
    ]

    # Rubricas completas (opcional, mas recomendado)
    if include_rubrics:
        system_parts.append(format_rubrics_for_prompt(include_full=False))

    if structure == "estruturada":
        system_parts.append(
            "Atribua uma nota individual para cada competência, dentro do "
            f"conjunto de valores válidos {sorted(VALID_SCORES)}. "
            "Não forneça justificativas, explicações ou comentários."
        )
    else:
        system_parts.append(
            "Atribua uma única nota geral à redação, na escala de 0 a 1000, "
            "sem decompor a avaliação por competência."
        )

    if technique == "cot":
        system_parts.append(
            # "Antes de atribuir a(s) nota(s) final(is), apresente explicitamente "
            # "as etapas de sua análise: (1) leitura e compreensão da proposta, "
            # "(2) identificação da tese e argumentação, (3) avaliação de cada competência "
            # "com base nas rubricas fornecidas, (4) proposta de intervenção. "
            # "Estruture seu raciocínio de forma clara."
            "Analise cuidadosamente a redação passo a passo antes de atribuir "
            "a(s) nota(s). Considere a proposta, a tese, a argumentação e os "
            "critérios de cada competência. Não apresente o raciocínio, análise "
            "ou justificativas na resposta final. Retorne somente o JSON solicitado."
        )

    system_parts.append(_output_schema_instructions(structure))
    system_prompt = "\n\n".join(system_parts)

    # User prompt
    user_parts: list[str] = []
    if technique == "few-shot":
        if not few_shot_examples:
            raise ValueError(
                "technique='few-shot' exige `few_shot_examples` "
                "(2 exemplos fixos, com nota 560 e 880)."
            )
        user_parts.append("Exemplos de redações já corrigidas:\n")
        for i, (ex_essay, ex_scores) in enumerate(few_shot_examples, start=1):
            user_parts.append(
                f"--- Exemplo {i} ---\n{_essay_block(ex_essay)}"
                f"Notas atribuídas: {ex_scores}\n"
            )
        user_parts.append("--- Fim dos exemplos ---\n")

    user_parts.append(_essay_block(essay))
    user_prompt = "\n".join(user_parts)

    # print("SYSTEM:", len(system_prompt), "caracteres")
    # print("USER:", len(user_prompt), "caracteres")
    # print(
    #     "RUBRICAS:",
    #     len(format_rubrics_for_prompt(include_full=False))
    # )
    # print(
    #     "ESSAY BLOCK:",
    #     len(_essay_block(essay))
    # )

    return {"system": system_prompt, "user": user_prompt}


ALL_CONDITIONS = [
    (technique, structure) for technique in TECHNIQUES for structure in STRUCTURES
]
