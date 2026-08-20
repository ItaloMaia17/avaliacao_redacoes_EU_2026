"""
Prompts para as 6 condições experimentais: 3 técnicas (zero-shot, few-shot, CoT) x
2 estruturas (holística, estruturada por competência).
"""

from __future__ import annotations

from typing import Any

COMPETENCIES = [
    "Competência 1 — Domínio da modalidade escrita formal da língua portuguesa",
    "Competência 2 — Compreensão da proposta e aplicação de conceitos das áreas de conhecimento",
    "Competência 3 — Organização e seleção de informações, fatos, opiniões e argumentos",
    "Competência 4 — Uso de mecanismos linguísticos para a construção da argumentação",
    "Competência 5 — Elaboração de proposta de intervenção",
]
VALID_SCORES = [0, 40, 80, 120, 160, 200]

TECHNIQUES = ("zero-shot", "few-shot", "cot")
STRUCTURES = ("holistica", "estruturada")


def _essay_block(essay: dict[str, Any]) -> str:
    return (
        f"Texto motivador:\n{essay['supporting_text']}\n\n"
        f"Proposta de redação:\n{essay['prompt']}\n\n"
        f"Redação a ser avaliada:\n{essay['essay_text']}\n"
    )


def _output_schema_instructions(structure: str) -> str:
    if structure == "holistica":
        return (
            "Responda SOMENTE em JSON válido, sem texto antes ou depois, no formato:\n"
            '{"nota_final": <inteiro de 0 a 1000>, "justificativa": "<texto>"}'
        )

    campos = ",\n".join(
        f'  "competencia_{i}": {{"nota": <um dos valores {VALID_SCORES}>, '
        f'"justificativa": "<texto>"}}'
        for i in range(1, 6)
    )
    return (
        "Responda SOMENTE em JSON válido, sem texto antes ou depois, com uma "
        "nota e uma justificativa para cada uma das cinco competências, no "
        "formato:\n{\n" + campos + "\n}"
    )


def build_prompt(
    essay: dict[str, Any],
    technique: str,
    structure: str,
    few_shot_examples: list[tuple[dict[str, Any], dict[str, Any]]] | None = None,
) -> dict[str, str]:
    """Monta o prompt (system + user) para uma das 6 condições experimentais.

    `few_shot_examples`: lista de até 2 tuplas (redação_exemplo, notas_exemplo)
    fixas, idênticas para todos os modelos e todas as redações (Seção 3.4).
    `notas_exemplo` deve ter o mesmo formato esperado na saída (ver
    `_output_schema_instructions`).
    """
    if technique not in TECHNIQUES:
        raise ValueError(f"technique deve ser um de {TECHNIQUES}")
    if structure not in STRUCTURES:
        raise ValueError(f"structure deve ser um de {STRUCTURES}")

    system_parts = [
        "Você é um avaliador especializado em correção de redações do ENEM.",
        "Avalie a redação a seguir segundo a matriz de referência do ENEM, "
        "que considera as seguintes competências:",
        "\n".join(COMPETENCIES),
    ]
    system_parts.append(
    "Sua resposta deve conter apenas um objeto JSON válido. "
    "Não utilize markdown, não utilize blocos ```json e não escreva qualquer texto fora do JSON."
    )

    if structure == "estruturada":
        system_parts.append(
            "Atribua uma nota individual para cada competência, dentro do "
            f"conjunto de valores válidos {VALID_SCORES}."
        )
    else:
        system_parts.append(
            "Atribua uma única nota geral à redação, na escala de 0 a 1000, "
            "sem decompor a avaliação por competência."
        )

    if technique == "cot":
        system_parts.append(
            "Antes de atribuir a nota final, apresente explicitamente as "
            "etapas de sua análise (leitura da proposta, identificação da "
            "tese, avaliação da argumentação e da proposta de intervenção) "
            "dentro do(s) campo(s) 'justificativa'."
        )

    system_parts.append(_output_schema_instructions(structure))
    system_prompt = "\n\n".join(system_parts)

    user_parts: list[str] = []
    if technique == "few-shot":
        if not few_shot_examples:
            raise ValueError(
                "technique='few-shot' exige `few_shot_examples` "
                "(2 exemplos fixos, ver Seção 3.4)."
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

    return {"system": system_prompt, "user": user_prompt}


ALL_CONDITIONS = [
    (technique, structure) for technique in TECHNIQUES for structure in STRUCTURES
]
