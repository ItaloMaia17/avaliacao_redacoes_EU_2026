"""
Matriz de competências do ENEM para avaliação de redações.

Fonte: INEP (Instituto Nacional de Estudos e Pesquisas Educacionais Anísio Teixeira)
       Matriz de Referência de Redação — ENEM

Estrutura:
  - 5 competências, cada uma com:
    • código (C1–C5)
    • nome curto (para gráficos)
    • descrição completa (para prompts)
    • 6 níveis de desempenho (rubricas): 0, 40, 80, 120, 160, 200 pontos
"""

from __future__ import annotations

from typing import TypedDict

# ── definição de tipos ───────────────────────────────────────────────────────

class RubricLevel(TypedDict):
    """Um nível de desempenho dentro de uma competência."""
    score: int          # 0, 40, 80, 120, 160, 200
    name: str           # "Excelente", "Bom", "Mediano", etc.
    description: str    # texto completo da rubrica


class CompetencyData(TypedDict):
    """Uma competência completa com sua rubrica."""
    code: str                    # "C1", "C2", etc.
    name: str                    # "Escrita formal", "Proposta", etc.
    full: str                    # descrição completa
    rubrics: list[RubricLevel]   # 6 níveis de desempenho


# ── dados das 5 competências ────────────────────────────────────────────────

COMPETENCIES: dict[int, CompetencyData] = {
    1: {
        "code": "C1",
        "name": "Escrita formal",
        "full": "Domínio da modalidade escrita formal da língua portuguesa",
        "rubrics": [
            {
                "score": 200,
                "name": "Excelente",
                "description": (
                    "Demonstra excelente domínio da modalidade escrita formal da língua portuguesa "
                    "e de escolha de registro. Desvios gramaticais ou de convenções da escrita serão "
                    "aceitos somente como excepcionalidade e quando não caracterizarem reincidência."
                ),
            },
            {
                "score": 160,
                "name": "Bom",
                "description": (
                    "Demonstra bom domínio da modalidade escrita formal da língua portuguesa "
                    "e de escolha de registro, com poucos desvios gramaticais e de convenções da escrita."
                ),
            },
            {
                "score": 120,
                "name": "Mediano",
                "description": (
                    "Demonstra domínio mediano da modalidade escrita formal da língua portuguesa "
                    "e de escolha de registro, com alguns desvios gramaticais e de convenções da escrita."
                ),
            },
            {
                "score": 80,
                "name": "Insuficiente",
                "description": (
                    "Demonstra domínio insuficiente da modalidade escrita formal da língua portuguesa, "
                    "com muitos desvios gramaticais, de escolha de registro e de convenções da escrita."
                ),
            },
            {
                "score": 40,
                "name": "Precário",
                "description": (
                    "Demonstra domínio precário da modalidade escrita formal da língua portuguesa, "
                    "de forma sistemática, com diversificados e frequentes desvios gramaticais, "
                    "de escolha de registro e de convenções da escrita."
                ),
            },
            {
                "score": 0,
                "name": "Nenhum",
                "description": "Demonstra desconhecimento da modalidade escrita formal da língua portuguesa.",
            },
        ],
    },
    2: {
        "code": "C2",
        "name": "Proposta",
        "full": "Compreensão da proposta e aplicação de conceitos das áreas de conhecimento",
        "rubrics": [
            {
                "score": 200,
                "name": "Excelente",
                "description": (
                    "Desenvolve o tema por meio de argumentação consistente, a partir de um"
                    "repertório sociocultural produtivo, e apresenta excelente domínio do texto"
                    "dissertativo-argumentativo."
                ),
            },
            {
                "score": 160,
                "name": "Bom",
                "description": (
                    "Desenvolve o tema por meio de argumentação consistente e apresenta bom"
                    "domínio do texto dissertativo-argumentativo, com proposição, argumentação"
                    "e conclusão."
                ),
            },
            {
                "score": 120,
                "name": "Mediano",
                "description": (
                    "Desenvolve o tema por meio de argumentação previsível e apresenta"
                    "domínio mediano do texto dissertativo-argumentativo, com proposição,"
                    "argumentação e conclusão."
                ),
            },
            {
                "score": 80,
                "name": "Insuficiente",
                "description": (
                    "Desenvolve o tema recorrendo à cópia de trechos dos textos motivadores"
                    "ou apresenta domínio insuficiente do texto disertativo-argumentativo,"
                    "não atendendo à estrutura com proposição, argumentação e conclusão."
                ),
            },
            {
                "score": 40,
                "name": "Precário",
                "description": (
                    "Apresenta o assunto, tangenciando o tema, ou demonstra domínio precário do"
                    "texto dissertativo-argumentativo, com traços constantes de outros tipos textuais."
                ),
            },
            {
                "score": 0,
                "name": "Nenhum",
                "description": "Fuga ao tema/não atendimento à estrutura dissertativo-argumentativa."
                "Nestes casos a redação recebe nota 0 (Zero) e é anulada",
            },
        ],
    },
    3: {
        "code": "C3",
        "name": "Argumentação",
        "full": "Organização e seleção de informações, fatos, opiniões e argumentos",
        "rubrics": [
            {
                "score": 200,
                "name": "Excelente",
                "description": (
                    "Apresenta informações, fatos e opiniões relacionados ao tema proposto,"
                    "de forma consistente e organizada, configurando autoria, em defesa de um ponto de vista"
                ),
            },
            {
                "score": 160,
                "name": "Bom",
                "description": (
                    "Apresenta informações, fatos e opiniões relacionados ao tema, de forma"
                    "organizada, com indícios de autoria, em defesa de um ponto de vista."
                ),
            },
            {
                "score": 120,
                "name": "Mediano",
                "description": (
                    "Apresenta informações, fatos e opiniões relacionados ao tema, limitados aos"
                    "argumentos dos textos motivadores e pouco organizados, em defesa de um ponto de vista."
                ),
            },
            {
                "score": 80,
                "name": "Insuficiente",
                "description": (
                    "Apresenta informações, fatos e opiniões relacionados ao tema, mas"
                    "desorganizados ou contraditórios e limitados aos argumentos dos textos"
                    "motivadores, em defesa de um ponto de vista. "
                    
                ),
            },
            {
                "score": 40,
                "name": "Precário",
                "description": (
                    "Apresenta informações, fatos e opiniões pouco relacionados ao tema ou"
                    "incoerentes e sem defesa de um ponto de vista."
                ),
            },
            {
                "score": 0,
                "name": "Nenhum",
                "description": (
                    "Apresenta informações, fatos e opiniões não relacionados ao tema e sem"
                    "defesa de um ponto de vista."
                ),
            },
        ],
    },
    4: {
        "code": "C4",
        "name": "Mecanismos linguísticos",
        "full": "Uso de mecanismos linguísticos para a construção da argumentação",
        "rubrics": [
            {
                "score": 200,
                "name": "Excelente",
                "description": (
                    "Articula bem as partes do texto e apresenta repertório diversificado de recursos coesivos."
                ),
            },
            {
                "score": 160,
                "name": "Bom",
                "description": (
                    "Articula as partes do texto, com poucas inadequações, e apresenta repertório"
                    "diversificado de recursos coesivos."
                ),
            },
            {
                "score": 120,
                "name": "Mediano",
                "description": (
                    "Articula as partes do texto, de forma mediana, com inadequações, e apresenta repertório pouco diversificado de recursos coesivos."
                ),
            },
            {
                "score": 80,
                "name": "Insuficiente",
                "description": (
                    "Articula as partes do texto, de forma insuficiente, com muitas inadequações, e"
                    "apresenta repertório limitado de recursos coesivos."
                ),
            },
            {
                "score": 40,
                "name": "Precário",
                "description": (
                    "Articula as partes do texto de forma precária"
                ),
            },
            {
                "score": 0,
                "name": "Nenhum",
                "description": (
                    "Não articula as informações"
                ),
            },
        ],
    },
    5: {
        "code": "C5",
        "name": "Intervenção",
        "full": "Elaboração de proposta de intervenção",
        "rubrics": [
            {
                "score": 200,
                "name": "Excelente",
                "description": (
                    "Elabora muito bem proposta de intervenção, detalhada, relacionada ao tema"
                    "e articulada à discussão desenvolvida no texto."
                ),
            },
            {
                "score": 160,
                "name": "Bom",
                "description": (
                    "Elabora bem proposta de intervenção relacionada ao tema e articulada à"
                    "discussão desenvolvida no texto."
                ),
            },
            {
                "score": 120,
                "name": "Mediano",
                "description": (
                    "Elabora, de forma mediana, proposta de intervenção relacionada ao tema e"
                    "articulada à discussão desenvolvida no texto. "
                ),
            },
            {
                "score": 80,
                "name": "Insuficiente",
                "description": (
                    "Elabora, de forma insulficiente, proposta de intervenção relacionada ao tema,"
                    "ou não articulada com a discussão desenvolvida no texto."
                ),
            },
            {
                "score": 40,
                "name": "Precário",
                "description": (
                    "Apresenta proposta de intervenção vaga, precária ou relacionada apenas aoassunto."
                ),
            },
            {
                "score": 0,
                "name": "Nenhum",
                "description": (
                    "Não apresenta proposta de intervenção ou apresenta proposta não"
                    "relacionada ao tema ou ao assunto"
                ),
            },
        ],
    },
}

# ── constantes derivadas para acesso rápido ──────────────────────────────────

N_COMPETENCIES = len(COMPETENCIES)
VALID_SCORES = {0, 40, 80, 120, 160, 200}

# Dicts de acesso por ID
COMPETENCY_CODES = {i: c["code"] for i, c in COMPETENCIES.items()}
COMPETENCY_NAMES = {i: c["name"] for i, c in COMPETENCIES.items()}
COMPETENCY_FULL = {i: c["full"] for i, c in COMPETENCIES.items()}

# String formatada para usar nos prompts
COMPETENCY_DESCRIPTIONS = "\n".join(
    f"{c['code']} — {c['full']}"
    for c in COMPETENCIES.values()
)

# Para gráficos: "C1\nEscrita formal"
COMPETENCY_LABELS = {
    i: f"{c['code']}\n{c['name']}"
    for i, c in COMPETENCIES.items()
}


# ── utilitários ──────────────────────────────────────────────────────────────

def get_rubric(competency_id: int, score: int) -> RubricLevel | None:
    """Retorna a rubrica para uma competência e nota específicas.
    
    Exemplo:
        rubric = get_rubric(1, 160)
        print(rubric['description'])  # "Demonstra bom domínio..."
    """
    if competency_id not in COMPETENCIES:
        return None
    for rubric in COMPETENCIES[competency_id]["rubrics"]:
        if rubric["score"] == score:
            return rubric
    return None


def get_rubrics_for_competency(competency_id: int) -> list[RubricLevel] | None:
    """Retorna todas as rubricas de uma competência."""
    if competency_id not in COMPETENCIES:
        return None
    return COMPETENCIES[competency_id]["rubrics"]


def validate_scores(competencies: list[int]) -> tuple[bool, list[str]]:
    """Valida se os scores estão na escala válida.
    
    Retorna (is_valid, error_messages).
    """
    errors = []
    if len(competencies) != N_COMPETENCIES:
        errors.append(f"Esperado {N_COMPETENCIES} competências, recebido {len(competencies)}")
    for i, score in enumerate(competencies, start=1):
        if score not in VALID_SCORES:
            errors.append(
                f"Competência {i}: score {score} não está em {sorted(VALID_SCORES)}"
            )
    return len(errors) == 0, errors


def format_rubrics_for_prompt(include_full: bool = True) -> str:
    """Formata todas as rubricas para incluir no prompt do sistema.
    
    Args:
        include_full: se True, inclui a descrição completa de cada nível.
                     se False, apenas o código, nome curto e score.
    """
    lines = ["Rubricas de avaliação por competência:\n"]
    
    for comp_id in range(1, N_COMPETENCIES + 1):
        comp = COMPETENCIES[comp_id]
        lines.append(f"\n{comp['code']} — {comp['full']}")
        
        for rubric in comp["rubrics"]:
            if include_full == False:
                lines.append(f"  • {rubric['score']:>3} pts: {rubric['name']}")
            lines.append(
                f"  {rubric['score']:>3} pts ({rubric['name']:6s}): {rubric['description']}"
            )
    return "\n".join(lines)


if __name__ == "__main__":
    # Teste: imprime todas as competências e rubricas
    print(format_rubrics_for_prompt(include_full=True))