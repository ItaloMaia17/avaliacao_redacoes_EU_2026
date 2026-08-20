"""
Pré-processamento do subconjunto sourceAWithGraders do dataset
kamel-usp/aes_enem_dataset (Silveira, Barbosa e Mauá, 2024), seguindo a
Seção 3.1/3.2 da metodologia.

Etapas:
  (i)   Verificação de integridade
  (ii)  Agrupamento para inferência (uma redação -> múltiplas notas humanas)
  (iii) Limpeza textual

IMPORTANTE sobre a chave de identificação da redação:
    O campo `id` do dataset NÃO é único globalmente — ele se repete entre
    temas diferentes (funciona como um nome de arquivo dentro da pasta de
    cada tema/`id_prompt`). A chave real de uma redação é o par
    (id_prompt, id). Usar apenas `id` como chave de agrupamento juntaria
    redações completamente diferentes que por acaso compartilham o mesmo
    `id` em temas distintos.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

VALID_COMPETENCY_SCORES = {0, 40, 80, 120, 160, 200}
N_COMPETENCIES = 5  # + 1 nota final = 6 valores no campo `grades`


@dataclass
class HumanAnnotation:
    reference: str          # identificador do avaliador (ex.: "grader_a")
    competencies: list[int]  # 5 notas, uma por competência
    total: int               # nota final (soma das competências)


@dataclass
class Essay:
    essay_key: str           # chave única = f"{id_prompt}::{id}"
    id_prompt: str
    essay_text: str
    supporting_text: str
    prompt: str               # comando / proposta de redação
    essay_year: Optional[int]
    annotations: list[HumanAnnotation] = field(default_factory=list)

    @property
    def n_graders(self) -> int:
        return len(self.annotations)


def clean_text(text: Optional[str]) -> str:
    """Etapa (iii): remoção de caracteres de controle e artefatos de codificação.

    Não realiza qualquer correção ortográfica ou alteração de conteúdo
    linguístico — apenas normalização de codificação e espaçamento.
    """
    if not text:
        return ""
    # Normaliza para a forma NFC (corrige combinações de acentos malformadas
    # que podem surgir de raspagem web com encoding inconsistente)
    text = unicodedata.normalize("NFC", text)
    # Remove caracteres de controle, preservando quebras de linha e tabulação
    text = "".join(
        ch for ch in text
        if ch in ("\n", "\t") or unicodedata.category(ch)[0] != "C"
    )
    # Colapsa espaços/tabs repetidos sem destruir a separação de parágrafos
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_valid_competency_row(row: dict[str, Any]) -> bool:
    """Etapa (i): verificação de integridade de uma linha bruta do dataset."""
    grades = row.get("grades")
    if not grades or len(grades) != N_COMPETENCIES + 1:
        return False

    competencies, total = list(grades[:N_COMPETENCIES]), grades[N_COMPETENCIES]

    if any(g not in VALID_COMPETENCY_SCORES for g in competencies):
        return False
    if total != sum(competencies):
        return False
    if not row.get("prompt") or not row.get("essay_text"):
        return False

    return True


def group_rows(rows: list[dict[str, Any]]) -> tuple[list[Essay], int]:
    """Aplica (i) e (ii) sobre uma lista de linhas brutas já carregadas.

    Separado de `load_and_group` para permitir testes locais sem depender
    da rede (ex.: usando um arquivo de amostra em vez do Hugging Face Hub).
    """
    discarded = 0
    essays: dict[str, Essay] = {}
    text_mismatches: list[str] = []

    for row in rows:
        if not is_valid_competency_row(row):
            discarded += 1
            continue

        key = f"{row['id_prompt']}::{row['id']}"
        grades = row["grades"]
        annotation = HumanAnnotation(
            reference=row.get("reference", "unknown"),
            competencies=list(grades[:N_COMPETENCIES]),
            total=grades[N_COMPETENCIES],
        )

        cleaned_essay_text = clean_text(row["essay_text"])

        if key not in essays:
            essays[key] = Essay(
                essay_key=key,
                id_prompt=row["id_prompt"],
                essay_text=cleaned_essay_text,
                supporting_text=clean_text(row.get("supporting_text", "")),
                prompt=clean_text(row.get("prompt", "")),
                essay_year=row.get("essay_year"),
            )
        elif essays[key].essay_text != cleaned_essay_text:
            # Sanidade: o texto deveria ser idêntico entre anotações da
            # mesma redação. Mantém a primeira versão e registra o caso.
            text_mismatches.append(key)

        essays[key].annotations.append(annotation)

    if text_mismatches:
        print(
            f"[aviso] {len(text_mismatches)} redação(ões) com texto "
            f"divergente entre anotações da mesma chave: {text_mismatches}"
        )

    return list(essays.values()), discarded


def load_and_group(
    dataset_name: str = "kamel-usp/aes_enem_dataset",
    subset: str = "sourceAWithGraders",
) -> list[Essay]:
    """Carrega o dataset do Hugging Face Hub e aplica (i) e (ii).

    Requer rede e a biblioteca `datasets` (`pip install datasets`).
    """
    from datasets import load_dataset  # import local: só é necessário aqui

    raw = load_dataset(dataset_name, subset)
    all_rows: list[dict[str, Any]] = []
    for split in raw.keys():
        all_rows.extend(raw[split])

    essays, discarded = group_rows(all_rows)

    print(
        f"[integridade] linhas brutas: {len(all_rows)} | "
        f"descartadas: {discarded} | "
        f"redações únicas após agrupamento: {len(essays)}"
    )

    n_with_multiple = sum(1 for e in essays if e.n_graders > 1)
    print(f"[agrupamento] redações com mais de 1 avaliador: {n_with_multiple}")

    return essays


def save_processed(
    essays: list[Essay], path: str = "data/processed_essays.jsonl"
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for essay in essays:
            f.write(json.dumps(asdict(essay), ensure_ascii=False) + "\n")
    print(f"[ok] {len(essays)} redações salvas em {path}")


def load_rows_from_jsonl(path: str) -> list[dict[str, Any]]:
    """Utilitário para testes locais: lê linhas brutas de um .jsonl de amostra."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


if __name__ == "__main__":
    essays = load_and_group()
    save_processed(essays)
