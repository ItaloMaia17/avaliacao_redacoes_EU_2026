"""
Análise exploratória do subconjunto sourceAWithGraders após pré-processamento.

Entrada: data/processed_essays.jsonl (gerado por preprocessing.py)
Saída:   data/figures/ com os gráficos gerados

Gráficos produzidos:
  1. Boxplot das notas por competência (por avaliador)
  2. Histograma da nota total (por avaliador)
  3. Distribuição das faixas de nota total (por avaliador)
  4. Boxplot do tamanho dos textos (número de caracteres)
  5. Dispersão nota total × tamanho do texto
  6. Variabilidade interavaliador (DP por redação com 2+ avaliadores)
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns

# ── configuração visual ──────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = sns.color_palette("muted")
FIG_DIR = Path("data/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

COMPETENCY_LABELS = {
    "competencia_1": "C1\nEscrita formal",
    "competencia_2": "C2\nProposta",
    "competencia_3": "C3\nArgumentação",
    "competencia_4": "C4\nMecanismos\nlinguísticos",
    "competencia_5": "C5\nIntervenção",
}
VALID_SCORES = [0, 40, 80, 120, 160, 200]

# ── carregamento ─────────────────────────────────────────────────────────────

def load_essays(path: str = "data/processed_essays.jsonl") -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def build_annotations_df(essays: list[dict]) -> pd.DataFrame:
    """Uma linha por (redação, avaliador) — nível de instância do dataset."""
    rows = []
    for e in essays:
        n_chars = len(e["essay_text"])
        n_words = len(e["essay_text"].split())
        for ann in e["annotations"]:
            row = {
                "essay_key": e["essay_key"],
                "id_prompt": e["id_prompt"],
                "essay_year": e.get("essay_year"),
                "reference": ann["reference"],
                "total": ann["total"],
                "n_chars": n_chars,
                "n_words": n_words,
            }
            for i, c in enumerate(ann["competencies"], start=1):
                row[f"competencia_{i}"] = c
            rows.append(row)
    return pd.DataFrame(rows)


def build_essays_df(essays: list[dict]) -> pd.DataFrame:
    """Uma linha por redação única — para métricas de texto."""
    rows = []
    for e in essays:
        competency_means = defaultdict(list)
        totals = []
        for ann in e["annotations"]:
            totals.append(ann["total"])
            for i, c in enumerate(ann["competencies"], start=1):
                competency_means[f"competencia_{i}"].append(c)
        row = {
            "essay_key": e["essay_key"],
            "id_prompt": e["id_prompt"],
            "essay_year": e.get("essay_year"),
            "n_graders": len(e["annotations"]),
            "n_chars": len(e["essay_text"]),
            "n_words": len(e["essay_text"].split()),
            "total_mean": sum(totals) / len(totals),
        }
        for key, vals in competency_means.items():
            row[f"{key}_mean"] = sum(vals) / len(vals)
        rows.append(row)
    return pd.DataFrame(rows)


# ── gráfico 1: boxplot de notas por competência ──────────────────────────────

def plot_competency_boxplots(df_ann: pd.DataFrame) -> None:
    comp_cols = list(COMPETENCY_LABELS.keys())
    melted = df_ann.melt(
        id_vars=["essay_key"],
        value_vars=comp_cols,
        var_name="competencia",
        value_name="nota",
    )
    melted["competencia"] = melted["competencia"].map(COMPETENCY_LABELS)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(
        data=melted, x="competencia", y="nota",
        hue="competencia", palette=PALETTE[:5], legend=False,
        width=0.5, linewidth=1.2,
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.4},
        ax=ax,
    )
    ax.set_yticks(VALID_SCORES)
    ax.set_xlabel("")
    ax.set_ylabel("Nota")
    ax.set_title("Distribuição das notas por competência\n(uma observação por avaliador)")
    fig.tight_layout()
    out = FIG_DIR / "01_boxplot_competencias.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[ok] {out}")


# ── gráfico 2: histograma da nota total ──────────────────────────────────────

def plot_total_histogram(df_ann: pd.DataFrame) -> None:
    bins = [i * 40 for i in range(26)]  # 0, 40, 80, …, 1000
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(df_ann["total"], bins=bins, color=PALETTE[0], edgecolor="white", linewidth=0.6)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(80))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(40))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Nota total (0–1000)")
    ax.set_ylabel("Frequência")
    ax.set_title("Histograma da nota total\n(uma observação por avaliador)")
    fig.tight_layout()
    out = FIG_DIR / "02_histograma_nota_total.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[ok] {out}")


# ── gráfico 3: distribuição por faixas de nota total ─────────────────────────

def plot_score_bands(df_ann: pd.DataFrame) -> None:
    bands = [(0, 200), (201, 400), (401, 600), (601, 800), (801, 1000)]
    labels = ["0–200", "201–400", "401–600", "601–800", "801–1000"]
    counts = [((df_ann["total"] >= lo) & (df_ann["total"] <= hi)).sum() for lo, hi in bands]
    pcts = [c / len(df_ann) * 100 for c in counts]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, pcts, color=PALETTE[:5], edgecolor="white", linewidth=0.8)
    for bar, pct, cnt in zip(bars, pcts, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{pct:.1f}%\n(n={cnt})",
            ha="center", va="bottom", fontsize=9,
        )
    ax.set_xlabel("Faixa de nota total")
    ax.set_ylabel("% de instâncias")
    ax.set_ylim(0, max(pcts) * 1.25)
    ax.set_title("Distribuição por faixa de nota total\n(uma observação por avaliador)")
    fig.tight_layout()
    out = FIG_DIR / "03_faixas_nota_total.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[ok] {out}")


# ── gráfico 4: boxplot do tamanho dos textos ─────────────────────────────────

def plot_text_length(df_ess: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, col, label in zip(
        axes,
        ["n_chars", "n_words"],
        ["Número de caracteres", "Número de palavras"],
    ):
        sns.boxplot(y=df_ess[col], color=PALETTE[2], width=0.4, linewidth=1.2,
                    flierprops={"marker": "o", "markersize": 3, "alpha": 0.4}, ax=ax)
        median = df_ess[col].median()
        mean = df_ess[col].mean()
        ax.axhline(mean, color="crimson", linestyle="--", linewidth=1, label=f"Média: {mean:.0f}")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=9)

    fig.suptitle("Tamanho dos textos (redações únicas)", y=1.01)
    fig.tight_layout()
    out = FIG_DIR / "04_tamanho_textos.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out}")


# ── gráfico 5: dispersão nota total × tamanho do texto ───────────────────────

def plot_score_vs_length(df_ess: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        df_ess["n_words"], df_ess["total_mean"],
        alpha=0.4, s=20, color=PALETTE[1],
    )
    # linha de tendência
    import numpy as np
    z = np.polyfit(df_ess["n_words"], df_ess["total_mean"], 1)
    p = np.poly1d(z)
    xs = sorted(df_ess["n_words"])
    ax.plot(xs, p(xs), color="crimson", linewidth=1.2, linestyle="--", label="Tendência linear")
    ax.set_xlabel("Número de palavras")
    ax.set_ylabel("Nota total (média entre avaliadores)")
    ax.set_title("Nota total × tamanho da redação")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = FIG_DIR / "05_nota_vs_tamanho.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[ok] {out}")


# ── gráfico 6: variabilidade interavaliador ───────────────────────────────────

def plot_interrater_variability(df_ess: pd.DataFrame, essays: list[dict]) -> None:
    import numpy as np
    multi = [e for e in essays if len(e["annotations"]) >= 2]
    if not multi:
        print("[aviso] nenhuma redação com 2+ avaliadores — gráfico 6 ignorado.")
        return

    sds: dict[str, list[float]] = {f"competencia_{i}": [] for i in range(1, 6)}
    sds["total"] = []

    for e in multi:
        totals = [a["total"] for a in e["annotations"]]
        sds["total"].append(float(np.std(totals, ddof=1)))
        for i in range(5):
            vals = [a["competencies"][i] for a in e["annotations"]]
            sds[f"competencia_{i+1}"].append(float(np.std(vals, ddof=1)))

    labels = [COMPETENCY_LABELS[f"competencia_{i}"] for i in range(1, 6)] + ["Total"]
    data = [sds[f"competencia_{i}"] for i in range(1, 6)] + [sds["total"]]

    fig, ax = plt.subplots(figsize=(11, 5))
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.5,
                    flierprops={"marker": "o", "markersize": 3, "alpha": 0.4})
    for patch, color in zip(bp["boxes"], PALETTE[:6]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Desvio padrão entre avaliadores")
    ax.set_title(
        f"Variabilidade interavaliador por competência\n"
        f"(n={len(multi)} redações com 2+ avaliadores)"
    )
    fig.tight_layout()
    out = FIG_DIR / "06_variabilidade_interavaliador.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[ok] {out}")


# ── estatísticas textuais ─────────────────────────────────────────────────────

def print_summary(df_ann: pd.DataFrame, df_ess: pd.DataFrame, essays: list[dict]) -> None:
    print("\n─── Resumo do corpus ──────────────────────────────────────────")
    print(f"  Redações únicas:              {len(essays)}")
    print(f"  Instâncias (redação×avaliador): {len(df_ann)}")
    n_multi = sum(1 for e in essays if len(e["annotations"]) >= 2)
    print(f"  Redações com 2+ avaliadores:  {n_multi}")

    print("\n─── Nota total (por instância) ───────────────────────────────")
    desc = df_ann["total"].describe(percentiles=[0.25, 0.5, 0.75])
    for stat in ["min", "25%", "50%", "mean", "75%", "max", "std"]:
        print(f"  {stat:>5s}: {desc[stat]:.1f}")

    print("\n─── Notas por competência (mediana) ──────────────────────────")
    for key, label in COMPETENCY_LABELS.items():
        med = df_ann[key].median()
        mean = df_ann[key].mean()
        print(f"  {key}: mediana={med:.0f}  média={mean:.1f}")

    print("\n─── Tamanho dos textos (redações únicas) ─────────────────────")
    for col, label in [("n_chars", "caracteres"), ("n_words", "palavras")]:
        print(
            f"  {label}: min={df_ess[col].min()}  "
            f"mediana={df_ess[col].median():.0f}  "
            f"média={df_ess[col].mean():.0f}  "
            f"max={df_ess[col].max()}"
        )

    if "essay_year" in df_ess.columns and df_ess["essay_year"].notna().any():
        print("\n─── Redações por ano ─────────────────────────────────────────")
        for year, cnt in df_ess["essay_year"].value_counts().sort_index().items():
            print(f"  {int(year)}: {cnt} redações")
    print()


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Análise exploratória do corpus pré-processado.")
    parser.add_argument("--data", default="data/processed_essays.jsonl")
    args = parser.parse_args()

    print(f"Carregando {args.data}...")
    essays = load_essays(args.data)
    df_ann = build_annotations_df(essays)
    df_ess = build_essays_df(essays)

    print_summary(df_ann, df_ess, essays)

    print("Gerando gráficos em data/figures/...")
    plot_competency_boxplots(df_ann)
    plot_total_histogram(df_ann)
    plot_score_bands(df_ann)
    plot_text_length(df_ess)
    plot_score_vs_length(df_ess)
    plot_interrater_variability(df_ess, essays)

    print("\nPronto. Arquivos em data/figures/")
