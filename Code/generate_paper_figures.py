"""Generate the paper figures for the frozen six-term ITSA formula.

The input score files are produced once by the frozen family-independent
formula. This script only evaluates and plots those scores; it does not fit or
select the model, and it never uses family identity to calculate a score.
"""

import glob
import json
import os
import tempfile
from pathlib import Path

MATPLOTLIB_CACHE = Path(tempfile.gettempdir()) / "itsa_matplotlib_cache"
MATPLOTLIB_CACHE.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "Results" / "figures"
RESULTS_DIR = ROOT / "Results"
FIG_DIR.mkdir(exist_ok=True)

SMALL_SCORED = RESULTS_DIR / "small_benchmark_six_term_scores.csv"
LARGE_SCORED = RESULTS_DIR / "large_benchmark_six_term_scores.csv"
FORMULA_SPEC = RESULTS_DIR / "frozen_interpretable_formula_v1.json"
SCORE_COL = "interpretable_family_independent_score"

TOKEN_SET = ["H", "I", "Y", "V", "P", "D", "U"]
REPEAT_RICH = [
    "beta_propeller",
    "tim_barrel",
    "rossmann_fold",
    "outer_membrane_barrel",
]
SUBMAT = np.array(
    [
        [3, 0, -1, -1, 0, 0, -1],
        [0, 3, -1, -1, 0, 0, -1],
        [-1, -1, 1, 0, -1, -1, -1],
        [-1, -1, 0, 0.5, -1, -1, -1],
        [0, 0, -1, -1, 3, 0, -1],
        [0, 0, -1, -1, 0, 3, -1],
        [-1, -1, -1, -1, -1, -1, 0],
    ],
    dtype=float,
)


def require_inputs():
    missing = [path for path in (SMALL_SCORED, LARGE_SCORED, FORMULA_SPEC) if not path.exists()]
    if missing:
        names = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing frozen six-term input files:\n{names}")
    spec = json.loads(FORMULA_SPEC.read_text())
    if spec.get("term_count") != 6 or spec.get("family_identity_used_as_predictor") is not False:
        raise ValueError("Formula specification is not the expected six-term family-independent model")
    return spec


def load_scores(path):
    df = pd.read_csv(path)
    required = {"pdb_id_1", "family_1", "pdb_id_2", "family_2", "pair_label", SCORE_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    df["pair_label"] = pd.to_numeric(df["pair_label"], errors="raise").astype(int)
    df[SCORE_COL] = pd.to_numeric(df[SCORE_COL], errors="raise")
    return df


def unordered_pair_id(df, left="pdb_id_1", right="pdb_id_2"):
    a = df[left].astype(str).str.upper()
    b = df[right].astype(str).str.upper()
    return np.where(a <= b, a + "__" + b, b + "__" + a)


def family_metrics(df):
    rows = []
    families = sorted(set(df["family_1"]).union(df["family_2"]))
    for family in families:
        subset = df[(df["family_1"] == family) | (df["family_2"] == family)]
        y = subset["pair_label"]
        if y.nunique() < 2:
            continue
        rows.append(
            {
                "family": family,
                "auc": roc_auc_score(y, subset[SCORE_COL]),
                "auprc": average_precision_score(y, subset[SCORE_COL]),
                "positive_pairs": int(y.sum()),
                "negative_pairs": int((1 - y).sum()),
            }
        )
    return pd.DataFrame(rows)


def extract_primary_tokens(tokens_json):
    if pd.isna(tokens_json) or not str(tokens_json).strip():
        return []
    data = json.loads(tokens_json)
    return [(token.get("interactions", []) or ["U"])[0] for token in data]


def token_distribution(primary_tokens):
    if not primary_tokens:
        return np.ones(len(TOKEN_SET), dtype=float) / len(TOKEN_SET)
    counts = {token: 0 for token in TOKEN_SET}
    for token in primary_tokens:
        counts[token if token in counts else "U"] += 1
    values = np.array([counts[token] for token in TOKEN_SET], dtype=float)
    return values / values.sum()


def shannon_entropy(primary_tokens):
    probabilities = token_distribution(primary_tokens)
    probabilities = probabilities[probabilities > 0]
    return float(-(probabilities * np.log(probabilities)).sum())


def build_entropy_table(small):
    tokens = pd.read_csv(ROOT / "Datasets" / "curated_benchmark" / "protein_tokens.csv")
    tokens = tokens[tokens["status"].astype(str).str.lower().eq("ok")].copy()
    tokens["primary_tokens"] = tokens["tokens_json"].apply(extract_primary_tokens)
    tokens["entropy"] = tokens["primary_tokens"].apply(shannon_entropy)
    family_entropy = (
        tokens.groupby("family", as_index=False)
        .agg(
            mean_entropy=("entropy", "mean"),
            median_entropy=("entropy", "median"),
            n_proteins=("pdb_id", "count"),
        )
    )
    return family_metrics(small).merge(family_entropy, on="family", how="inner")


def load_tmalign():
    paths = glob.glob(str(ROOT / "TM-Align" / "tmalign_*.csv"))
    if not paths:
        raise FileNotFoundError("No TM-align family CSV files were found")
    tm = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    tm = tm[tm["status"].astype(str).str.lower().eq("ok")].copy()
    tm["pair_id"] = unordered_pair_id(tm)
    tm["tm_score_avg"] = pd.to_numeric(tm["tm_score_avg"], errors="coerce")
    tm = tm.dropna(subset=["tm_score_avg"])
    return tm.sort_values("tm_score_avg").drop_duplicates("pair_id", keep="last")


def save_substitution_heatmap():
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    image = ax.imshow(SUBMAT, cmap="coolwarm", vmin=-1, vmax=3)
    ax.set_xticks(range(len(TOKEN_SET)), TOKEN_SET)
    ax.set_yticks(range(len(TOKEN_SET)), TOKEN_SET)
    ax.set_title("ITSA substitution matrix")
    for row in range(len(TOKEN_SET)):
        for col in range(len(TOKEN_SET)):
            ax.text(col, row, f"{SUBMAT[row, col]:g}", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "substitution_matrix_heatmap.png", dpi=250)
    plt.close(fig)


def save_curated_roc(small, tm):
    y_itsa = small["pair_label"]
    fpr_itsa, tpr_itsa, _ = roc_curve(y_itsa, small[SCORE_COL])
    auc_itsa = roc_auc_score(y_itsa, small[SCORE_COL])

    y_tm = tm["pair_label"].astype(int)
    fpr_tm, tpr_tm, _ = roc_curve(y_tm, tm["tm_score_avg"])
    auc_tm = roc_auc_score(y_tm, tm["tm_score_avg"])

    fig, ax = plt.subplots(figsize=(6.2, 5.8))
    ax.plot(fpr_itsa, tpr_itsa, label=f"Six-term ITSA (AUC = {auc_itsa:.3f})", linewidth=2.2, color="#1f77b4")
    ax.plot(fpr_tm, tpr_tm, label=f"TM-align (AUC = {auc_tm:.3f})", linewidth=2.2, color="#d95f02")
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Curated benchmark ROC")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "roc_curated_itsa_vs_tmalign.png", dpi=250)
    plt.close(fig)


def save_large_roc_and_distribution(large):
    y = large["pair_label"]
    score = large[SCORE_COL]
    fpr, tpr, _ = roc_curve(y, score)
    auc_value = roc_auc_score(y, score)

    fig, ax = plt.subplots(figsize=(6.2, 5.8))
    ax.plot(fpr, tpr, linewidth=2.2, color="#1f77b4", label=f"Six-term ITSA (AUC = {auc_value:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Large benchmark ROC")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "roc_large_itsa.png", dpi=250)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    bins = np.linspace(float(score.min()), float(score.max()), 50)
    ax.hist(score[y == 0], bins=bins, density=True, alpha=0.65, color="#d95f02", label="Negative")
    ax.hist(score[y == 1], bins=bins, density=True, alpha=0.65, color="#1b9e77", label="Positive")
    ax.set(xlabel="Frozen six-term ITSA score", ylabel="Density", title="Large benchmark score distribution")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "score_distribution_large.png", dpi=250)
    plt.close(fig)


def save_entropy_scatter(entropy_table):
    r_auc, p_auc = pearsonr(entropy_table["mean_entropy"], entropy_table["auc"])
    r_auprc, p_auprc = pearsonr(entropy_table["mean_entropy"], entropy_table["auprc"])

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
    for ax, metric, title, r_value, p_value in [
        (axes[0], "auc", "Entropy vs AUC", r_auc, p_auc),
        (axes[1], "auprc", "Entropy vs AUPRC", r_auprc, p_auprc),
    ]:
        ax.scatter(entropy_table["mean_entropy"], entropy_table[metric], s=55, color="#1f77b4")
        for _, row in entropy_table.iterrows():
            ax.annotate(row["family"], (row["mean_entropy"], row[metric]), xytext=(3, 3), textcoords="offset points", fontsize=7)
        ax.set_xlabel("Mean token entropy")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{title} (Pearson r = {r_value:.3f}, p = {p_value:.3f})")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "entropy_vs_performance.png", dpi=250)
    plt.close(fig)


def save_tmalign_vs_itsa_scatter(small, tm):
    itsa = small.copy()
    itsa["pair_id"] = unordered_pair_id(itsa)
    merged = itsa.merge(tm[["pair_id", "tm_score_avg"]], on="pair_id", how="inner").dropna()
    colors = merged["pair_label"].map({0: "#d95f02", 1: "#1b9e77"})

    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    ax.scatter(merged["tm_score_avg"], merged[SCORE_COL], c=colors, s=14, alpha=0.5, edgecolors="none")
    ax.set(
        xlabel="TM-align average TM-score",
        ylabel="Frozen six-term ITSA score",
        title="TM-align vs family-independent ITSA",
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "tmalign_vs_itsa_scatter.png", dpi=250)
    plt.close(fig)


def save_repeat_rich_comparison(new_family_metrics):
    comparison = pd.DataFrame(
        {
            "family": REPEAT_RICH,
            "Original family-aware ITSA": [0.7330, 0.7222, 0.7595, 0.8298],
            "Six-term family-independent ITSA": [
                float(new_family_metrics.set_index("family").loc[family, "auc"])
                for family in REPEAT_RICH
            ],
            "TM-align": [0.6355, 0.7614, 0.8250, 0.9934],
            "Foldseek": [0.4930, 0.6222, 0.6475, 0.9997],
        }
    )
    comparison.to_csv(RESULTS_DIR / "six_term_repeat_rich_comparison.csv", index=False)

    x = np.arange(len(comparison))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    palette = ["#94a3b8", "#0f766e", "#d95f02", "#6d28d9"]
    columns = [column for column in comparison.columns if column != "family"]
    for index, (column, color) in enumerate(zip(columns, palette)):
        ax.bar(x + (index - 1.5) * width, comparison[column], width, label=column, color=color)
    ax.set_xticks(x, [family.replace("_", " ") for family in comparison["family"]])
    ax.set_ylim(0.45, 1.02)
    ax.set_ylabel("Family-level AUC")
    ax.set_title("Performance in repeat-rich structural families")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "repeat_rich_family_auc_comparison.png", dpi=250)
    plt.close(fig)


def print_summary(small, large, family_table, entropy_table):
    for name, data in [("Small fitted", small), ("Large frozen", large)]:
        y = data["pair_label"]
        auc_value = roc_auc_score(y, data[SCORE_COL])
        auprc = average_precision_score(y, data[SCORE_COL])
        print(f"{name}: AUC={auc_value:.6f}, AUPRC={auprc:.6f}, prevalence={y.mean():.6f}")
    print("\nFamily metrics:\n", family_table.sort_values("auc", ascending=False).to_string(index=False))
    for metric in ("auc", "auprc"):
        pearson = pearsonr(entropy_table["mean_entropy"], entropy_table[metric])
        spearman = spearmanr(entropy_table["mean_entropy"], entropy_table[metric])
        print(
            f"Entropy vs {metric.upper()}: "
            f"Pearson r={pearson.statistic:.6f}, p={pearson.pvalue:.6f}; "
            f"Spearman rho={spearman.statistic:.6f}, p={spearman.pvalue:.6f}"
        )


def main():
    spec = require_inputs()
    small = load_scores(SMALL_SCORED)
    large = load_scores(LARGE_SCORED)
    tm = load_tmalign()
    family_table = family_metrics(small)
    entropy_table = build_entropy_table(small)
    family_table.to_csv(RESULTS_DIR / "six_term_family_metrics.csv", index=False)
    entropy_table.to_csv(RESULTS_DIR / "six_term_entropy_family_metrics.csv", index=False)

    save_substitution_heatmap()
    save_curated_roc(small, tm)
    save_large_roc_and_distribution(large)
    save_entropy_scatter(entropy_table)
    save_tmalign_vs_itsa_scatter(small, tm)
    save_repeat_rich_comparison(family_table)
    print_summary(small, large, family_table, entropy_table)
    print(f"\nGenerated figures from {spec['method_name']} in {FIG_DIR}")


if __name__ == "__main__":
    main()
