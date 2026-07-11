import json
from pathlib import Path
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr
from sklearn.metrics import roc_curve, auc


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

TOKEN_SET = ["H", "I", "Y", "V", "P", "D", "U"]
RAW_PREFERRED = {"beta_propeller", "tim_barrel", "rossmann_fold", "outer_membrane_barrel"}
SUBMAT = np.array([
    [3, 0, -1, -1, 0, 0, -1],
    [0, 3, -1, -1, 0, 0, -1],
    [-1, -1, 1, 0, -1, -1, -1],
    [-1, -1, 0, 0.5, -1, -1, -1],
    [0, 0, -1, -1, 3, 0, -1],
    [0, 0, -1, -1, 0, 3, -1],
    [-1, -1, -1, -1, -1, -1, 0],
], dtype=float)


def min_max_scale(series):
    series = pd.to_numeric(series, errors="coerce")
    lo, hi = series.min(), series.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - lo) / (hi - lo + 1e-9)


def rank_scale(series):
    series = pd.to_numeric(series, errors="coerce")
    return series.rank(method="average", pct=True)


def safe_log1p_scale(series):
    series = pd.to_numeric(series, errors="coerce")
    shifted = series - series.min()
    return min_max_scale(np.log1p(shifted))


def extract_primary_tokens(tokens_json):
    if pd.isna(tokens_json) or not str(tokens_json).strip():
        return []
    data = json.loads(tokens_json)
    return [(tok.get("interactions", []) or ["U"])[0] for tok in data]


def token_distribution(primary_tokens):
    if not primary_tokens:
        return np.ones(len(TOKEN_SET), dtype=float) / len(TOKEN_SET)
    counts = {token: 0 for token in TOKEN_SET}
    for token in primary_tokens:
        counts[token if token in counts else "U"] += 1
    vec = np.array([counts[token] for token in TOKEN_SET], dtype=float)
    return vec / vec.sum()


def shannon_entropy(primary_tokens):
    p = token_distribution(primary_tokens)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def js_similarity(p, q):
    return 1.0 - float(jensenshannon(p, q))


def build_large_blend():
    results = pd.read_csv(ROOT / "20000" / "itsa_cpp.csv")
    tokens = pd.read_csv(ROOT / "20000" / "protein_tokens.csv")

    results = results[results["status"].astype(str).str.lower() == "ok"].copy()
    results["pair_label"] = pd.to_numeric(results["pair_label"], errors="coerce")
    results["sw_norm"] = pd.to_numeric(results["sw_norm"], errors="coerce")
    results["sw_raw_score"] = pd.to_numeric(results["sw_raw_score"], errors="coerce")
    results = results.dropna(subset=["pair_label", "sw_norm", "sw_raw_score"]).copy()

    tokens = tokens[tokens["status"].astype(str).str.lower() == "ok"].copy()
    dist_lookup = {}
    for _, row in tokens.iterrows():
        key = (str(row["family"]).strip(), str(row["pdb_id"]).strip())
        dist_lookup[key] = token_distribution(extract_primary_tokens(row["tokens_json"]))

    js_sims = []
    for _, row in results.iterrows():
        k1 = (str(row["family_1"]).strip(), str(row["pdb_id_1"]).strip())
        k2 = (str(row["family_2"]).strip(), str(row["pdb_id_2"]).strip())
        js_sims.append(js_similarity(dist_lookup[k1], dist_lookup[k2]) if k1 in dist_lookup and k2 in dist_lookup else np.nan)
    results["js_sim"] = js_sims
    results = results.dropna(subset=["js_sim"]).copy()

    results["sw_norm_mm"] = min_max_scale(results["sw_norm"])
    results["sw_raw_mm"] = min_max_scale(results["sw_raw_score"])
    results["sw_raw_log_mm"] = safe_log1p_scale(results["sw_raw_score"])
    results["js_rank"] = rank_scale(results["js_sim"])
    raw_mask = results["family_1"].isin(RAW_PREFERRED) | results["family_2"].isin(RAW_PREFERRED)
    results["current_hybrid_mm"] = np.where(raw_mask, results["sw_raw_mm"], results["sw_norm_mm"])
    results["best_score"] = (
        0.55 * results["current_hybrid_mm"] +
        0.05 * results["js_rank"] +
        0.40 * results["sw_raw_log_mm"]
    )
    return results


def build_entropy_table():
    results = pd.read_csv(ROOT / "itsa results" / "itsa_cpp.csv")
    tokens = pd.read_csv(ROOT / "itsa results" / "protein_tokens.csv")

    results = results[results["status"].astype(str).str.lower() == "ok"].copy()
    results["pair_label"] = pd.to_numeric(results["pair_label"], errors="coerce")
    results["sw_norm"] = pd.to_numeric(results["sw_norm"], errors="coerce")
    results["sw_raw_score"] = pd.to_numeric(results["sw_raw_score"], errors="coerce")
    results = results.dropna(subset=["pair_label", "sw_norm", "sw_raw_score"]).copy()

    tokens = tokens[tokens["status"].astype(str).str.lower() == "ok"].copy()
    tokens["primary_tokens"] = tokens["tokens_json"].apply(extract_primary_tokens)
    tokens["entropy"] = tokens["primary_tokens"].apply(shannon_entropy)
    family_entropy = (
        tokens.groupby("family", as_index=False)
        .agg(mean_entropy=("entropy", "mean"), median_entropy=("entropy", "median"), n_proteins=("pdb_id", "count"))
    )

    dist_lookup = {}
    for _, row in tokens.iterrows():
        dist_lookup[(str(row["family"]).strip(), str(row["pdb_id"]).strip())] = token_distribution(row["primary_tokens"])

    js_sims = []
    for _, row in results.iterrows():
        k1 = (str(row["family_1"]).strip(), str(row["pdb_id_1"]).strip())
        k2 = (str(row["family_2"]).strip(), str(row["pdb_id_2"]).strip())
        js_sims.append(js_similarity(dist_lookup[k1], dist_lookup[k2]) if k1 in dist_lookup and k2 in dist_lookup else np.nan)
    results["js_sim"] = js_sims
    results = results.dropna(subset=["js_sim"]).copy()

    results["sw_norm_mm"] = min_max_scale(results["sw_norm"])
    results["sw_raw_mm"] = min_max_scale(results["sw_raw_score"])
    results["sw_raw_log_mm"] = safe_log1p_scale(results["sw_raw_score"])
    results["js_rank"] = rank_scale(results["js_sim"])
    raw_mask = results["family_1"].isin(RAW_PREFERRED) | results["family_2"].isin(RAW_PREFERRED)
    results["current_hybrid_mm"] = np.where(raw_mask, results["sw_raw_mm"], results["sw_norm_mm"])
    results["best_score"] = 0.55 * results["current_hybrid_mm"] + 0.05 * results["js_rank"] + 0.40 * results["sw_raw_log_mm"]

    family_rows = []
    for family in sorted(set(results["family_1"]).union(results["family_2"])):
        subset = results[(results["family_1"] == family) | (results["family_2"] == family)].copy()
        y = subset["pair_label"].astype(int)
        if len(subset) < 50 or y.nunique() < 2:
            continue
        from sklearn.metrics import roc_auc_score, average_precision_score
        family_rows.append({
            "family": family,
            "auc": roc_auc_score(y, subset["best_score"]),
            "auprc": average_precision_score(y, subset["best_score"]),
        })
    family_perf = pd.DataFrame(family_rows)
    return family_perf.merge(family_entropy, on="family", how="inner")


def save_substitution_heatmap():
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(SUBMAT, cmap="coolwarm", vmin=-1, vmax=3)
    ax.set_xticks(range(len(TOKEN_SET)))
    ax.set_yticks(range(len(TOKEN_SET)))
    ax.set_xticklabels(TOKEN_SET)
    ax.set_yticklabels(TOKEN_SET)
    ax.set_title("ITSA substitution matrix")
    for i in range(len(TOKEN_SET)):
        for j in range(len(TOKEN_SET)):
            ax.text(j, i, f"{SUBMAT[i,j]:g}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "substitution_matrix_heatmap.png", dpi=250)
    plt.close(fig)


def save_curated_roc():
    itsa = pd.read_csv(ROOT / "itsa results" / "itsa_results_with_js_hybrid.csv")
    itsa = itsa[itsa["status"].astype(str).str.lower() == "ok"].copy()
    y1 = itsa["pair_label"].astype(int)
    s1 = pd.to_numeric(itsa["best_hybrid_score"], errors="coerce")
    fpr1, tpr1, _ = roc_curve(y1, s1)
    auc1 = auc(fpr1, tpr1)

    tm = pd.concat([pd.read_csv(path) for path in glob.glob(str(ROOT / "TM-Align Things" / "tmalign_*.csv"))], ignore_index=True)
    tm = tm[tm["status"].astype(str).str.lower() == "ok"].copy()
    y2 = tm["pair_label"].astype(int)
    s2 = pd.to_numeric(tm["tm_score_avg"], errors="coerce")
    fpr2, tpr2, _ = roc_curve(y2, s2)
    auc2 = auc(fpr2, tpr2)

    fig, ax = plt.subplots(figsize=(6.2, 5.8))
    ax.plot(fpr1, tpr1, label=f"ITSA (AUC = {auc1:.3f})", linewidth=2.2, color="#1f77b4")
    ax.plot(fpr2, tpr2, label=f"TM-align (AUC = {auc2:.3f})", linewidth=2.2, color="#d95f02")
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Curated benchmark ROC")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "roc_curated_itsa_vs_tmalign.png", dpi=250)
    plt.close(fig)


def save_large_roc_and_distribution():
    df = build_large_blend()
    y = df["pair_label"].astype(int)
    s = df["best_score"]
    fpr, tpr, _ = roc_curve(y, s)
    auc_val = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6.2, 5.8))
    ax.plot(fpr, tpr, linewidth=2.2, color="#1f77b4", label=f"ITSA large benchmark (AUC = {auc_val:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Large benchmark ROC")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "roc_large_itsa.png", dpi=250)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    bins = np.linspace(float(s.min()), float(s.max()), 50)
    ax.hist(df.loc[y == 0, "best_score"], bins=bins, density=True, alpha=0.65, color="#d95f02", label="negative")
    ax.hist(df.loc[y == 1, "best_score"], bins=bins, density=True, alpha=0.65, color="#1b9e77", label="positive")
    ax.set_xlabel("Blended ITSA score")
    ax.set_ylabel("Density")
    ax.set_title("Large benchmark score distribution")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "score_distribution_large.png", dpi=250)
    plt.close(fig)


def save_entropy_scatter():
    merged = build_entropy_table()
    r_auc, _ = pearsonr(merged["mean_entropy"], merged["auc"])
    r_auprc, _ = pearsonr(merged["mean_entropy"], merged["auprc"])

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
    for ax, metric, title, rval in [
        (axes[0], "auc", "Entropy vs AUC", r_auc),
        (axes[1], "auprc", "Entropy vs AUPRC", r_auprc),
    ]:
        ax.scatter(merged["mean_entropy"], merged[metric], s=55, color="#1f77b4")
        for _, row in merged.iterrows():
            ax.text(row["mean_entropy"] + 0.003, row[metric] + 0.003, row["family"], fontsize=7)
        ax.set_xlabel("Mean token entropy")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{title} (Pearson r = {rval:.3f})")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "entropy_vs_performance.png", dpi=250)
    plt.close(fig)


def save_tmalign_vs_itsa_scatter():
    itsa = pd.read_csv(ROOT / "itsa results" / "itsa_results_with_js_hybrid.csv")
    itsa = itsa[itsa["status"].astype(str).str.lower() == "ok"].copy()
    itsa["pair_id"] = itsa["pdb_id_1"].astype(str) + "__" + itsa["pdb_id_2"].astype(str)
    itsa["itsa_score"] = pd.to_numeric(itsa["best_hybrid_score"], errors="coerce")
    itsa = itsa[["pair_id", "pair_label", "itsa_score", "family_1", "family_2"]]

    tm = pd.concat([pd.read_csv(path) for path in glob.glob(str(ROOT / "TM-Align Things" / "tmalign_*.csv"))], ignore_index=True)
    tm = tm[tm["status"].astype(str).str.lower() == "ok"].copy()
    tm["pair_id"] = tm["pdb_id_1"].astype(str) + "__" + tm["pdb_id_2"].astype(str)
    tm["tm_score_avg"] = pd.to_numeric(tm["tm_score_avg"], errors="coerce")
    tm = tm[["pair_id", "tm_score_avg"]]

    merged = itsa.merge(tm, on="pair_id", how="inner").dropna()
    colors = merged["pair_label"].map({0: "#d95f02", 1: "#1b9e77"})
    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    ax.scatter(merged["tm_score_avg"], merged["itsa_score"], c=colors, s=14, alpha=0.5, edgecolors="none")
    ax.set_xlabel("TM-align average TM-score")
    ax.set_ylabel("ITSA blended score")
    ax.set_title("TM-align vs ITSA on curated benchmark")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "tmalign_vs_itsa_scatter.png", dpi=250)
    plt.close(fig)


def main():
    save_substitution_heatmap()
    save_curated_roc()
    save_large_roc_and_distribution()
    save_entropy_scatter()
    save_tmalign_vs_itsa_scatter()
    print("Saved figures to", FIG_DIR)


if __name__ == "__main__":
    main()
