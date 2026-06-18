import json
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import roc_auc_score, average_precision_score

# =========================
# FILES
# =========================
RESULTS_FILE = "20000/itsa_cpp.csv"
TOKENS_FILE = "20000/protein_tokens.csv"

# =========================
# HELPERS
# =========================
TOKEN_SET = ["H", "I", "Y", "V", "P", "D", "U"]


def min_max_scale(s):
    s = pd.to_numeric(s, errors="coerce")
    lo, hi = s.min(), s.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo + 1e-9)


def zscore_sigmoid(s):
    s = pd.to_numeric(s, errors="coerce")
    mu = s.mean()
    sd = s.std()
    if pd.isna(mu) or pd.isna(sd) or sd == 0:
        return pd.Series(np.full(len(s), 0.5), index=s.index)
    z = (s - mu) / (sd + 1e-9)
    return 1.0 / (1.0 + np.exp(-z))


def rank_scale(s):
    s = pd.to_numeric(s, errors="coerce")
    return s.rank(method="average", pct=True)


def safe_log1p_scale(s):
    s = pd.to_numeric(s, errors="coerce")
    s_shift = s - s.min()
    logged = np.log1p(s_shift)
    return min_max_scale(logged)


def extract_primary_tokens(tokens_json):
    if pd.isna(tokens_json) or not str(tokens_json).strip():
        return []
    try:
        data = json.loads(tokens_json)
        primaries = []
        for tok in data:
            interactions = tok.get("interactions", [])
            primaries.append(interactions[0] if interactions else "U")
        return primaries
    except Exception:
        return []


def token_distribution(primary_tokens):
    if not primary_tokens:
        return np.ones(len(TOKEN_SET), dtype=float) / len(TOKEN_SET)

    counts = {t: 0 for t in TOKEN_SET}
    for t in primary_tokens:
        if t not in counts:
            counts["U"] += 1
        else:
            counts[t] += 1

    vec = np.array([counts[t] for t in TOKEN_SET], dtype=float)
    return vec / vec.sum()


def js_similarity(p, q):
    return 1.0 - float(jensenshannon(p, q))


def compute_metrics(y_true, y_score):
    return roc_auc_score(y_true, y_score), average_precision_score(y_true, y_score)


def precision_at_k(y_true, y_score, k_frac=0.10):
    df = pd.DataFrame({"y_true": y_true, "y_score": y_score}).sort_values("y_score", ascending=False)
    k = max(1, int(len(df) * k_frac))
    top_k = df.head(k)
    return top_k["y_true"].sum() / len(top_k)


def evaluate_score(name, y_true, score, rows):
    auc_val, auprc_val = compute_metrics(y_true, score)
    p5 = precision_at_k(y_true, score, k_frac=0.05)
    p10 = precision_at_k(y_true, score, k_frac=0.10)
    rows.append({
        "method": name,
        "auc": auc_val,
        "auprc": auprc_val,
        "p_at_5": p5,
        "p_at_10": p10
    })


# =========================
# LOAD FILES
# =========================
results = pd.read_csv(RESULTS_FILE)
tokens = pd.read_csv(TOKENS_FILE)

results = results[results["status"].astype(str).str.lower() == "ok"].copy()
results["pair_label"] = pd.to_numeric(results["pair_label"], errors="coerce")
results["sw_norm"] = pd.to_numeric(results["sw_norm"], errors="coerce")
results["sw_raw_score"] = pd.to_numeric(results["sw_raw_score"], errors="coerce")
results = results.dropna(subset=["pair_label", "sw_norm", "sw_raw_score"]).copy()

tokens = tokens[tokens["status"].astype(str).str.lower() == "ok"].copy()

# =========================
# BUILD TOKEN DISTRIBUTION LOOKUP
# =========================
dist_lookup = {}
for _, row in tokens.iterrows():
    key = (str(row["family"]).strip(), str(row["pdb_id"]).strip())
    primaries = extract_primary_tokens(row["tokens_json"])
    dist_lookup[key] = token_distribution(primaries)

# =========================
# ADD JS TOKEN SIMILARITY
# =========================
js_sims = []
for _, row in results.iterrows():
    k1 = (str(row["family_1"]).strip(), str(row["pdb_id_1"]).strip())
    k2 = (str(row["family_2"]).strip(), str(row["pdb_id_2"]).strip())
    if k1 in dist_lookup and k2 in dist_lookup:
        js_sims.append(js_similarity(dist_lookup[k1], dist_lookup[k2]))
    else:
        js_sims.append(np.nan)

results["js_sim"] = js_sims
results = results.dropna(subset=["js_sim"]).copy()

print(f"Usable pairs after JS merge: {len(results)}")

y_true = results["pair_label"].astype(int)

# =========================
# MULTIPLE CALIBRATIONS
# =========================
results["sw_norm_mm"] = min_max_scale(results["sw_norm"])
results["sw_norm_rank"] = rank_scale(results["sw_norm"])
results["sw_norm_sig"] = zscore_sigmoid(results["sw_norm"])

results["sw_raw_mm"] = min_max_scale(results["sw_raw_score"])
results["sw_raw_rank"] = rank_scale(results["sw_raw_score"])
results["sw_raw_sig"] = zscore_sigmoid(results["sw_raw_score"])
results["sw_raw_log_mm"] = safe_log1p_scale(results["sw_raw_score"])

results["js_mm"] = min_max_scale(results["js_sim"])
results["js_rank"] = rank_scale(results["js_sim"])
results["js_sig"] = zscore_sigmoid(results["js_sim"])

# current family-based hybrid
raw_preferred = {"beta_propeller", "tim_barrel", "rossmann_fold", "outer_membrane_barrel"}
raw_mask = results["family_1"].isin(raw_preferred) | results["family_2"].isin(raw_preferred)

results["current_hybrid_mm"] = np.where(raw_mask, results["sw_raw_mm"], results["sw_norm_mm"])
results["current_hybrid_rank"] = np.where(raw_mask, results["sw_raw_rank"], results["sw_norm_rank"])
results["current_hybrid_sig"] = np.where(raw_mask, results["sw_raw_sig"], results["sw_norm_sig"])
results["current_hybrid_log"] = np.where(raw_mask, results["sw_raw_log_mm"], results["sw_norm_mm"])

# =========================
# BASELINE EVALUATION
# =========================
summary_rows = []

base_cols = [
    "sw_norm_mm", "sw_norm_rank", "sw_norm_sig",
    "sw_raw_mm", "sw_raw_rank", "sw_raw_sig", "sw_raw_log_mm",
    "js_mm", "js_rank", "js_sig",
    "current_hybrid_mm", "current_hybrid_rank", "current_hybrid_sig", "current_hybrid_log"
]

for col in base_cols:
    evaluate_score(col, y_true, results[col], summary_rows)

# =========================
# BLEND SEARCH
# =========================
candidate_cols = {
    "sw_norm_mm": results["sw_norm_mm"],
    "sw_norm_rank": results["sw_norm_rank"],
    "sw_raw_mm": results["sw_raw_mm"],
    "sw_raw_rank": results["sw_raw_rank"],
    "sw_raw_log_mm": results["sw_raw_log_mm"],
    "js_mm": results["js_mm"],
    "js_rank": results["js_rank"],
    "current_hybrid_mm": results["current_hybrid_mm"],
    "current_hybrid_rank": results["current_hybrid_rank"],
    "current_hybrid_log": results["current_hybrid_log"],
}

best_auc = None
best_auprc = None

weights = np.arange(0.0, 1.01, 0.02)

# 2-way blends
for n1, s1 in candidate_cols.items():
    for n2, s2 in candidate_cols.items():
        if n1 >= n2:
            continue
        for w in weights:
            score = (1.0 - w) * s1 + w * s2
            name = f"{n1} * {1.0-w:.2f} + {n2} * {w:.2f}"
            auc_val, auprc_val = compute_metrics(y_true, score)
            p5 = precision_at_k(y_true, score, k_frac=0.05)
            p10 = precision_at_k(y_true, score, k_frac=0.10)

            row = {
                "method": name,
                "auc": auc_val,
                "auprc": auprc_val,
                "p_at_5": p5,
                "p_at_10": p10
            }
            summary_rows.append(row)

            if best_auc is None or (auc_val > best_auc["auc"]) or (
                np.isclose(auc_val, best_auc["auc"]) and auprc_val > best_auc["auprc"]
            ):
                best_auc = row.copy()

            if best_auprc is None or (auprc_val > best_auprc["auprc"]) or (
                np.isclose(auprc_val, best_auprc["auprc"]) and auc_val > best_auprc["auc"]
            ):
                best_auprc = row.copy()

# 3-way blends centered on current hybrid
hybrid_variants = ["current_hybrid_mm", "current_hybrid_rank", "current_hybrid_log"]
js_variants = ["js_mm", "js_rank"]
sw_variants = ["sw_norm_mm", "sw_norm_rank", "sw_raw_log_mm"]

for h in hybrid_variants:
    for j in js_variants:
        for s in sw_variants:
            for wh in np.arange(0.4, 0.91, 0.05):
                for wj in np.arange(0.0, 0.41, 0.05):
                    ws = 1.0 - wh - wj
                    if ws < 0:
                        continue
                    score = wh * candidate_cols[h] + wj * candidate_cols[j] + ws * candidate_cols[s]
                    name = f"{h}*{wh:.2f} + {j}*{wj:.2f} + {s}*{ws:.2f}"
                    auc_val, auprc_val = compute_metrics(y_true, score)
                    p5 = precision_at_k(y_true, score, k_frac=0.05)
                    p10 = precision_at_k(y_true, score, k_frac=0.10)

                    row = {
                        "method": name,
                        "auc": auc_val,
                        "auprc": auprc_val,
                        "p_at_5": p5,
                        "p_at_10": p10
                    }
                    summary_rows.append(row)

                    if (auc_val > best_auc["auc"]) or (
                        np.isclose(auc_val, best_auc["auc"]) and auprc_val > best_auc["auprc"]
                    ):
                        best_auc = row.copy()

                    if (auprc_val > best_auprc["auprc"]) or (
                        np.isclose(auprc_val, best_auprc["auprc"]) and auc_val > best_auprc["auc"]
                    ):
                        best_auprc = row.copy()

summary_df = pd.DataFrame(summary_rows)
summary_df = summary_df.sort_values(["auc", "auprc"], ascending=False).reset_index(drop=True)

print("\nTop 20 by AUC:")
print(summary_df.head(20).to_string(index=False))

print("\nTop 20 by AUPRC:")
print(summary_df.sort_values(["auprc", "auc"], ascending=False).head(20).to_string(index=False))

print("\nBEST BY AUC")
print(best_auc)

print("\nBEST BY AUPRC")
print(best_auprc)

# quick reference baselines
prevalence = float(y_true.mean())
print(f"\nPositive prevalence / random AUPRC baseline: {prevalence:.4f}")
