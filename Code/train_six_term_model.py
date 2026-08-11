"""Train, select, freeze, and evaluate the interpretable six-term ITSA model.

This is the self-contained reproduction script for the final family-independent
score. It:

1. constructs intrinsic alignment, JSD, and repeat-sensitive features;
2. compares predefined 6-, 8-, and 10-term candidates;
3. sweeps L2 logistic-regression regularization strength using five
   protein-identity-held-out folds on the small benchmark;
4. selects the fewest-term candidate within 0.005 ROC AUC of the best;
5. refits the selected model on the complete small benchmark; and
6. freezes and applies the model unchanged to the large benchmark.

Family identity is used only to locate token records and balance the validation
folds. It is never supplied to the logistic regression as a predictor. Small
labels fit/select the model; large labels are used only for final evaluation.

Repeat percentiles are cohort-relative and are computed without labels. This
matches the analysis reported in the paper materials: small percentiles use the
small protein cohort and large percentiles use the large protein cohort.

Run from anywhere with:

    python3 Code/train_six_term_model.py

Required packages: numpy, pandas, scikit-learn.
"""

from __future__ import annotations

import hashlib
import json
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "Results"
SMALL_SOURCE = RESULTS_DIR / "small_benchmark_pairwise_results.csv"
LARGE_SOURCE = RESULTS_DIR / "large_benchmark_pairwise_results.csv"
LARGE_RAW_SOURCE = ROOT / "Datasets" / "large_benchmark" / "itsa_cpp.csv"
TOKENS_SOURCE = ROOT / "Datasets" / "large_benchmark" / "protein_tokens.csv"

SMALL_SCORED_OUTPUT = RESULTS_DIR / "small_benchmark_six_term_scores.csv"
LARGE_SCORED_OUTPUT = RESULTS_DIR / "large_benchmark_six_term_scores.csv"
FORMULA_OUTPUT = RESULTS_DIR / "frozen_interpretable_formula_v1.json"
SEARCH_OUTPUT = RESULTS_DIR / "six_term_candidate_cv_search.csv"
STABILITY_OUTPUT = RESULTS_DIR / "six_term_coefficient_stability.csv"
SUMMARY_OUTPUT = RESULTS_DIR / "six_term_training_summary.csv"

EPS = 1e-9
RNG = np.random.default_rng(20260806)
TOKEN_SET = ["H", "I", "Y", "V", "P", "D", "U"]
SS_SET = ["H", "E", "C"]
SCORE_COL = "interpretable_family_independent_score"
REGULARIZATION_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]

CANDIDATES = {
    "six_term": [
        "alignment_norm",
        "alignment_raw_geom",
        "js_all",
        "js_ss",
        "js_joint",
        "repeat_x_joint_js",
    ],
    "eight_term": [
        "alignment_norm",
        "alignment_raw_geom",
        "js_all",
        "js_ss",
        "js_joint",
        "repeat_compress_max",
        "repeat_x_raw_minus_norm",
        "repeat_x_joint_js",
    ],
    "ten_term": [
        "alignment_norm",
        "alignment_raw_geom",
        "length_similarity",
        "js_all",
        "js_ss",
        "js_joint",
        "repeat_compress_max",
        "repeat_x_raw_minus_norm",
        "repeat_x_joint_js",
        "raw_x_primary_js",
    ],
}

DEFINITIONS = {
    "alignment_norm": "ITSA normalized alignment divided by 100",
    "alignment_raw_geom": "Raw alignment divided by the geometric mean of the two self-alignment scores",
    "length_similarity": "Shorter token-sequence length divided by longer length",
    "js_all": "Jensen-Shannon similarity of all interaction-token distributions",
    "js_ss": "Jensen-Shannon similarity of secondary-structure distributions",
    "js_joint": "Jensen-Shannon similarity of joint primary-token/secondary-structure distributions",
    "repeat_compress_max": "Larger of the two percentile-ranked token-sequence compressibility values",
    "repeat_x_raw_minus_norm": "Repeat-consensus rank multiplied by raw/self alignment minus normalized alignment",
    "repeat_x_joint_js": "Repeat-consensus rank multiplied by joint token/structure JSD similarity",
    "raw_x_primary_js": "Raw/self alignment multiplied by primary-token JSD similarity",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(labels, scores):
    return (
        float(roc_auc_score(labels, scores)),
        float(average_precision_score(labels, scores)),
    )


def sequence_metrics(sequence: str):
    sequence = str(sequence)
    length = len(sequence)
    counts = np.array(list(Counter(sequence).values()), dtype=float)
    probabilities = counts / counts.sum()
    kmers = [sequence[index : index + 3] for index in range(max(0, length - 2))]
    if kmers:
        kmer_counts = Counter(kmers)
        repeated_kmer_fraction_3 = float(np.mean([kmer_counts[kmer] > 1 for kmer in kmers]))
    else:
        repeated_kmer_fraction_3 = 0.0
    return {
        "seq_length": length,
        "entropy": float(-(probabilities * np.log(probabilities)).sum()),
        "compressibility": 1.0 - len(zlib.compress(sequence.encode())) / max(length, 1),
        "repeated_kmer_fraction_3": repeated_kmer_fraction_3,
    }


def build_protein_metrics(tokens: pd.DataFrame):
    proteins = tokens[tokens["status"].astype(str).str.lower().eq("ok")].copy()
    proteins = proteins.drop_duplicates(["family", "pdb_id"])
    calculated = pd.DataFrame([sequence_metrics(sequence) for sequence in proteins["primary_seq"]])
    return pd.concat(
        [proteins[["family", "pdb_id"]].reset_index(drop=True), calculated], axis=1
    )


def add_repeat_features(pairs: pd.DataFrame, protein_metrics: pd.DataFrame):
    protein_keys = pd.concat(
        [
            pairs[["family_1", "pdb_id_1"]].rename(
                columns={"family_1": "family", "pdb_id_1": "pdb_id"}
            ),
            pairs[["family_2", "pdb_id_2"]].rename(
                columns={"family_2": "family", "pdb_id_2": "pdb_id"}
            ),
        ]
    ).drop_duplicates()
    proteins = protein_keys.merge(
        protein_metrics, on=["family", "pdb_id"], validate="one_to_one"
    )
    if len(proteins) != len(protein_keys):
        raise RuntimeError("Missing repeat metrics for one or more proteins")

    ranks = {
        "kmer3": proteins["repeated_kmer_fraction_3"].rank(pct=True, method="average"),
        "compress": proteins["compressibility"].rank(pct=True, method="average"),
        "length": proteins["seq_length"].rank(pct=True, method="average"),
        "entropy": proteins["entropy"].rank(pct=True, method="average"),
    }
    ranks["repeat_consensus"] = pd.concat(
        [ranks["kmer3"], ranks["compress"], ranks["length"]], axis=1
    ).mean(axis=1)
    for name, values in ranks.items():
        proteins[f"rp_{name}"] = values.to_numpy()

    rank_columns = [f"rp_{name}" for name in ranks]
    left = proteins[["family", "pdb_id", *rank_columns]].rename(
        columns={
            "family": "family_1",
            "pdb_id": "pdb_id_1",
            **{column: f"{column}_1" for column in rank_columns},
        }
    )
    right = proteins[["family", "pdb_id", *rank_columns]].rename(
        columns={
            "family": "family_2",
            "pdb_id": "pdb_id_2",
            **{column: f"{column}_2" for column in rank_columns},
        }
    )
    return pairs.merge(
        left, on=["family_1", "pdb_id_1"], validate="many_to_one"
    ).merge(right, on=["family_2", "pdb_id_2"], validate="many_to_one")


def normalize_distribution(values):
    values = np.asarray(values, dtype=float)
    total = values.sum()
    return values / total if total > 0 else np.full(len(values), 1.0 / len(values))


def js_divergence_parts(left, right):
    left = normalize_distribution(left)
    right = normalize_distribution(right)
    midpoint = 0.5 * (left + right)
    with np.errstate(divide="ignore", invalid="ignore"):
        left_parts = np.where(left > 0, left * np.log(left / midpoint), 0.0)
        right_parts = np.where(right > 0, right * np.log(right / midpoint), 0.0)
    return 0.5 * (left_parts + right_parts)


def build_token_lookup(tokens: pd.DataFrame):
    lookup = {}
    for row in tokens.itertuples(index=False):
        if str(row.status).lower() != "ok":
            continue
        key = (str(row.family), str(row.pdb_id))
        if key in lookup:
            continue
        token_rows = json.loads(row.tokens_json)
        primary = np.zeros(len(TOKEN_SET))
        all_interactions = np.zeros(len(TOKEN_SET))
        secondary = np.zeros(len(SS_SET))
        joint = np.zeros(len(TOKEN_SET) * len(SS_SET))
        interaction_count = np.zeros(5)
        for token in token_rows:
            interactions = token.get("interactions", []) or ["U"]
            first = interactions[0] if interactions[0] in TOKEN_SET else "U"
            sec_struct = token.get("sec_struct", "C")
            sec_struct = sec_struct if sec_struct in SS_SET else "C"
            token_index = TOKEN_SET.index(first)
            secondary_index = SS_SET.index(sec_struct)
            primary[token_index] += 1
            secondary[secondary_index] += 1
            joint[token_index * len(SS_SET) + secondary_index] += 1
            interaction_count[min(len(interactions), 4)] += 1
            for interaction in interactions:
                index = TOKEN_SET.index(interaction) if interaction in TOKEN_SET else TOKEN_SET.index("U")
                all_interactions[index] += 1
        lookup[key] = {
            "primary": normalize_distribution(primary),
            "all": normalize_distribution(all_interactions),
            "ss": normalize_distribution(secondary),
            "joint": normalize_distribution(joint),
            "nint": normalize_distribution(interaction_count),
        }
    return lookup


def token_js_features(pairs: pd.DataFrame, lookup):
    records = []
    for row in pairs.itertuples(index=False):
        left_key = (str(row.family_1), str(row.pdb_id_1))
        right_key = (str(row.family_2), str(row.pdb_id_2))
        if left_key not in lookup or right_key not in lookup:
            raise RuntimeError(f"Missing token record for {left_key} or {right_key}")
        record = {}
        for name in ["primary", "all", "ss", "joint", "nint"]:
            parts = js_divergence_parts(lookup[left_key][name], lookup[right_key][name])
            record[f"token_{name}_js_similarity"] = 1.0 - float(np.sqrt(parts.sum()))
        records.append(record)
    return pd.DataFrame(records, index=pairs.index)


def build_features(pairs: pd.DataFrame, token_features: pd.DataFrame):
    raw_score = pairs["sw_raw_score"].to_numpy(float)
    normalized = np.clip(pairs["sw_norm"].to_numpy(float) / 100.0, 0.0, 1.5)
    self_1 = pairs["sw_self1_score"].to_numpy(float)
    self_2 = pairs["sw_self2_score"].to_numpy(float)
    length_1 = pairs["len1"].to_numpy(float)
    length_2 = pairs["len2"].to_numpy(float)
    raw_geometric = raw_score / np.sqrt(np.maximum(self_1 * self_2, EPS))

    features = pd.DataFrame(index=pairs.index)
    features["alignment_norm"] = normalized
    features["alignment_raw_geom"] = raw_geometric
    features["length_similarity"] = np.minimum(length_1, length_2) / np.maximum(length_1, length_2)
    for name in ["primary", "all", "ss", "joint", "nint"]:
        features[f"js_{name}"] = token_features[f"token_{name}_js_similarity"].to_numpy()

    for proxy in ["kmer3", "compress", "length", "entropy", "repeat_consensus"]:
        left = pairs[f"rp_{proxy}_1"].to_numpy(float)
        right = pairs[f"rp_{proxy}_2"].to_numpy(float)
        features[f"repeat_{proxy}_max"] = np.maximum(left, right)
        features[f"repeat_{proxy}_min"] = np.minimum(left, right)

    repeat_rank = features["repeat_repeat_consensus_max"].to_numpy()
    features["repeat_x_raw_minus_norm"] = repeat_rank * (raw_geometric - normalized)
    features["repeat_x_joint_js"] = repeat_rank * features["js_joint"]
    features["raw_x_primary_js"] = raw_geometric * features["js_primary"]
    return features.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def protein_identity_folds(pairs: pd.DataFrame, number_of_folds=5):
    """Return folds with no protein identity shared between train and test."""
    proteins = pd.concat(
        [
            pairs[["pdb_id_1", "family_1"]].rename(
                columns={"pdb_id_1": "pdb_id", "family_1": "family"}
            ),
            pairs[["pdb_id_2", "family_2"]].rename(
                columns={"pdb_id_2": "pdb_id", "family_2": "family"}
            ),
        ]
    ).drop_duplicates()
    fold_map = {}
    for _, group in proteins.groupby("family", sort=True):
        for index, row in enumerate(group.sort_values("pdb_id").itertuples(index=False)):
            fold_map[(str(row.family), str(row.pdb_id))] = index % number_of_folds

    left_fold = np.array(
        [fold_map[(str(family), str(pdb))] for family, pdb in zip(pairs.family_1, pairs.pdb_id_1)]
    )
    right_fold = np.array(
        [fold_map[(str(family), str(pdb))] for family, pdb in zip(pairs.family_2, pairs.pdb_id_2)]
    )
    folds = []
    for fold in range(number_of_folds):
        train = np.flatnonzero((left_fold != fold) & (right_fold != fold))
        test = np.flatnonzero((left_fold == fold) & (right_fold == fold))
        train_proteins = set(pairs.iloc[train].pdb_id_1) | set(pairs.iloc[train].pdb_id_2)
        test_proteins = set(pairs.iloc[test].pdb_id_1) | set(pairs.iloc[test].pdb_id_2)
        if train_proteins & test_proteins:
            raise AssertionError("Protein identity leaked between training and validation")
        folds.append((train, test))
    return folds


def logistic_model(regularization_c):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=regularization_c,
            penalty="l2",
            class_weight="balanced",
            solver="lbfgs",
            max_iter=20000,
        ),
    )


def load_feature_matrices():
    small = pd.read_csv(SMALL_SOURCE)
    large_processed = pd.read_csv(LARGE_SOURCE)
    large_raw = pd.read_csv(LARGE_RAW_SOURCE)
    large_raw = large_raw[large_raw["status"].astype(str).str.lower().eq("ok")]
    merge_keys = ["pdb_id_1", "family_1", "pdb_id_2", "family_2", "pair_label", "pair_type"]
    large = large_processed.merge(large_raw, on=merge_keys, validate="one_to_one")

    tokens = pd.read_csv(TOKENS_SOURCE)
    protein_metrics = build_protein_metrics(tokens)
    token_lookup = build_token_lookup(tokens)
    small = add_repeat_features(small, protein_metrics)
    large = add_repeat_features(large, protein_metrics)
    small_features = build_features(small, token_js_features(small, token_lookup))
    large_features = build_features(large, token_js_features(large, token_lookup))
    return small, large_processed, large, small_features, large_features


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    small, large_processed, large, small_features, large_features = load_feature_matrices()
    labels = small["pair_label"].astype(int).to_numpy()
    folds = protein_identity_folds(small)

    search_rows = []
    best_by_candidate = {}
    for candidate_name, columns in CANDIDATES.items():
        best = None
        for regularization_c in REGULARIZATION_GRID:
            predictions = np.full(len(labels), np.nan)
            fold_aucs = []
            for train, test in folds:
                model = logistic_model(regularization_c)
                model.fit(small_features.iloc[train][columns], labels[train])
                predictions[test] = model.predict_proba(small_features.iloc[test][columns])[:, 1]
                fold_aucs.append(float(roc_auc_score(labels[test], predictions[test])))
            valid = ~np.isnan(predictions)
            heldout_auc, heldout_auprc = metrics(labels[valid], predictions[valid])
            search_rows.append(
                {
                    "candidate": candidate_name,
                    "n_terms": len(columns),
                    "C": regularization_c,
                    "protein_heldout_auc": heldout_auc,
                    "protein_heldout_auprc": heldout_auprc,
                    "fold_aucs": json.dumps(fold_aucs),
                    "heldout_pairs": int(valid.sum()),
                }
            )
            result = (heldout_auc, heldout_auprc, regularization_c)
            if best is None or result[:2] > best[:2]:
                best = result
        best_by_candidate[candidate_name] = best

    search = pd.DataFrame(search_rows)
    search.to_csv(SEARCH_OUTPUT, index=False)
    best_auc = max(result[0] for result in best_by_candidate.values())
    eligible = [
        (len(CANDIDATES[name]), name, result)
        for name, result in best_by_candidate.items()
        if result[0] >= best_auc - 0.005
    ]
    _, selected_name, (heldout_auc, heldout_auprc, selected_c) = min(eligible)
    selected_columns = CANDIDATES[selected_name]

    final_model = logistic_model(selected_c)
    final_model.fit(small_features[selected_columns], labels)
    small_scores = final_model.predict_proba(small_features[selected_columns])[:, 1]
    large_scores = final_model.predict_proba(large_features[selected_columns])[:, 1]
    small_auc, small_auprc = metrics(labels, small_scores)
    large_auc, large_auprc = metrics(large["pair_label"].astype(int), large_scores)
    scaler = final_model[0]
    classifier = final_model[1]

    positive_indices = np.flatnonzero(labels == 1)
    negative_indices = np.flatnonzero(labels == 0)
    bootstrap_coefficients = []
    for _ in range(250):
        sample = np.concatenate(
            [
                RNG.choice(positive_indices, len(positive_indices), replace=True),
                RNG.choice(negative_indices, len(negative_indices), replace=True),
            ]
        )
        bootstrap_model = logistic_model(selected_c)
        bootstrap_model.fit(small_features.iloc[sample][selected_columns], labels[sample])
        bootstrap_coefficients.append(bootstrap_model[-1].coef_[0])
    bootstrap_coefficients = np.asarray(bootstrap_coefficients)
    stability = pd.DataFrame(
        {
            "feature": selected_columns,
            "definition": [DEFINITIONS[column] for column in selected_columns],
            "training_mean": scaler.mean_,
            "training_scale": scaler.scale_,
            "coefficient": classifier.coef_[0],
            "bootstrap_p2_5": np.percentile(bootstrap_coefficients, 2.5, axis=0),
            "bootstrap_p97_5": np.percentile(bootstrap_coefficients, 97.5, axis=0),
            "bootstrap_sign_stability": np.maximum(
                np.mean(bootstrap_coefficients > 0, axis=0),
                np.mean(bootstrap_coefficients < 0, axis=0),
            ),
        }
    )
    stability.to_csv(STABILITY_OUTPUT, index=False)

    specification = {
        "method_name": "interpretable_family_independent_formula_v1",
        "model_type": "L2-regularized logistic regression over standardized features",
        "selected_candidate": selected_name,
        "term_count": len(selected_columns),
        "selection_rule": "Choose the fewest-term candidate within 0.005 ROC AUC of the best protein-held-out candidate",
        "family_identity_used_as_predictor": False,
        "family_equality_used": False,
        "small_labels_used_to_fit_coefficients": True,
        "large_labels_used_to_fit_or_select": False,
        "repeat_percentile_preprocessing": "Unlabeled cohort-relative percentile ranks computed separately within the small and large protein cohorts",
        "selected_regularization_C": selected_c,
        "feature_names_in_order": selected_columns,
        "feature_definitions": {column: DEFINITIONS[column] for column in selected_columns},
        "training_means": dict(zip(selected_columns, scaler.mean_.tolist())),
        "training_scales": dict(zip(selected_columns, scaler.scale_.tolist())),
        "coefficients": dict(zip(selected_columns, classifier.coef_[0].tolist())),
        "intercept": float(classifier.intercept_[0]),
        "score_definition": "sigmoid(intercept + sum coefficient_i*((feature_i-training_mean_i)/training_scale_i))",
        "small_full_auc": small_auc,
        "small_full_auprc": small_auprc,
        "small_protein_heldout_auc": heldout_auc,
        "small_protein_heldout_auprc": heldout_auprc,
        "large_frozen_auc": large_auc,
        "large_frozen_auprc": large_auprc,
        "small_source_sha256": sha256(SMALL_SOURCE),
        "large_source_sha256": sha256(LARGE_SOURCE),
        "tokens_source_sha256": sha256(TOKENS_SOURCE),
    }
    unsigned = json.dumps(specification, sort_keys=True, separators=(",", ":")).encode()
    specification["spec_sha256_without_this_field"] = hashlib.sha256(unsigned).hexdigest()
    FORMULA_OUTPUT.write_text(json.dumps(specification, indent=2))

    scored_small = pd.read_csv(SMALL_SOURCE)
    scored_small[SCORE_COL] = small_scores
    scored_small.to_csv(SMALL_SCORED_OUTPUT, index=False)
    scored_large = large_processed.copy()
    scored_large[SCORE_COL] = large_scores
    scored_large.to_csv(LARGE_SCORED_OUTPUT, index=False)

    summary = pd.DataFrame(
        [
            {"dataset": "small", "role": "development fit", "roc_auc": small_auc, "auprc": small_auprc},
            {"dataset": "small", "role": "protein-held-out selection", "roc_auc": heldout_auc, "auprc": heldout_auprc},
            {"dataset": "large", "role": "frozen application", "roc_auc": large_auc, "auprc": large_auprc},
        ]
    )
    summary.to_csv(SUMMARY_OUTPUT, index=False)

    print("Best candidate results:")
    for name, (candidate_auc, candidate_auprc, candidate_c) in best_by_candidate.items():
        print(
            f"  {name:10s} AUC={candidate_auc:.6f}, "
            f"AUPRC={candidate_auprc:.6f}, C={candidate_c:g}"
        )
    print(f"\nSelected: {selected_name}, C={selected_c:g}")
    print(summary.to_string(index=False))
    print(f"\nWrote frozen model and scored benchmarks to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
