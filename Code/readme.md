# Code Directory

This directory contains the scripts and helper program needed to reproduce the model fitting, frozen score generation, and figure production used in the manuscript.

## Quick reproduction steps

From the repository root, run these commands in order:

1. Build the token cache (if needed):

```bash
python3 Code/tokenizer.py \
  --input_csv Datasets/master_dataset.csv \
  --base_dir SCOP_Dataset \
  --output_csv Datasets/protein_tokens.csv \
  --workers 8
```

2. Train and freeze the final six-term model:

```bash
python3 Code/train_six_term_model.py
```

3. Generate the figures:

```bash
python3 Code/generate_paper_figures.py
```

## Files

### `tokenizer.py`

Tokenizes proteins and creates a cached CSV of residue interaction tokens.

- Input: a master dataset CSV with at least `pdb_id` and `family` columns.
- Output: `protein_tokens.csv` containing `tokens_json`, `primary_seq`, and token status metadata.
- Notes: uses BioPython to parse PDB files and extract interaction-based residue tokens.

### `train_six_term_model.py`

Trains, selects, freezes, and evaluates the interpretable six-term ITSA model.

- Input:
  - `Results/small_benchmark_pairwise_results.csv`
  - `Results/large_benchmark_pairwise_results.csv`
  - `Datasets/large_benchmark/protein_tokens.csv`
- Process:
  - computes token-based Jensen-Shannon similarities,
  - derives alignment and repeat-sensitive features,
  - performs protein-identity-held-out cross-validation,
  - selects the smallest candidate within 0.005 ROC AUC of the best,
  - fits the final model on the full small benchmark.
- Output:
  - `Results/small_benchmark_six_term_scores.csv`
  - `Results/large_benchmark_six_term_scores.csv`
  - `Results/frozen_interpretable_formula_v1.json`
  - summary CSVs in `Results/`

### `generate_paper_figures.py`

Produces figures from frozen scores and auxiliary benchmark data.

- Input:
  - scored benchmark CSVs in `Results/`
  - `Results/frozen_interpretable_formula_v1.json`
  - TM-align output under `TM-Align/`
  - curated token entropy data under `Datasets/curated_benchmark/`
- Output: PNG files in `Results/figures/`.

### `itsa_precomputed_tokens.cpp`

C++ helper for caching precomputed tokens and writing batch pairwise results.

- Compile with a C++17 compiler:

```bash
g++ -std=c++17 -O2 Code/itsa_precomputed_tokens.cpp -o Code/itsa_cached_dssp
```

- Run with:

```bash
./Code/itsa_cached_dssp \
  --input_csv file.csv \
  --output_csv out.csv \
  --tokens_csv Datasets/protein_tokens.csv \
  --base_dir SCOP_Dataset \
  --dssp_bin mkdssp \
  --gap -2.0 \
  --workers 8
```

## Dependencies

- Python 3
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `scipy`
- `biopython`

Install with:

```bash
pip install numpy pandas scikit-learn matplotlib scipy biopython
```

## Notes

- The Python scripts use fixed output paths under `Results/`.
- Run from the repository root or adjust the path references as needed.
- This Code directory is organized so reviewers can reproduce the full pipeline or run individual steps if frozen outputs already exist.
