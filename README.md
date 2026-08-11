# ITSA

This repository contains the code and data layout needed to reproduce the analyses and figures described in the manuscript.

## Overview

The code reproduces an interpretable, family-independent six-term ITSA score and the figures derived from its frozen application to benchmark datasets.

## Repository structure

- `Code/`
  - Python and C++ source files for tokenization, model training, scoring, and figure generation.
- `Results/`
  - Output files produced by the training and figure-generation scripts.
- `Datasets/`
  - Input benchmark datasets and token metadata required by the scripts.
- `44,253 Pair PDB Files/`
  - Large-scale pairwise PDB files organized by structural family.
- `4950 Pair PDB Files/`
  - Curated benchmark pairwise PDB files.

## Reviewer reproduction steps

Run these commands from the repository root.

1. Install dependencies:

```bash
pip install numpy pandas scikit-learn matplotlib scipy biopython
```

2. Generate or verify the token cache (if not already available):

```bash
python3 Code/tokenizer.py \
  --input_csv Datasets/master_dataset.csv \
  --base_dir SCOP_Dataset \
  --output_csv Datasets/protein_tokens.csv \
  --workers 8
```

3. Train and freeze the six-term model and score the benchmarks:

```bash
python3 Code/train_six_term_model.py
```

4. Generate the figures from the frozen score outputs:

```bash
python3 Code/generate_paper_figures.py
```

## Notes for reviewers

- `Code/train_six_term_model.py` fits and selects the final frozen model using the small benchmark, then applies it to the large benchmark.
- `Code/generate_paper_figures.py` expects the scored benchmark files and formula specification to be present in `Results/`.
- `Code/itsa_precomputed_tokens.cpp` is a supplementary helper to build cached pairwise comparison data from precomputed tokens.
- For code-level details and alternative execution paths, see `Code/readme.md`.

