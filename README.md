# Interaction-Token Structural Alignment (ITSA)

Reviewer-accessible code and data for the manuscript:

**Interaction-Token Structural Alignment (ITSA): An Interpretable Framework for Protein Similarity Based on Residue-Level Chemical Interactions**

ITSA converts protein structures into residue-ordered sequences of biochemical interaction tokens and compares those token sequences using Smith-Waterman local alignment. The goal of the method is to provide an interpretable interaction-centered signal for protein similarity based on conserved residue-level biochemical environments.

This repository is provided for confidential editorial and peer-review access.

---

## What Each Code File Does

### `tokenizer.py`

This script converts raw PDB structures into ITSA interaction-token sequences.

It performs:

- PDB parsing
- standard amino-acid residue filtering
- backbone atom validation using `N`, `CA`, and `C`
- DSSP-based secondary-structure assignment when available
- fallback coil assignment when DSSP is unavailable
- local residue-neighborhood search
- residue-level interaction classification
- token construction
- JSON/CSV export of the resulting token sequences

Interaction tokens:

```text
H = hydrogen bond
I = ionic interaction
Y = hydrophobic interaction
V = van der Waals contact
P = aromatic stacking
D = disulfide bond
U = undefined / no dominant interaction
```

Input:

```text
PDB files organized by family
benchmark/family metadata CSV
```

Output:

```text
protein_tokens.csv
```

---

### `itsa_precomputed_tokens.cpp`

This is the optimized C++ ITSA alignment and scoring implementation.

It takes precomputed token sequences from `tokenizer.py` and computes pairwise ITSA alignment scores.

It performs:

- loading precomputed ITSA token CSVs
- parsing serialized token JSON
- Smith-Waterman local alignment
- interaction-token substitution scoring
- secondary-structure contextual scoring
- extra-interaction overlap scoring
- raw Smith-Waterman score calculation
- self-alignment score calculation
- normalized ITSA score calculation
- pairwise result CSV generation

Input:

```text
benchmark pair CSV
protein_tokens.csv
```

Output:

```text
itsa_cpp.csv
```

Main output columns:

```text
sw_raw_score
sw_norm
sw_self1_score
sw_self2_score
len1
len2
status
```

Compile:

```bash
g++ -O3 -std=c++17 -pthread code/itsa_precomputed_tokens.cpp -o itsa_precomputed_tokens
```

Example run:

```bash
./itsa_precomputed_tokens \
  --input_csv data/benchmarks/curated_pairs.csv \
  --tokens_csv data/outputs/protein_tokens_curated.csv \
  --output_csv data/outputs/itsa_cpp_curated.csv \
  --workers 8
```

---

### `compute_js_hybrid_score.py`

This script computes the final calibrated/blended ITSA score used for benchmark evaluation.

It performs:

- loading C++ ITSA pairwise score outputs
- loading token CSVs
- extracting primary interaction tokens
- computing token-distribution vectors
- computing Jensen-Shannon token-distribution similarity
- min-max scaling
- rank scaling
- log-scaling of raw Smith-Waterman scores
- family-aware hybrid raw/normalized score construction
- final blended score calculation
- ROC AUC calculation
- AUPRC calculation
- precision-at-k style summaries

Input:

```text
itsa_cpp.csv
protein_tokens.csv
```

Output:

```text
printed evaluation metrics
final hybrid-score analysis
```

This file is used to evaluate the ITSA scores reported in the manuscript.

---

### `generate_paper_figures.py`

This script regenerates the manuscript figures from saved result CSV files.

It generates:

- substitution matrix heat map
- curated benchmark ROC curve
- large benchmark ROC curve
- large benchmark score distribution
- TM-align vs ITSA comparison plot
- entropy vs performance plots

Input:

```text
ITSA result CSVs
token CSVs
TM-align result CSVs
```

Output:

```text
figure image files saved to figures/
```

Example run:

```bash
python code/generate_paper_figures.py
```

---

### `tm_align_wrapper.py`

This script runs the external TM-align baseline and parses the results.

It performs:

- loading the same benchmark pair CSV format used by ITSA
- locating PDB files by family and PDB ID
- running the external `TMalign` executable
- parsing TM-score output
- extracting TM-score normalized by chain 1
- extracting TM-score normalized by chain 2
- calculating average TM-score
- extracting RMSD, aligned length, and sequence identity
- writing TM-align results to CSV
- optional multiprocessing
- optional single-family filtering

Input:

```text
benchmark pair CSV
PDB files organized by family
external TM-align executable
```

Output:

```text
tmalign_results.csv
```

TM-align itself is not included in this repository. To run this script, install TM-align separately and either place `TMalign` on PATH or set:

```bash
export TMALIGN_BIN=/path/to/TMalign
```

Example run:

```bash
TMALIGN_BIN=/path/to/TMalign python code/tm_align_wrapper.py \
  --input_csv data/benchmarks/curated_pairs.csv \
  --output_csv data/outputs/tmalign_results.csv \
  --base_dir data/pdbs \
  --workers 8
```

---

## Repository Structure

```text
ITSA/
  code/
    tokenizer.py
    itsa_precomputed_tokens.cpp
    compute_js_hybrid_score.py
    generate_paper_figures.py
    tm_align_wrapper.py

  data/
    pdbs/
      beta_propeller/
      cytochrome_c/
      globins/
      immunoglobulin/
      insulin_like/
      lysozyme/
      outer_membrane_barrel/
      rossmann_fold/
      tim_barrel/
      zinc_finger/
      ...

    benchmarks/
      curated_pairs.csv
      large_pairs.csv

    outputs/
      protein_tokens_curated.csv
      protein_tokens_large.csv
      itsa_cpp_curated.csv
      itsa_cpp_large.csv
      itsa_results_with_js_hybrid.csv
      tmalign_results.csv

  figures/
    generated_figures/

  README.md
  requirements.txt
```

The exact folder names may differ slightly depending on the uploaded repository organization, but the expected workflow is the same: PDB files are organized by protein family, benchmark CSVs define pairwise comparisons, and output CSVs contain tokenized structures, ITSA scores, hybrid scores, and TM-align outputs.

---

## Data Files

### PDB files

PDB structures are organized by SCOP family:

```text
data/pdbs/<family>/<pdb_id>.pdb
```

Example:

```text
data/pdbs/beta_propeller/1gy3.pdb
data/pdbs/beta_propeller/2r6c.pdb
```

---

### Benchmark pair CSVs

The benchmark CSVs define the exact pairwise comparisons used in the manuscript.

Expected columns:

```text
pdb_id_1
family_1
pdb_id_2
family_2
pair_label
pair_type
```

Where:

```text
pair_label = 1 for same-family positive pairs
pair_label = 0 for different-family negative pairs
```

The curated benchmark corresponds to the 4,950-pair benchmark described in the manuscript.

The large benchmark corresponds to the 44,253-pair benchmark described in the manuscript.

---

### Precomputed token CSVs

These contain serialized ITSA tokens generated from the PDB files.

Expected columns include:

```text
pdb_id
family
tokens_json
status
```

These files let reviewers inspect or reproduce downstream scoring without rerunning the full tokenization step.

---

### ITSA result CSVs

These are produced by the C++ scoring implementation.

Expected columns include:

```text
pdb_id_1
family_1
pdb_id_2
family_2
pair_label
pair_type
sw_raw_score
sw_norm
sw_self1_score
sw_self2_score
len1
len2
status
```

Only rows with:

```text
status = ok
```

are used for final evaluation.

---

### TM-align result CSVs

These are produced by `tm_align_wrapper.py`.

Expected columns include:

```text
tm_score_1
tm_score_2
tm_score_avg
rmsd
aligned_len
seq_id
status
```

The manuscript uses:

```text
tm_score_avg
```

for the TM-align AUC comparison.

---

## Reproducibility Workflow

Full ITSA workflow:

```text
PDB structures
→ tokenizer.py
→ protein_tokens.csv
→ itsa_precomputed_tokens.cpp
→ itsa_cpp.csv
→ compute_js_hybrid_score.py
→ final scores and AUC/AUPRC
→ generate_paper_figures.py
→ manuscript figures
```

TM-align comparison workflow:

```text
PDB structures + benchmark pair CSV
→ tm_align_wrapper.py
→ tmalign_results.csv
→ generate_paper_figures.py
→ TM-align comparison figures
```

Reviewers may either rerun the full pipeline from the provided PDB structures or use the included precomputed token and result CSVs.

---

## Python Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

Recommended `requirements.txt`:

```text
biopython
numpy
pandas
scipy
scikit-learn
matplotlib
```

DSSP is recommended for secondary-structure assignment. If DSSP is unavailable, the tokenizer falls back to coil assignments.

---

## External Dependencies

### DSSP

DSSP is used for secondary-structure assignment through Biopython when available.

If DSSP is unavailable or fails for a structure, residues default to coil secondary structure.

### TM-align

TM-align is used only for the geometry-centered baseline comparison.

TM-align is not included in this repository. Reviewers who wish to rerun the TM-align baseline should install TM-align separately and set the executable path using `TMALIGN_BIN`.

---

## Notes for Reviewers

This repository contains the code and data materials used to generate the submitted ITSA benchmark results and manuscript figures.

The code is provided for confidential peer review. Because the ITSA methodology is associated with an active patent process, the implementation is being made available to the Editorial Board and reviewers for review access rather than as a public code supplement at this stage.

The repository includes:

```text
- ITSA tokenizer
- optimized C++ ITSA alignment/scoring implementation
- Jensen-Shannon hybrid scoring script
- figure-generation script
- TM-align wrapper/parser
- benchmark pair CSVs
- PDB structures organized by SCOP family
- precomputed token CSVs
- ITSA result CSVs
- TM-align result CSVs, where applicable
```

---

## Contact

For questions about the repository or reviewer access, please contact:

**Samanyu Kulkarni**  
samanyuckulkarni@gmail.com
