# Interpretable Gene Programs or Foundation Models?

**Comparing Sparse Dictionary Learning and scGPT for Single-Cell Classification**

**Course:** EN.553.740 Machine Learning 1

**Instructor:** Dr. Sichen Yang

**Group members:** [Jonathan Ma](https://jonathanma03.github.io/), [Sijia Zhu](https://chesleaz.github.io/sijiazhu.github.io/), Jiatong Gao, Yunwei Chai

## Abstract

This project will compare pretrained scGPT representations with sparse dictionary learning for cell-type classification using public single-cell RNA-sequencing data. Dictionary learning represents each cell as a sparse combination of learned gene programs, providing an interpretable alternative to foundation-model embeddings. Both representations will be evaluated with the same regularized multinomial logistic regression classifier, alongside a baseline using processed gene-expression features directly.

We will compare Macro-F1, balanced accuracy, and predictive log loss, and examine the highest-weight genes and cell-type activation patterns of the learned dictionary atoms. The goal is to understand whether pretrained scGPT provides predictive information beyond a sparse representation learned from the target data, and to characterize the tradeoff between predictive performance and interpretability.

See the [project abstract](docs/references/group1_abstract.pdf) and [project proposal](docs/references/group1_proposal.pdf) for more detail.

## Reproduce with the notebook

Clone the repository, download the two resources below, install the project in a virtual environment, and run **[workflow.ipynb](workflow.ipynb) from top to bottom**. This setup has been verified in a clean checkout without previous results or the old `code/` directory.

The default run trains and evaluates logistic classifiers using expression, sparse dictionary, and scGPT features. It displays validation metrics and figures and saves them under `results/workflow/ta_notebook_smoke_v2/`. This is a small real-data execution check; model selection, final test evaluation, and donor experiments require explicitly enabling the notebook's optional sections.

### 1. Clone and download

```bash
git clone https://github.com/JonathanMa03/gene-complexity.git
cd gene-complexity
```

**Google Drive download: [TBD — add the project resource-bundle link here]**

Download both resources from that folder:

- **`baron_raw.h5ad`** — put it in `data/processed/`.
- **`external.zip`** — extract it at the repository root to create `external/`.

Use the supplied raw-count file, which matches the embedding provenance. A separately downloaded or re-exported Baron file may fail the intentional hash checks. The underlying public dataset is [Baron human pancreas, GSE84133](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE84133).

After extraction, the layout should be:

```text
gene-complexity/
├── workflow.ipynb
├── gene_complexity/
├── pyproject.toml
├── requirements.txt
├── data/processed/baron_raw.h5ad
└── external/
    ├── bundle_manifest.json
    ├── baron_splits_final/
    │   ├── baron_split_manifest.csv
    │   ├── baron_exclusions.csv
    │   ├── split_settings.json
    │   └── ...                     # Cell-count tables
    ├── scgpt/pretraining_handoff/
    │   ├── assets/                 # Fixed gene panel and vocabulary
    │   ├── outputs/baron/
    │   │   ├── baron_scgpt_full.npz
    │   │   ├── baron_scgpt_full.metadata.json
    │   │   ├── checkpoint_selection.json
    │   │   └── preparation_report.json
    │   └── ...                     # Handoff code and provenance
    └── report_artifacts/           # Saved sources for optional paper snapshots
```

Avoid an extra `external/external/` nesting level. Keep the downloaded files unchanged. No precomputed dictionary features, trained classifiers, or previous `results/` directory are needed. The default run uses supplied scGPT embeddings and requires neither R, GPU access, nor checkpoint weights.

### 2. Set up Python

Create a virtual environment from the repository root:

```bash
python3 -m venv .venv
```

Activate it:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

On Windows, use `python` instead of `python3` if needed. Then install the package and its dependencies from `requirements.txt`, and register the notebook kernel:

```bash
python -m pip install -e .
python -m ipykernel install --user --name gene-complexity --display-name "Python (gene-complexity)"
```

### 3. Check inputs and run

```bash
python -m gene_complexity run-end-to-end --check-only
python -m notebook workflow.ipynb
```

Select **Python (gene-complexity)** and choose **Restart Kernel and Run All Cells**. In VS Code, open the notebook, select the same kernel or the project's `.venv`, and run all cells.

Leave the configuration defaults in place for the first run. The notebook checks input files, provenance hashes, and stage dependencies before fitting models. It generates processed features, dictionary codes, classifiers, validation metrics, confusion matrices, and dictionary interpretation figures.

## Changing or repeating a run

- `PROFILE="smoke"` uses 256 genes, 8 dictionary atoms, and at most 2 epochs. `"standard"` uses 2,000 genes, 32 atoms, and at most 10 epochs. Neither preset performs model selection automatically.
- The configuration cell exposes dictionary size, sparsity penalty, classifier regularization, and other settings. Set `RUN_END_TO_END=False` to run individual stages after setup.
- Choose a new `RUN_NAME` after changing settings or code, or after a failed/partial run. An unchanged completed run verifies and reuses its saved outputs.
- Optional model-selection, freezing, test-evaluation, and donor-study cells are disabled by default. Follow their notebook instructions when running the full experimental workflow.

Component documentation is in [docs/logging/](docs/logging/). For the optional command-line interface, run `python -m gene_complexity --help`.
