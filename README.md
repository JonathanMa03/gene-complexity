# Interpretable Gene Programs or Foundation Models?

**Comparing Sparse Dictionary Learning and scGPT for Single-Cell Classification**

**Course:** EN.553.740 Machine Learning 1

**Instructor:** Dr. Sichen Yang

**Group members:** [Jonathan Ma](https://jonathanma03.github.io/), [Sijia Zhu](https://chesleaz.github.io/sijiazhu.github.io/), Jiatong Gao, Yunwei Chai

## Project Overview

This project will compare pretrained scGPT representations with sparse dictionary learning for cell-type classification using public single-cell RNA-sequencing data. Dictionary learning represents each cell as a sparse combination of learned gene programs, providing an interpretable alternative to foundation-model embeddings. Both representations will be evaluated with the same regularized multinomial logistic regression classifier, alongside a baseline using processed gene-expression features directly.

We will compare Macro-F1, balanced accuracy, and predictive log loss, and examine the highest-weight genes and cell-type activation patterns of the learned dictionary atoms. The goal is to understand whether pretrained scGPT provides predictive information beyond a sparse representation learned from the target data, and to characterize the tradeoff between predictive performance and interpretability.

See the [project abstract](docs/references/group1_abstract.pdf) and [project proposal](docs/references/group1_proposal.pdf) for more detail.

## Data

Download the [Pancreas benchmark dataset](https://exampledata.scverse.org/scvi-tools/pancreas.h5ad) and save it as `data/raw/pancreas.h5ad`. This dataset is described in the [scvi-tools reference-mapping tutorial](https://docs.scvi-tools.org/en/latest/tutorials/notebooks/multimodal/scarches_scvi_tools.html).

From the repository root:

```bash
mkdir -p data/raw
curl -fL https://exampledata.scverse.org/scvi-tools/pancreas.h5ad \
  -o data/raw/pancreas.h5ad
```

Downloaded datasets are excluded from Git. Each teammate should download their own local copy. Open `environment_test.ipynb` to inspect the expression data and cell metadata. Preprocessing details will be added as the pipeline is developed.

## Methods

TBD — Representation-learning pipelines, classifier setup, and evaluation procedures will be added here.

## Repository Structure

```text
gene-complexity/
├── .github/
│   └── CODEOWNERS
├── code/
│   ├── dict-learn/              # Sparse dictionary learning
│   ├── preprocessing/           # Data preparation
│   └── scgpt/                   # scGPT representations
├── data/                        # Project datasets
├── docs/
│   ├── logging/
│   │   └── CHANGELOG.md
│   ├── notes/
│   │   └── main.tex             # LaTeX reference notes
│   ├── references/
│   │   ├── denAdel.pdf
│   │   ├── group1_abstract.pdf
│   │   └── group1_proposal.pdf
│   ├── Python_git.md
│   └── R_git.md
├── results/                     # Experiment outputs and figures
├── .gitignore
├── LICENSE
├── README.md
├── environment_test.ipynb       # Environment checks and data preview
└── requirements.txt            # Python dependencies
```

Empty project directories contain `.gitkeep` placeholders, omitted above for readability.

## Getting Started

Clone the repository and create a Python virtual environment:

```bash
git clone https://github.com/JonathanMa03/gene-complexity.git
cd gene-complexity
python3 -m venv .venv
```

Activate the environment:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the dependencies (PyTorch, NumPy, pandas, Matplotlib, Jupyter Notebook, and AnnData), register the project kernel, then launch the environment test:

```bash
python -m pip install -r requirements.txt
python -m ipykernel install --user --name gene-complexity --display-name "Python (gene-complexity)"
python -m notebook environment_test.ipynb
```

Select the **Python (gene-complexity)** kernel and run all cells. The notebook checks that the kernel uses this repository's `.venv`, imports every dependency in `requirements.txt`, and previews the downloaded dataset. The notebook dependencies include `ipykernel` transitively. On Windows, use `python` instead of `python3` when creating the environment.

Additional dependencies for the project pipelines will be documented as they are implemented.

## Team Workflow: Branches and Pull Requests

1. Start from an up-to-date `main` branch with no uncommitted changes:

   ```bash
   git switch main
   git pull --ff-only origin main
   ```

2. Create your own branch for a task. Replace the example with your name and a short task description:

   ```bash
   git switch -c jonathan/data-preprocessing
   ```

3. Make and check your changes, then stage the relevant files and commit. Replace the example path with the files you changed:

   ```bash
   git status
   git diff
   git add code/preprocessing/example.py
   git commit -m "Add initial data preprocessing"
   ```

4. Push your branch to GitHub:

   ```bash
   git push -u origin your-name-here/data-preprocessing
   ```

5. Open the [repository on GitHub](https://github.com/JonathanMa03/gene-complexity) and select **Compare & pull request**, or open **Pull requests → New pull request**. Set the base branch to `main` and the compare branch to your task branch. Describe what changed and how you checked it, then request a teammate's review.

6. Address review feedback by committing and pushing to the same branch; the pull request updates automatically. After review and any required checks pass, merge the pull request into `main` and delete the merged branch on GitHub.

7. Update your local `main` before starting your next task:

   ```bash
   git switch main
   git pull --ff-only origin main
   ```

Keep each pull request focused on one task and use a new branch for subsequent work.
