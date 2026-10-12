# Dictionary-learning baseline

Open [synthetic_demo.ipynb](synthetic_demo.ipynb) in VS Code, select the project's **Python (gene-complexity)** kernel, and choose **Run All**. The notebook runs the shared Python implementation and displays data previews, reconstruction metrics, sparsity, learned atoms, and the training objective. Each run saves artifacts in a new `results/dict-learning/notebook_seed.../` directory, preserving earlier runs. It works with the working directory set to either the repository root or this folder.

Alternatively, launch it from the repository root after activating `.venv`:

```bash
python -m notebook code/dict-learn/synthetic_demo.ipynb
```

From the repository root, activate `.venv`, install the root requirements, and run:

```bash
python -m pip install -r requirements.txt
python code/dict-learn/synthetic_demo.py
```

The deterministic toy dataset has 240 cells, 40 synthetic genes, eight signed unit-norm atoms, two nonzero generating coefficients per cell, and Gaussian noise. It represents processed continuous features, **not raw RNA counts**. Splits are 160 training, 40 validation, and 40 test cells. It contains no biological labels.

The script fits training-mean centering and a small dictionary on training cells only, then encodes all cells with the dictionary frozen. It explicitly uses penalized Lasso encoding (`lasso_cd`, matching training `alpha`) rather than the library's default OMP encoding. See the [scikit-learn objective and API](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.DictionaryLearning.html) and [theory note](../../docs/notes/jm.tex).

Outputs under `results/dict-learning/synthetic_seed42/`:

- `synthetic_data.npz`: expression `X`, true factors, cell/gene IDs, and splits.
- `dictionary.npz`: learned dictionary, ordered gene IDs, training mean, penalty, training cell IDs.
- `codes.npz`: `vectors`, ordered `cell_ids`, and splits.
- `metrics.json`: reconstruction, sparsity, objective history, iterations, settings, versions.

Load arrays with `np.load(path, allow_pickle=False)`. The run checks atom norms, finite codes, frozen-dictionary transformation, reconstruction against zero, and re-encoding after export/reload. Factor recovery is not tested elementwise because signs and atom ordering are not identifiable. Validation/test metrics here check the synthetic pipeline; they are not used for tuning or claims about real data.

Existing output directories are preserved. Use `--output results/dict-learning/another_run` for another run and `--seed 7` to vary the synthetic data and learner initialization. Generated outputs are ignored by Git; the generator is tracked.

`fit_dictionary(X_train, ...)` is the minimal reusable entry point for already-processed dense features. Real-data ingestion, sparse/memory-efficient processing, hyperparameter search, labels, classifiers, and biological interpretation remain future work. Real-data callers must enforce the shared cell/gene ID and split contract described in the [preprocessing handoff](../../docs/notes/preprocessing_handoff.md).
