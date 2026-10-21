# Reusable dictionary learning

All dictionary-specific implementations live in `gene_complexity/dictionary_learning/`. Run public commands from the repository root with `.venv` activated.

| Module | Responsibility |
|---|---|
| `training.py` | Verify classical features, fit a training dictionary, freeze it for validation encoding, export arrays and diagnostics. |
| `grid.py` | Fit a configurable candidate grid and verify matching cached runs. |
| `solver.py` | Sparse encoding, controlled initialization, KKT/projected-gradient diagnostics, and sign/permutation-aware atom matching. Used by donor evaluation and held-out encoding. |
| `interpretation.py` | Export signed gene weights and cell-type activities, with optional plots. |
| `configs/` | Historical starter grid and refined dictionary grid. |

```bash
python -m gene_complexity run-dictionary --atoms 64 --alpha 2 --epochs 10 --seed 42 \
  --output results/dict-learning/new_dictionary
python -m gene_complexity dictionary-grid \
  --config gene_complexity/dictionary_learning/configs/dictionary_grid_v2.json \
  --output results/dict-learning/new_grid
python -m gene_complexity tune-models --output results/workflow/new_selection
```

The single-run input defaults to `data/processed/baron_classical_v1/`. Fitting uses only training cells; validation encoding does not update atoms. Outputs contain `dictionary.npz`, `train_codes.npz`, `validation_codes.npz`, and `pilot_report.json`. The historical report filename/status are retained for compatibility with verified existing artifact bundles; this is the reusable real-data implementation, not a demo.

The solver is scikit-learn's mini-batch dictionary learning, with signed unit-ball atoms, float64 computation, matching L1 penalties for fitting/encoding, batch size 128, and Lasso coordinate-descent encoding. No new solver or biological recovery guarantee is claimed. See the [report appendix](../notes/sections/05_appendices.tex) for the mathematical formulation.

`gene_complexity.dictionary_learning.fit_arrays` supports controlled normalized-training-cell initializations with a separate fixed mini-batch seed. The donor orchestrator refits this module within each inner/outer training fold. `encode` transforms new cells without changing the supplied dictionary.

For interpretation, register the candidate in an input config (or use the generated tuning config), then run:

```bash
python -m gene_complexity review-dictionary \
  --config results/workflow/new_selection/validation_config.json \
  --candidate YOUR_SELECTED_DICTIONARY --plots
```

Existing outputs are preserved. Source hashes change when code is reorganized; old candidates cannot be reused as caches for newly fingerprinted runs. `api.tune_models()` fits all candidates by default. Historical frozen models and their data hashes are not altered by this cleanup.
