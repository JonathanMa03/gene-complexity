# Pretraining extension: 444k and 888k cells

Canonical lab implementation approved on 2026-09-26. Three sampling seeds (0,1,2), six new raw matrices, 18 neural training runs, six PCA fits. Existing cell IDs/counts and artifacts are preserved.

Run `bash submit.sh` from this directory. Dependencies: manifests -> matrix array -> PCA/GPU arrays -> projection, held-out evaluation, fits, plots. Submission receipt prevents accidental duplicate submission. Inspect failed jobs before explicit resubmission.

Original scripts are snapshotted here with hashes in source_hashes.json. Original indices 0..23 are unchanged; extension indices 24..29 map to (seed0,444k), (seed0,888k), (seed1,444k), (seed1,888k), (seed2,444k), (seed2,888k).

Matrices: data/nested_raw/seed_{seed}/sctab_seed{seed}_n{size}_raw.h5ad. Gzip, CSR float32 raw counts, original 19,331-gene order. Original 222k prefix retained. Append 666k unique cells sampled without replacement from the same eligible universe, excluding the old prefix. RNG seed is 202609260 + sampling_seed; manifests record it. Downloads resume from 50k-cell chunks. Validation checks persisted counts, nesting, uniqueness, gene order and original-file checksum before writing success markers.

Training: from scratch, 10 epochs, fixed original 2,000-gene panel, unchanged objectives/architecture/batch size/learning rates. Fixed model seed 20260821 for all models. Historical scVI logs explicitly confirm this for all 24 models; scGPT code and saved metadata agree. Geneformer historical metadata records this seed, but original source history is unavailable; the saved metadata is used as provenance. Current shared scripts had since changed scVI/Geneformer to sampling-seed initialization, so these isolated copies restore the recorded pretraining seed. No fine-tuning checkpoint or environment overrides are inherited.

New results and figures are isolated under outputs/cell_extension_888k and figures/cell_extension_888k. Historical PCA bases, representations and evaluation checkpoints are linked for read-only reuse. Evaluation retains the original DOGMA split (16,524 train / 5,000 held-out), 10 downstream subsamples, ranks and analysis seed. Expanded results contain 480 method/task/size/seed observations. Plot style is unchanged, with cell-count axis extended to 888k.

No raw matrices, environments or model checkpoints are synchronized locally. These runs use fixed epochs, so compute increases with sample size. 888k is a new measurement point, not a predicted saturation threshold.

Cluster scheduling: GPU memory is 64 GB per GPU. Eighteen training runs execute in two balanced sequential waves (nine per array task) to respect the user GPU submission limit. Each model still has its own checkpoint and embedding success marker.

## First-run failure and repair
The initial manifests job 48772184 completed. Matrix array 48772188 failed for all three seeds while serializing the first downloaded chunk: assigning the soma_joinid Series to obs_names also assigned its name, conflicting with the numeric soma_joinid column. No larger matrix or trained model was completed. Downstream jobs never started and ended CANCELLED. The repair clears obs_names.name and var_names.name before saving chunks and assembled matrices. test_h5ad_serialization.py reproduces the original exception and verifies chunk and assembled-matrix save/reload against real metadata and raw counts for every seed. Original manifests and original matrices are reused unchanged on retry.

2026-09-27 scheduling update: train_by_seed.sbatch runs all three models and both sizes for one seed. Seeds 0/1 released immediately as job 48947750; validated PCA tasks 0..3 also released. Original GPU job 48814661 cancelled while pending. GPU QoS allows two submissions, so dispatcher 48947812 waits for ready training, matrices and PCA to succeed, submits seed 2, sets final evaluation dependency to that new GPU job, then releases held evaluation 48814662. Receipt: manifests/cell_extension_888k/submitted_jobs.tsv; seed-2 job ID will be saved in seed2_training_job.txt.
