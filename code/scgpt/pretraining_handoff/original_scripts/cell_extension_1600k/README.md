# 1.6 million cell extension
Approved 2026-09-29. Three seeds, exactly 1,600,000 cells each. Preserve each complete ordered 888k prefix, append 712k unique eligible cells using RNG 202609290+seed. Save raw float32 CSR gzip H5AD files in the existing nested_raw layout. Same Census, gene universe, model configurations, 10 epochs and fixed model seed 20260821. Nine neural runs and three PCA fits. Model indices 30..32; historical indices unchanged.

Sources copied from the successful cell_extension_888k implementation, including the H5AD axis-name repair. Historical results are linked read-only. New results/figures are isolated under cell_extension_1600k. Matrix jobs validate saved counts, IDs, order, nesting and unchanged original-file hashes before training. Full evaluation expects 528 observations (11 sizes x 3 seeds x 4 methods x 4 tasks).

Scheduling: CPU controller checks readiness once per minute and submits each seed separately when GPU QoS permits. It records every accepted GPU job ID, checks upstream failures, and submits evaluation only after all nine checkpoint/embedding success markers and three PCA success markers exist. No held evaluation job or dependency on expired job IDs is used. Matrix time limit 48h; controller 72h; GPU 12h per seed.

2026-10-01: stopped the slow remote-expression pipeline at user request. Replacement downloads immutable 2023-05-15 source H5AD files over HTTPS into data/source_h5ad_2023-05-15 (246 files, 366.6 GB total). Metadata-only Census reads establish per-dataset join ID ranges. Filtering and row order follow archived builder commit 1598cfd7d30521ee8d3622378a4f12356c485f91; every filtered row's assay/cell-type/tissue/donor/primary metadata must match. Raw expression uses raw.X when present, otherwise X, aligned to the original 19,331-gene universe with absent genes zero-filled. Each source extraction must exactly match up to 32 previously saved reference cells. Dataset-specific sampled extracts are saved and reused across all three seeds. Final assembly preserves each old 888k prefix, checks the whole saved matrix blockwise, and writes readiness markers. Old downloaded chunks are retained. Source download and extraction logs record timings. No full expression queries against Census are used.

Local HDF5 open diagnostic: default locking failed to finish within 8s; HDF5_USE_FILE_LOCKING=FALSE opened the same cached file in 0.038s. Replacement jobs set this variable before Python starts. Each output has a single writer, uses atomic rename, and downstream readers wait for success markers/dependencies. This does not establish the cause of the separate stalled remote expression read.

2026-10-02 recovery: first direct-source workers stopped together with exit 1 and no traceback after 163 validated datasets. Root cause remains unknown. Retain all validated extracts and partial source prefixes. resume_sources.sbatch assigns one dataset per task (up to eight tasks concurrently). Each file uses four bounded 16 MiB HTTP range readers with six retries; completed blocks are flushed/fsynced to a contiguous resumable prefix. Three process-level attempts per dataset; explicit exit logging. Submission includes only missing dataset indices.

Root cause confirmed 2026-10-02: writes failed with EDQUOT (disk quota exceeded). The lab /n/holylabs allocation was at 4.0 TiB of 4.0 TiB; filesystem-wide free space was misleading. Lab /n/netscratch quota was 2.4 TiB of 50 TiB. This experiment's source H5AD cache and direct extracts were moved with rsync --remove-source-files into /n/netscratch/rongma_lab/Lab/jingyuan/pretraining_cell_scaling, preserving original paths with symlinks. Completed transfer files are removed from the old location only after rsync verifies transfer; interrupted migration can resume. New 1.6m matrices are written/validated in scratch and linked into the canonical nested_raw paths. Existing matrices/checkpoints are not removed.


## Scratch analysis relocation (20261002_131959)
The 1.6m downloads, extracted counts, assembled matrices, analysis outputs,
model checkpoints, and new job logs are on /n/netscratch/rongma_lab/Lab/jingyuan/pretraining_cell_scaling.
The canonical outputs/cell_extension_1600k is a symlink to scratch; historical
comparison inputs remain absolute links to their existing results.
Scripts, manifests, and final figures remain in the lab workspace.
Slurm analysis/training jobs use scratch as working directory and TMPDIR.
GPU resources and all scientific settings are unchanged.
Scratch has a 90-day purge policy and no backup. Retain final models and results
on permanent storage before expiry. Previous output directory preserved under
.cleanup_archive/scratch_migration_20261002_131959; no original results were deleted.
