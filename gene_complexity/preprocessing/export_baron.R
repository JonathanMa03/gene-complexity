# Run from the repository root. First use requires internet access and a
# writable R package library; subsequent runs reuse packages and the RDS cache.
required_packages <- c("scRNAseq", "SingleCellExperiment", "Matrix")
missing_packages <- required_packages[!vapply(
  required_packages, requireNamespace, logical(1), quietly = TRUE
)]
if (length(missing_packages) > 0L) {
  if (!requireNamespace("BiocManager", quietly = TRUE)) {
    install.packages("BiocManager", repos = "https://cloud.r-project.org")
  }
  # Some macOS Rhdf5lib builds link against OpenSSL without supplying its
  # Homebrew library path. Make an existing installation visible to clang.
  previous_library_path <- Sys.getenv("LIBRARY_PATH", unset = NA_character_)
  if (Sys.info()[["sysname"]] == "Darwin") {
    openssl_candidates <- c(
      "/opt/homebrew/opt/openssl@3/lib", "/usr/local/opt/openssl@3/lib"
    )
    openssl_found <- openssl_candidates[
      file.exists(file.path(openssl_candidates, "libssl.dylib")) &
      file.exists(file.path(openssl_candidates, "libcrypto.dylib"))
        ]
    if (length(openssl_found) > 0L) {
      paths <- c(openssl_found[[1]], previous_library_path)
      paths <- paths[!is.na(paths) & nzchar(paths)]
      message("Using OpenSSL libraries for package installation: ", openssl_found[[1]])
      Sys.setenv(LIBRARY_PATH = paste(paths, collapse = .Platform$path.sep))
    }
  }
  tryCatch(
    BiocManager::install(missing_packages, ask = FALSE, update = FALSE),
    finally = {
      if (is.na(previous_library_path)) {
        Sys.unsetenv("LIBRARY_PATH")
      } else {
        Sys.setenv(LIBRARY_PATH = previous_library_path)
      }
    }
  )
}
for (package in required_packages) {
  if (!requireNamespace(package, quietly = TRUE)) {
    stop("Required package could not be installed: ", package)
  }
}

dir.create("data/raw", recursive = TRUE, showWarnings = FALSE)
dir.create("data/processed", recursive = TRUE, showWarnings = FALSE)

# Download once, preserving the original object for repeatable exports.
cache_path <- "data/raw/baron_human_counts.rds"
if (!file.exists(cache_path)) {
  message("Downloading Baron human pancreas counts...")
  sce <- scRNAseq::BaronPancreasData(which = "human")
  saveRDS(sce, cache_path)
} else {
  message("Using cached Baron data: ", cache_path)
  sce <- readRDS(cache_path)
}
counts <- SummarizedExperiment::assay(sce, "counts")

# Export sparse matrix (genes x cells)
Matrix::writeMM(counts, "data/processed/baron_counts.mtx")

# Export gene symbols
write.csv(
  data.frame(gene_symbol = rownames(sce)),
  "data/processed/baron_genes.csv",
  row.names = FALSE
)

# Export cell metadata
write.csv(
  data.frame(
    cell_id = colnames(sce),
    donor = as.character(sce$donor),
    cell_type = as.character(sce$label)
  ),
  "data/processed/baron_cells.csv",
  row.names = FALSE
)

cat("Export complete!\n")
cat("Genes:", nrow(sce), "\n")
cat("Cells:", ncol(sce), "\n")
