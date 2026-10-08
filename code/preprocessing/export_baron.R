library(SingleCellExperiment)
library(Matrix)

# Load original Baron human pancreas dataset
sce <- readRDS("data/raw/baron_human_counts.rds")
counts <- assay(sce, "counts")

# Export sparse matrix (genes x cells)
writeMM(counts, "data/processed/baron_counts.mtx")

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
