library(ggplot2)
library(openxlsx)
library(ggrepel)
library(umap)
library(dplyr)

my_colors <- c(
  '#FE8883', '#37A2E9', '#66D0AA', '#DAACEC', '#FF81CF', '#F39B7F'
)

data_raw <- read.xlsx('data.xlsx', sheet = 1, rowNames = TRUE)
data <- data_raw
an_col <- read.xlsx('an_col.xlsx', sheet = 1)

sample_names <- colnames(data)
if (!all(c("Sample_id", "Type", "Batch") %in% colnames(an_col))) {
  stop("Required columns: Sample_id, Type, Batch")
}
an_col <- an_col[match(sample_names, an_col$Sample_id), ]

set.seed(42) 
n_samples <- ncol(data)
optimal_k <- min(5, n_samples - 1)  

custom_config <- umap.defaults
custom_config$n_neighbors <- optimal_k
custom_config$min_dist <- 0.1

umap_results <- umap(t(data), config = custom_config)

df_umap <- data.frame(
  Sample = sample_names,
  UMAP1  = umap_results$layout[, 1],
  UMAP2  = umap_results$layout[, 2],
  Type   = factor(an_col$Type),
  Batch  = factor(an_col$Batch)
)

plot_umap_custom <- function(plot_df, color_var, plot_title) {
  ggplot(plot_df, aes_string(x = "UMAP1", y = "UMAP2", color = color_var)) +
    geom_point(size = 6, alpha = 0.6) +
    geom_text_repel(
      aes(label = Sample),
      size = 4,
      max.overlaps = 100,
      segment.color = "gray50",
      segment.alpha = 0.6,
      segment.size = 0.3,
      force = 0.3,
      box.padding = 0.5
    ) +
    labs(
      x = "UMAP 1",
      y = "UMAP 2",
      title = plot_title
    ) +
    scale_color_manual(values = my_colors) +
    theme_bw(base_size = 20) +
    theme(
      panel.grid = element_blank(),
      plot.title = element_text(hjust = 0.5, size = 10),
      axis.title = element_text(),
      axis.text = element_text(size = 14),
      legend.title = element_text(),
      legend.text = element_text(size = 14),
      legend.position = "right",
      legend.key.size = unit(1.2, "cm"),
      aspect.ratio = NULL,
      plot.margin = unit(rep(1, 4), "cm")
    )
}

p_type  <- plot_umap_custom(df_umap, "Type",  "UMAP Analysis - Colored by Type")
p_batch <- plot_umap_custom(df_umap, "Batch", "UMAP Analysis - Colored by Batch")

pdf("UMAP_Result_by_Type.pdf", width = 13, height = 9)
print(p_type)
dev.off()

jpeg("UMAP_Result_by_Type.jpg", width = 13, height = 9, units = "in", res = 2000)
print(p_type)
dev.off()

pdf("UMAP_Result_by_Batch.pdf", width = 13, height = 9)
print(p_batch)
dev.off()

jpeg("UMAP_Result_by_Batch.jpg", width = 13, height = 9, units = "in", res = 2000)
print(p_batch)
dev.off()