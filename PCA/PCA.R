library(ggplot2)
library(openxlsx)
library(ggrepel)
library(RColorBrewer)

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

pca <- prcomp(t(data), scale. = TRUE, center = TRUE)
df <- data.frame(
  Sample = sample_names,
  PC1 = pca$x[,1],
  PC2 = pca$x[,2],
  Type = factor(an_col$Type),
  Batch = factor(an_col$Batch)
)
PC_var <- round(pca$sdev^2 / sum(pca$sdev^2) * 100, 2)

plot_pca_custom <- function(plot_df, color_var, plot_title) {
  ggplot(plot_df, aes_string(x = "PC1", y = "PC2", color = color_var)) +
    geom_point(size = 4, alpha = 0.9) +
    geom_text_repel(
      aes(label = Sample),
      size = 4,
      max.overlaps = 100,
      segment.color = "gray50",
      segment.alpha = 0.6,
      force = 3,
      box.padding = 0.8
    ) +
    labs(
      x = paste0("PC1 (", PC_var[1], "%)"),
      y = paste0("PC2 (", PC_var[2], "%)"),
      title = plot_title
    ) +
    scale_color_manual(values = my_colors) +
    theme_bw(base_size = 10) +
    theme(
      panel.grid = element_blank(),
      plot.title = element_text(hjust = 0.5, size = 10),
      axis.title = element_text(),
      axis.text = element_text(size = 8),
      legend.title = element_text(),
      legend.text = element_text(size = 8),
      legend.position = "right",
      legend.key.size = unit(1.2, "cm"),
      aspect.ratio = NULL,
      plot.margin = unit(rep(2, 4), "cm")
    )
}

p_type  <- plot_pca_custom(df, "Type",  "PCA Analysis - Colored by Type")
p_batch <- plot_pca_custom(df, "Batch", "PCA Analysis - Colored by Batch")

pdf("PCA_Result_by_Type.pdf", width = 8, height = 8)
print(p_type)
dev.off()

jpeg("PCA_Result_by_Type.jpg", width = 8, height = 8, units = "in", res = 2000)
print(p_type)
dev.off()

pdf("PCA_Result_by_Batch.pdf", width = 8, height = 8)
print(p_batch)
dev.off()

jpeg("PCA_Result_by_Batch.jpg", width = 8, height = 8, units = "in", res = 2000)
print(p_batch)
dev.off()