# 加载必要的包
library(Mfuzz)
library(dplyr)
library(ggsci)
library(scales)
library(ggvenn)
library(ggpubr)
library(scales)
library(DescTools)
library(ComplexHeatmap)
library(dendextend)
library(RColorBrewer)
library(circlize)
library(openxlsx)
library(stringr)

# 定义颜色映射
npg <- pal_npg()(10)
colmap <- c(npg[1],npg[2],"#808080")

# 设置随机种子
set.seed(1234)

# 读取和预处理数据
rawdata <- read.csv('Diffanalysis(p.adjust0.05_fc2).csv',
                    check.names = F, stringsAsFactors = F)
rownames(rawdata) <- rawdata$protein

data <- rawdata[which(rawdata$Diff == 'YES'), c('NS_mean', 'OA_mean', 'HS_mean', 'MA_mean', 'SCO_mean')]

# 聚类数量
cluster_num <- 6

# 创建 ExpressionSet 对象并标准化
mfuzz_class <- new('ExpressionSet',exprs = as.matrix(data))
mfuzz_class <- standardise(mfuzz_class)

# 进行 Mfuzz 聚类
mfuzz_cluster <- mfuzz(mfuzz_class,
                       c = cluster_num, 
                       m = mestimate(mfuzz_class))

# 绘制聚类结果图
pdf('siteglycan_ANOVA_Mfuzz.pdf', 
    height = 12, width = 8)
mfuzz.plot(mfuzz_class,mfuzz_cluster,mfrow=c(3,2),
           new.window=FALSE,min.mem=0.35,
           time.labels = c('NS', 'OA', 'HS', 'MA', 'SCO'))
dev.off()

# 将聚类结果合并到原始数据中
MC <- data.frame(Mfuzz_cluster = mfuzz_cluster[["cluster"]])
MC$protein <- rownames(MC)

rawdata$Mfuzz_cluster = 'NO'
for (i in sort(unique(MC$Mfuzz_cluster))) {
  rawdata[which(rawdata$Diff == 'YES' & 
                  rawdata$protein %in% 
                  MC[which(MC$Mfuzz_cluster == i), 'protein']), 'Mfuzz_cluster'] = paste0('Cluster', i)
}

# 查看聚类结果分布
print(table(rawdata$Mfuzz_cluster))


# 准备热图数据和注释
df <- rawdata[which(rawdata$Diff == 'YES'),]
an_col <- read.xlsx('an_col.xlsx', 1)
an_col$Type = factor(an_col$Type, levels = unique((an_col$Type)))
an_col = an_col[order(an_col$Type), ]

df$Cluster <- df$Mfuzz_cluster
df = df[order(df$Cluster),]
dfexprs <- df[,an_col$Sample_id]
dfscal = t(scale(t(dfexprs)))

# 定义列注释颜色映射
col_p <- pal_npg("nrc")(length(unique(an_col$Type)))
names(col_p) <- unique(an_col$Type)

# 创建列注释
ha_col = HeatmapAnnotation(df = data.frame(Type = an_col$Type), 
                           col = list(Type = col_p))

# 定义行注释颜色映射
row_p <- pal_nejm()(length(unique(df$Cluster)))
names(row_p) <- unique(df$Cluster)

# 创建行注释数据
an_row <- df[,c('protein', 'Cluster')]
# 创建行注释
ha_row <- rowAnnotation(df = data.frame(Cluster = an_row$Cluster), 
                        col = list(Cluster = row_p))

# 定义热图颜色映射
hmcol <- rev(colorRampPalette(brewer.pal(11,"RdBu"))(100))

# 创建热图对象
hp1 <- Heatmap(as.matrix(dfscal),
               name = "Z-score",
               border = TRUE,
               clustering_distance_rows = "euclidean",
               clustering_method_rows = "complete",
               col = colorRamp2(seq(-4,4,length.out = 100), hmcol),
               cluster_rows = F,
               show_column_dend = F,
               show_column_names = F,
               show_row_names = F,
               show_row_dend = F,
               column_order = colnames(dfscal),
               row_split = df$Cluster,
               column_split = an_col$Type,
               row_dend_gp = gpar(col = "black"),
               row_names_gp = gpar(fontsize = 3, col = "black"),
               top_annotation = ha_col,
               left_annotation = ha_row
)

# 绘制热图
ht1 <- draw(hp1)

# 保存热图为 PDF 文件
pdf(file = 'siteglycan_ANOVA_Mfuzz_heatmap.pdf',
    width = 10, height = 8)
ht1 <- draw(hp1)
dev.off()