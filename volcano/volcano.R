library(ggplot2)
library(dplyr)
library(ggplot2)
library(ggsci)
library(ggpubr)
library(scales)
library(ggplot2)
library(ggrepel)
# library(webr)
npg <- pal_npg()(10)
show_col(pal_npg()(10))

colmap <- c(npg[1],npg[2],"#808080")

result <- read.csv('Diffanalysis(p.adjust0.05_fc2).csv', 
                   row.names = 1, check.names = FALSE)

result$`Log2FC(OA/HS)` <- result$OA_mean - result$HS_mean
result$`Log2FC(OA/MA)` <- result$OA_mean - result$MA_mean
result$`Log2FC(OA/SCO)` <- result$OA_mean - result$SCO_mean
result$`Log2FC(HS/MA)` <- result$HS_mean - result$MA_mean
result$`Log2FC(HS/SCO)` <- result$HS_mean - result$SCO_mean
result$`Log2FC(MA/SCO)` <- result$MA_mean - result$SCO_mean



data <- data.frame(Log2FC = result$OA_mean - result$HS_mean,
                   pvalue = result$`OA-HS`,
                   FC =result$`Log2FC(Max/Min)`,
                   qvalue = result$ANOVA_p.adjust) 


plotvo <- function(data, t) {
  fc <- 2
  pval <- 0.05
  
  data$Regulation <- "No significance"
  data[which(data$Log2FC > 0 & data$pvalue < pval &
               data$FC > 1 & data$qvalue < 0.05),'Regulation']<-"Up-regulated"
  data[which(data$Log2FC < 0 & data$pvalue < pval &
               data$FC > 1 & data$qvalue < 0.05),'Regulation']<-"Down-regulated"
  
  data$Regulation <- factor(data$Regulation,levels = c("Up-regulated","Down-regulated","No significance"))
  data$alpha <- "transparent"
  data[which(data$Regulation != "No significance"), 'alpha'] = "solid"
  
  p <- ggplot(data, aes(x=Log2FC, y=-log10(pvalue))) +
    geom_point(size=3.5, aes(color=Regulation, alpha=alpha))+
    theme_classic() + theme(panel.grid=element_blank()) +
    geom_vline(xintercept = 0,color=colmap[3],linetype="longdash")+
    geom_hline(yintercept = -log10(pval),color=colmap[3],linetype="longdash")+
    ylab("-Log10(Pvalue)")+xlab("Log2(FC)")+ 
    scale_color_manual(name=NULL, values=colmap,
                       labels=c(sprintf("Up: %d", length(which(data$Regulation=="Up-regulated"))),
                                sprintf("Down: %d", length(which(data$Regulation=="Down-regulated"))),
                                sprintf("NoDiff: %d", length(which(data$Regulation=="No significance")))))+
    scale_alpha_manual(guide="none",values=c(1,0.3))+
    # scale_x_continuous(breaks = seq(-3,3,1))+
    # ylim(0, 8)+
    # xlim(-3, 3.4)+
    geom_rug(aes(color= Regulation))+
    # theme(legend.position=c(0.2, 0.89),
    #       legend.background = element_rect(fill = alpha("gray94", 0.6),colour = "black", linetype = "solid"))+
    theme(plot.title = element_text(color="black", size=18),
          axis.title.x = element_text(color="black", size=16),
          axis.title.y = element_text(color="black", size=16))+
    theme(axis.text.x = element_text(color="black",size=16),
          axis.text.y = element_text( color="black",size=16)) +
    theme(legend.title = element_text(size= 14),
          legend.text = element_text(size=14))+
    theme(aspect.ratio = 1) +
    labs(title = t)
  # theme(legend.position = "bottom",
  #       legend.justification = "center") +
  # guides(size = FALSE, alpha = FALSE) +
  # guides(color = guide_legend(override.aes = list(size=5))) +
  # theme(legend.key.height = unit(4, "line"))
  return(p)
}

data1 <- data.frame(Log2FC = result$OA_mean - result$HS_mean,
                    pvalue = result$`OA-HS`,
                    FC =result$`Log2FC(Max/Min)`,
                    qvalue = result$ANOVA_p.adjust)
p1 <- plotvo(data1, 'OA vs. HA')
p1

data2 <- data.frame(Log2FC = result$OA_mean - result$MA_mean,
                    pvalue = result$`OA-MA`,
                    FC =result$`Log2FC(Max/Min)`,
                    qvalue = result$ANOVA_p.adjust)
p2 <- plotvo(data2, 'OA vs. MA')
p2

data3 <- data.frame(Log2FC = result$OA_mean - result$SCO_mean,
                    pvalue = result$`SCO-OA`,
                    FC =result$`Log2FC(Max/Min)`,
                    qvalue = result$ANOVA_p.adjust)
p3 <- plotvo(data3, 'OA vs. SCO')
p3

data4 <- data.frame(Log2FC = result$HS_mean - result$MA_mean,
                    pvalue = result$`MA-HS`,
                    FC =result$`Log2FC(Max/Min)`,
                    qvalue = result$ANOVA_p.adjust)
p4 <- plotvo(data4, 'HS vs. MA')
p4

data5 <- data.frame(Log2FC = result$HS_mean - result$SCO_mean,
                    pvalue = result$`SCO-HS`,
                    FC =result$`Log2FC(Max/Min)`,
                    qvalue = result$ANOVA_p.adjust)
p5 <- plotvo(data5, 'HS vs. SCO')
p5

data6 <- data.frame(Log2FC = result$MA_mean - result$SCO_mean,
                    pvalue = result$`SCO-MA`,
                    FC =result$`Log2FC(Max/Min)`,
                    qvalue = result$ANOVA_p.adjust)
p6 <- plotvo(data6, 'MA vs. SCO')
p6

library(patchwork)
p <- (p1|p2|p3)/(p4|p5|p6)





# 保存 p1
ggsave(p1, file = 'GlycanSite_volcano_OA_vs_HS.pdf', width = 11, height = 8)

# 保存 p2
ggsave(p2, file = 'GlycanSite_volcano_OA_vs_MA.pdf', width = 11, height = 8)

# 保存 p3
ggsave(p3, file = 'GlycanSite_volcano_OA_vs_SCO.pdf', width = 11, height = 8)

# 保存 p4
ggsave(p4, file = 'GlycanSite_volcano_HS_vs_MA.pdf', width = 11, height = 8)

# 保存 p5
ggsave(p5, file = 'GlycanSite_volcano_HS_vs_SCO.pdf', width = 11, height = 8)

# 保存 p6
ggsave(p6, file = 'GlycanSite_volcano_MA_vs_SCO.pdf', width = 11, height = 8)




