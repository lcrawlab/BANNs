rm(list = ls())
library(BANN)

#read in example data
X = as.matrix(read.table("example_data/Xtest.txt"))
y = read.table("example_data/ytest.txt")
y = y[,1]
mask = as.matrix(read.table("example_data/masktest.txt"))
#run BANN
res = BANN(X, mask ,y, centered=FALSE, show_progress = TRUE)
#Access posterior inclusion probability for SNP layer
#res$SNP_level$pip
#Access posterior inclusion probability for Gene layer
#res$SNPset_level$pip
#Access PVE estimate for SNP layer
#res$SNP_level$model.pve
#Access PVE estimate for Gene layer
#res$SNPset_level$model.pve
