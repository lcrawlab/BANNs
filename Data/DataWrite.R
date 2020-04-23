load("/Users/pinardemetci/Desktop/BANN-Data/Br19_10k_X.RData")
load("/Users/pinardemetci/Desktop/BANN-Data/Br19_5k_0.6_0.005_0.01_5.RData")
simu1=all_simu[[1]]

X=raw[[1]]
library(HiClimR)
R<-fastCor(X, nSplit = 7, upperTri = FALSE)
newR<-get(load("/Users/pinardemetci/Desktop/BANN-Data/Rcorr.RData"))

write.table(newR,"/Users/pinardemetci/Desktop/BANN-Data/LD.txt", sep="\t")