library(SKAT)
library(RcppCNPy)
install.packages("doParallel")
library(doParallel)

load("/Users/pinar/Desktop/ML-GSEA/Data/RData_Simulation/British/Br19_10k_0.6_0.005_0.01_5.RData")
simu=all_simu[[1]]
y=simu$pheno_result$y
causal_snps=simu$assign_result$all_snps$causal_snps
print(causal_snps)
pvals=c()
for (i in 0:1155){
  print(i)
  filename=paste("/Users/pinar/Desktop/ML-GSEA/Data/SKAT/Br10k_Set", toString(i), ".npy",sep="")
  Z=npyLoad(filename)
  obj<- SKAT_Null_Model(y~1, out_type="C")
  pv=SKAT(Z,obj)$p.value
  pvals=c(pvals,pv)
}
View(pvals)
pvals<-as.data.frame(pvals)
View(pvals)


