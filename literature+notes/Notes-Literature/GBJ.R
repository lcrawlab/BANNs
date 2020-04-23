library(GBJ)
library(RcppCNPy)
load("/Users/pinar/Desktop/ML-GSEA/Data/RData_Simulation/British/Br19_10k_0.6_0.005_0.01_5.RData")
simu1=all_simu[[1]]
y=simu1$pheno_result$y
null_mod <- glm(y~1, family=gaussian(link="identity"))
pvals=c()
start_time=Sys.time()
for (i in 0:1155){
  print(i)
  filename=paste("/Users/pinar/Desktop/ML-GSEA/Data/SKAT/Br10k_Set", toString(i), ".npy",sep="")
  genotype_data  = npyLoad(filename)
  # Fit the null model, calculate marginal score statistics for each SNP
  # (asymptotically equivalent to those calculated by, for example, PLINK)
  log_reg_stats <- calc_score_stats(null_model=null_mod, factor_matrix=genotype_data, link_function="linear")
  # Run the test
  a=GBJ(test_stats=log_reg_stats$test_stats, cor_mat=log_reg_stats$cor_mat)
  p=a$GBJ_pvalue
  pvals=c(pvals,p)
  print(i)
}
end_time1=Sys.time()
print((end_time1-start_time)*2)

View(pvals)
write.table(pvals,"/Users/pinar/Desktop/pvalsGBJ.txt",row.names=FALSE, col.names=FALSE)

