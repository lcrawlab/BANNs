import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

def pick_causals(DF, percent):
	if percent==1:
		nc=30
		low=0.9
		high=1.1
	elif percent==10:
		nc=300
		low=9
		high=11
	i=0
	while ((low<=i)&(i<=high))==False:
		SNPs=[]
		causal_genes=[]
		genes=DF.sample(n=nc)
		for index,row in genes.iterrows():
			SNP=row["SNPindex"]
			for s in SNP:
				SNPs.append(s)
			causal_genes.append(index)
		i=len(SNPs)/36348*100

	return causal_genes, SNPs

simulationDF=pd.read_pickle("/Users/pinardemetci/Desktop/BANNs/src/python/IntergenicDF_0kb.pkl")
print(len(simulationDF))
# X=np.load("/Users/pinardemetci/Desktop/BANNs/src/python/Simulation_X.npy")
# n=X.shape[0]
# p=X.shape[1]

# pca=PCA(n_components=10)
# PCs= pca.fit_transform(X)
# betaPC=np.repeat(1,10)
# y_PC=np.dot(PCs, betaPC)
# betaPC=betaPC*np.sqrt(0.1/np.var(y_PC))
# y_PC=np.dot(PCs, betaPC)
# y_PC=y_PC.reshape(n,1)

# X= ((np.random.uniform(size=[n,p])>maf)&(np.random.uniform(size=[n,p])>maf))*1.0
# This is basically a way to simulate SNP data. If the "random allele 
# frequencies" we generate are larger than the maf both times, we say this
# is the version of the locus that is most commonly observed in the population,
# and gets the value of 1.0. If not, then it is a variation from the common nucleotide 
# for that locus, and gets the value of 0.0
# Xmean= np.mean(X, axis=0) #mean of each column, which corresponds to a SNP locus
# Xstd= np.std(X,axis=0) #standard deviation of each column
# X=np.nan_to_num((X-Xmean)/Xstd) # Standardized genotype matrix based on means and standard deviations
# of each SNP locus. This is the final X. I use np.nan_to_num in case we get NaN values due to 
# 0 division, i.e. if everything in a column is 0, the mean and standard deviation will also be 0, 
# then we will get NaN values. np.nan_to_num turns NaNs into "0"s. This seemed like a good idea to me
# H2= 0.6 #80% of phenotypic variation is explained by genotype
# rho= 0.5 #75% of H^2 is explained by additive effects
# percent=1

# for sim in range(0,100):
# 	print(sim)
# 	causal_genes, causal=pick_causals(simulationDF, percent)
# 	ncausal=len(causal)#number of causal SNPs in total

# 	Xadditive=X[:, causal] #the values for causal SNPs.
# 	betaAdd= np.repeat(1, ncausal)#additive effect sizes initializes as "1"
# 	y_additive=np.dot(Xadditive, betaAdd) #initialize the value of the portion of phenotypic variation
# 	# caused by the additive effects as XB.
# 	betaAdd= betaAdd * np.sqrt(H2*rho/np.var(y_additive)) #Update additive effect sizes based on H^2, rho, and variation
# 	#in y_additive.
# 	y_additive=np.dot(Xadditive, betaAdd) #final value of y_additive, updated.
# 	y_additive=y_additive.reshape(n,1)

# 	Xepi=[]
# 	for i in causal:
# 	    for j in range(i+1, causal[-1]+1):
# 	        Xepi.append(np.multiply(X[:, i], X[:, j]))
# 	Xepi=np.column_stack(Xepi) #Matrix that holds the values for 
# 	# pairwise interactions between all causal SNPs.

# 	betaEpi= (np.array([1]*Xepi.shape[1])).reshape(-1,1) #Similar to the additive effects,
# 	# epistatic effect sizes are initialized as 1 and will be corrected below.
# 	y_epi= np.dot(Xepi, betaEpi)
# 	betaEpi= betaEpi* np.sqrt(H2*(1-rho)/np.var(y_epi))
# 	y_epi= np.dot(Xepi, betaEpi) # Final value for the portion of phenotypes explained by
# 	y_epi=y_epi.reshape(n,1)

# 	y_noise = np.random.normal(size=n)
# 	y_noise = y_noise * np.sqrt((1 - H2-0.1) / np.var(y_noise))
# 	y_noise=y_noise.reshape(n,1)

# 	y = y_additive + y_noise + y_PC +y_epi #np.add(y_additive.reshape(n, 1), y_noise.reshape(n, 1), y_PC, y_epi.reshape(n, 1))
# 	#y = y_additive + y_noise + y_PC + y_epi #np.add(y_additive.reshape(n, 1), y_noise.reshape(n, 1), y_PC, y_epi.reshape(n, 1))
# 	# np.savetxt("timing/X_1000_1000.csv", X, delimiter=",")

# 	y_filename="Simu/06H05r1pPC/y_"+str(sim)+".txt"
# 	snp_filename="Simu/06H05r1pPC/snp_"+str(sim)+".csv"
# 	gene_filename="Simu/06H05r1pPC/gene_"+str(sim)+".csv"
# 	np.savetxt(y_filename, y, delimiter=",")
# 	np.savetxt(snp_filename, np.asarray(causal), delimiter=",")
# 	np.savetxt(gene_filename, np.asarray(causal_genes), delimiter=",")













