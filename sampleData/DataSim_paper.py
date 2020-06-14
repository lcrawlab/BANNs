import numpy as np
import pandas as pd

def simulateData(Xfilename, H2, rho, causals, PC, bufferSz): 
	### Read in the genotype data from UKBiobank Chromosome 1:
	X=np.genfromtxt(Xfilename, delimiter=",")
	print(X.shape)
	np.save("Chr1_X.npy",X)	
	### Normalize X matrix:
	Xmean=np.mean(X, axis=0)
	Xstd=np.std(X,axis=0)
	X=np.nan_to_num((X-Xmean)/Xstd)
	### Get X dimensions:
	n=X.shape[0]
	p=X.shape[1]

	### Get annotationDF:
	if bufferSz==0:
		annotationDF= pd.read_pickle("Chr1_annotaionDF.pkl")
	elif bufferSz==50000:
		annotationDF= pd.read_pickle("Chr1_annotaionDF_50kb.pkl")

	nGenes=len(annotationDF)
	nGeneCausal=p*causals
	print(nGenes, nGeneCausal)
	#Pick causal genes:
	print(annotationDF)
	#Pick causal SNPs:

simulateData(Xfilename="Chr1_10000_X.txt", H2=0.6, rho=1, causals=0.01, PC=False, bufferSz=50000)
# causal=np.arange(0,10) #Causal SNPs are SNP #10 through SNP #35.
# ncausal=len(causal)#number of causal SNPs in total



# #### Simulate additive effects
# Xadditive=X[:, causal] #the values for causal SNPs.
# betaAdd= np.repeat(1, ncausal)#additive effect sizes initializes as "1"
# y_additive=np.dot(Xadditive, betaAdd) #initialize the value of the portion of phenotypic variation
# # caused by the additive effects as XB.
# betaAdd= betaAdd * np.sqrt(H2*rho/np.var(y_additive)) #Update additive effect sizes based on H^2, rho, and variation
# #in y_additive.
# y_additive=np.dot(Xadditive, betaAdd) #final value of y_additive, updated.

# Xepi=[]
# for i in causal:
#     for j in range(i+1, causal[-1]+1):
#         Xepi.append(np.multiply(X[:, i], X[:, j]))
# Xepi=np.column_stack(Xepi) #Matrix that holds the values for 
# # pairwise interactions between all causal SNPs.

# betaEpi= (np.array([1]*Xepi.shape[1])).reshape(-1,1) #Similar to the additive effects,
# # epistatic effect sizes are initialized as 1 and will be corrected below.
# y_epi= np.dot(Xepi, betaEpi)
# betaEpi= betaEpi* np.sqrt(H2*(1-rho)/np.var(y_epi))
# y_epi= np.dot(Xepi, betaEpi) # Final value for the portion of phenotypes explained by
# #the epistatic effects

# y_noise = np.random.normal(size=n)
# y_noise = y_noise * np.sqrt((1 - H2) / np.var(y_noise))

# y = np.add(y_additive.reshape(n, 1),y_epi.reshape(n, 1), y_noise.reshape(n, 1)) 

# np.savetxt("X_epistaticTOY.csv", X, delimiter=",")
# np.savetxt("y_epistaticTOY.csv", y, delimiter=",")