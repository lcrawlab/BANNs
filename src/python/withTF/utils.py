import numpy as np
import pandas as pd
import math
from annotation import *
from customModel import *

def betavar(p,mu,s):
  s=np.ones(p.shape)
  return p*(s+(1-p)*(mu**2))

def getGeneMask(geneAnnotationDF,N,p):
	mask=np.zeros(shape=[N,p])
	for index, row in geneAnnotationDF.iterrows():
		SNPidx=row["SNPindex"]
		for i in SNPidx:
			mask[i,index]=1
	return mask


def buildModel(p1,p2,mask,activation_fn):
	layers = []
	layers.append(SNP_Layer(p1, mask, activation=activation_fn))
	# layers.append(tf.keras.layers.BatchNormalization())
	layers.append(Gene_Layer(p2))
	return layers

def logits2pip(logits):
	logits=np.asarray(logits)
	exps=np.exp(logits)
	pips=(exps/(1+exps))
	return pips

def computePVE_SNP(bnn,mask):
  pip_logits=K.eval(bnn.model.layers[-2].variables[0])
  snp_betas==K.eval(bnn.model.layers[-2].variables[1])
  snp_bias==K.eval(bnn.model.layers[-2].variables[1])
  numerator= (pip_logits*mask)*snp_betas
  denominator=numerator+snp_bias
  return(numerator/denominator)

def computePVE_Genes(bnn):
  pip_logits=K.eval(bnn.model.layers[-1].variables[0])
  gene_betas==K.eval(bnn.model.layers[-2].variables[1])
  gene_bias==K.eval(bnn.model.layers[-2].variables[1])
  numerator= pip_logits*gene_betas
  denominator=numerator+gene_bias
  return(numerator/denominator)

def modelLoss(Xr, d, y, sigma, prob, mu, s, sa, logodds):
  pi=math.pi
  Xr=np.ones(y.shape)
  linearLoss=(-len(y)/2*np.log(2* pi *sigma) -np.linalg.norm(y - Xr)**2/(2*sigma)-np.dot(d,betavar(prob,mu,s))/(2*sigma))
  kleffect=((np.sum(prob) + 
  np.dot(prob,np.log(s/sa)) - 
  np.dot(prob, mu**2)/sa)/2 -
  np.dot(prob,np.log(prob)) - 
  np.dot(1 - prob,np.log(1 - prob)))


##################################################################################################################################
################################################  GET NEW INDICES FOR CAUSALS  ###################################################
##################################################### (FROM SIMULATIONS) #########################################################
def getCausalSNPidx(causals, mapfile):
  """
  Can take it either a ".map" file or a ".bim" file for SNPList. Needs to be specified. Default is ".map"
  Sorted by chromosome and then location for ease of search.
  """
  causalSNPs=[]
  SNPList=pd.DataFrame(pd.read_csv(mapfile, sep='\t', header=None))
  SNPList.columns=["OldIndex","Chromosome","VariantID","Morgans","Position"]
  SNPList.Chromosome = SNPList.Chromosome.astype('str') #For chromosomes like X, y etc, this has to be a string for ease of comparison
  SNPList.Position = SNPList.Position.astype('int32') # Locations should be read as integers for ease of comparison in annotation
  SNPList.sort_values(by=["OldIndex"], ascending=[True], inplace=True)
  SNPList.sort_values(by=["Chromosome","Position"], ascending=[True,True], inplace=True)
  for i in causals:
    idx=SNPList.index[SNPList['OldIndex'] == i].tolist()
    for  j in idx:
      causalSNPs.append(j)
  print(SNPList)
  return causalSNPs

def getCausalGenes(geneDF, causalSNPs):
  causalGenes=[]
  for index,row in geneDF.iterrows():
    geneID=row["GeneID"]
    geneindx=index
    snps=row["SNPindex"]
    causals=[i for i in snps if i in causalSNPs]
    if len(causals)>0:
      causalGenes.append(geneindx)
  return causalGenes

def getCausalPathways(pathwayDF, causalGenes):
  causalPathway=[]
  for index,row in geneDF.iterrows():
    pathwayID=row["Pathway"]
    geneIndx=row["GeneIndex"]
    pathIndx=index
    causals=[i for i in geneIndx if i in causalGenes]
    if len(causals)>0:
      causalPathway.append(pathIndx)
  return causalPathway
