"""
@author: Pinar Demetci
If using gene-level inference, we need two files:
1) PLINK-formatted .map or .bim file (tab-delimited), specifying the SNP list, with the format detailed here: 
https://www.cog-genomics.org/plink2/formats#map
Note: It is important that the ordering of SNPs in the .map/.bim file is consistent with the columns of the genotype (X) matrix!

2) Gene range list file (tab-delimited), with the format of glist-hg19: https://www.cog-genomics.org/plink/1.9/resources
"""

# Packages we depend on
import pandas as pd
import numpy as np 
from tqdm import tqdm #just for progress bars 
import natsort as ns #natural sorting 
import time 
import math
##################################################################################################################################
###################################  READING IN GUIDE FILES AND CONVERTING INTO DATA FRAMES  #####################################
##################################################################################################################################

def read_SNP_file(path_SNPList, file_type='map'):
	"""
	Can take it either a ".map" file or a ".bim" file for SNPList. Needs to be specified. Default is ".map"
	Sorted by chromosome and then location for ease of search.
	"""
	SNPList=pd.DataFrame(pd.read_csv(path_SNPList, sep='\t', header=None))
	if file_type=='map':
		SNPList.columns=["Index","Chromosome","VariantID","Morgans","Position"]
	if file_type=='bim':
		SNPList.columns=["Chromosome","VariantID","Morgans","Position","Minor","Major"]
		SNPList=SNPList.drop(columns=["Minor","Major"]) #We don't need this information, so no need to store it
	SNPList.Chromosome = SNPList.Chromosome.astype('str') #For chromosomes like X, y etc, this has to be a string for ease of comparison
	SNPList.Position = SNPList.Position.astype('int32') # Locations should be read as integers for ease of comparison in annotation
	
	# The original map file has the wrong indices due to sampling in simulations:
	# SNPList=SNPList.drop(columns="Index")
	SNPList.sort_values(by=["Chromosome","Position"], ascending=[True,True], inplace=True)
	return SNPList

def read_gene_file(path_geneGuide):
	"""
	Returns a data frames object from gene range file.
	Sorted by chromosome and then location for ease of search.
	"""
	geneGuide=pd.DataFrame(pd.read_csv(path_geneGuide, sep='\t', header=None)) #Useful note: the gene names do not repeat, they are unique
	geneGuide.columns=["Chromosome","Start","End","GeneID"]
	# Fix datatypes:
	geneGuide.Chromosome = geneGuide.Chromosome.astype('str') #For chromosomes like X, y etc, this has to be a string for ease of comparison
	geneGuide.Start = geneGuide.Start.astype('int32') # Locations should be read as integers for ease of comparison in annotation
	geneGuide.End = geneGuide.End.astype('int32') # Locations should be read as integers for ease of comparison in annotation
	# Sort dataframe for ease of search:
	geneGuide['Chromosome'] = pd.Categorical(geneGuide['Chromosome'], ordered=True, categories= ns.natsorted(geneGuide['Chromosome'].unique()))
	geneGuide.sort_values(by=["Chromosome","Start","End"],ascending=[True,True,True],inplace=True)
	geneGuide= geneGuide.reset_index(drop=True) #update indices
	return geneGuide

##################################################################################################################################
################################################  CREATE ANNOTATION DATAFRAMES  ##################################################
##################################################################################################################################

def generate_intergenicDF(SNPList,geneList):
	"""
	Generate a dataframe for intergenic regions (including end and start of a chromosome upstream/downstream of genes) 
	to be merged with the gene dataframe. Used for intergenic annotations
	"""
	prevChr="-1"
	intergenicDF=pd.DataFrame(columns=["Chromosome","Start","End","GeneID"])
	maxEnd=np.amax(SNPList["Position"].tolist())

	print("creating intergenic DF")

	for index, row in tqdm(geneList.iterrows()):

		# Look up the next chromosome
		if index == len(geneList)-1:
			nextChr="inf"
		else:
			nextChr=geneList.loc[index+1, "Chromosome"]

		# Look up the current chromosome
		gChr=row["Chromosome"]
		gStart=row["Start"]
		gStop=row["End"]

		# Start to append to the intergenic DF
		if prevChr<gChr:
			if gStart>2:
				regionID="Upstream_"
				regionID+=row["GeneID"]
				intergenicDF=intergenicDF.append({"Chromosome":gChr, "Start":1, "End":gStart-1, "GeneID":regionID}, ignore_index=True )

		if gChr==nextChr:
			regionID="Intergenic_"
			regionID+=row["GeneID"]
			regionID+="_"
			regionID+=geneList.loc[index+1,"GeneID"]
			intergenicDF=intergenicDF.append({"Chromosome":gChr, "Start":gStop+1, "End":geneList.loc[index+1,"Start"]-1, "GeneID":regionID}, ignore_index=True )

		if gChr<nextChr:
			regionID="Downstream_"
			regionID+=row["GeneID"]
			intergenicDF=intergenicDF.append({"Chromosome":gChr, "Start":gStop+1, "End":maxEnd, "GeneID":regionID}, ignore_index=True )

		prevChr=gChr

	geneDF=geneList.append(intergenicDF)
	geneDF.Chromosome = geneDF.Chromosome.astype('str') #For chromosomes like X, y etc, this has to be a string for ease of comparison
	geneDF.Start = geneDF.Start.astype('int32') # Locations should be read as integers for ease of comparison in annotation
	geneDF.End = geneDF.End.astype('int32') # Locations should be read as integers for ease of comparison in annotation
	# Sort dataframe for ease of search:
	geneDF['Chromosome'] = pd.Categorical(geneDF['Chromosome'], ordered=True, categories= ns.natsorted(geneDF['Chromosome'].unique()))
	geneDF.sort_values(by=["Chromosome","Start","End"],ascending=[True,True,True],inplace=True)
	geneDF= geneDF.reset_index(drop=True) #update indices
	print("annotation length", len(geneDF))
	return geneDF


def annotateGenes(SNPList, geneList, buffer=0):
	"""
	Generates SNP-gene annotation dataframe for the first hidden layer connections.
	Groups all intergenic/intronic SNPs into one "unannotated" group.
	If want to have intergenic regions, use annotateGenesIntergenic() function.
	"""
	dfAnnotation= pd.DataFrame(geneList["GeneID"], columns=["GeneID"])
	dfAnnotation=dfAnnotation.append(pd.Series(["UnAnnotated"], index=['GeneID'], name=len(dfAnnotation)))
	dfAnnotation["SNPindex"]= [[] for _ in range(len(dfAnnotation))]

	print("annotating genes")

	for index, row in tqdm(SNPList.iterrows()):
		SNPidx=index #integer
		SNPchr=row["Chromosome"]#string
		SNPpos=row["Position"] #integer
		ind=geneList.index[((geneList["Chromosome"]==SNPchr) & (geneList["Start"]-buffer <=SNPpos) &  (geneList["End"]+buffer >=SNPpos))].tolist()
		if ind==[]:
			#This means no matching genes were found for this SNP in annotation step.
			dfAnnotation.at[len(dfAnnotation)-1, "SNPindex"].append(SNPidx)
		else:
			for i in ind:
				dfAnnotation.at[i,"SNPindex"].append(SNPidx)
	# 	# pbar.update(1)
	dfAnnotation=dfAnnotation[dfAnnotation.astype(str)["SNPindex"] != '[]']
	dfAnnotation=dfAnnotation.sort_index()
	dfAnnotation.index = range(len(dfAnnotation))
	return dfAnnotation

def annotateGenes_intergenic(SNPList, geneList, buffer=0):
	geneList=generate_intergenicDF(SNPList,geneList)
	dfAnnotation=annotateGenes(SNPList, geneList, buffer=buffer)
	return dfAnnotation

def getGeneMask(geneAnnotationDF,N,p):
	mask=np.zeros(shape=[N,p])
	print("creating mask")
	for index, row in tqdm(geneAnnotationDF.iterrows()):
		SNPidx=row["SNPindex"]
		for i in SNPidx:
			mask[i,index]=1
	return mask

def annotate(path_SNPList,path_geneGuide, SNP_filetype="map", intergenic=False, buffer=0):
	SNPList=read_SNP_file(path_SNPList, file_type=SNP_filetype)
	geneList=read_gene_file(path_geneGuide)

	if intergenic==False:
		annotationDF=annotateGenes(SNPList,geneList, buffer=buffer)
	else:
		annotationDF=annotateGenes_intergenic(SNPList,geneList,buffer=buffer)
	N=len(SNPList)
	p=len(annotationDF)
	mask=getGeneMask(annotationDF, N,p)
	return annotationDF, mask 


### Example script:
# SNPList_path="/Users/pinardemetci/Desktop/RealData/FHS.txt"
# geneList_path="/Users/pinardemetci/Desktop/BANNs/Data/glist-hg19.tsv"
# annotationDF, mask=annotate(SNPList_path,geneList_path,SNP_filetype="bim", intergenic=True, buffer=0)
# np.save("/Users/pinardemetci/Desktop/RealData/FRmask.npy", mask)
# annotationDF.to_pickle("/Users/pinardemetci/Desktop/RealData/FRannotation.pkl")






