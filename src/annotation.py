"""
@author: Pinar Demetci
If using gene-level inference, we need two files:
1) PLINK-formatted .map or .bim file (tab-delimited), specifying the SNP list, with the format detailed here: 
https://www.cog-genomics.org/plink2/formats#map
Note: It is important that the ordering of SNPs in the .map/.bim file is consistent with the columns of the genotype (X) matrix!

2) Gene range list file (tab-delimited), with the format of glist-hg19: https://www.cog-genomics.org/plink/1.9/resources

If using chromosome-level inference, we need a third file in addition:
3) Pathway List: The one we use is KEGG gene set from MSigDB. The format is as follows: Pathway/Gene set Name | Link | Gene names included in the pathway/gene set

"""

# Packages we depend on
import pandas as pd
import numpy as np 
from tqdm import tqdm #just for progress bars 
import natsort as ns #natural sorting 

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
		SNPList.columns=["Index","Chromosome","VariantID","Morgans","Position","Minor","Major"]
		SNPList=SNPList.drop(columns=["Minor","Major"])
	SNPList.Chromosome = SNPList.Chromosome.astype('str') #For chromosomes like X, y etc, this has to be a string for ease of comparison
	SNPList.Position = SNPList.Position.astype('int32') # Locations should be read as integers for ease of comparison in annotation
	
	# The original map file has the wrong indices due to sampling in simulations:
	SNPList=SNPList.drop(columns="Index")
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

def read_pathway_Files(path_pathwayGuide):
	pathwayGuide=pd.DataFrame(pd.read_csv(path_pathwayGuide, sep='\t', header=None, error_bad_lines=False)) #Useful note: the gene names do not repeat, they are unique
	geneSetList=pathwayGuide.iloc[:,2:len(pathwayGuide.columns)].values.tolist()
	geneSetList = [[x for x in g if str(x)!='nan'] for g in geneSetList]
	pathwayGuide.insert(1,"Genes", geneSetList)
	pathwayGuide=pathwayGuide.iloc[:,0:2]
	pathwayGuide.columns=["Pathway","Genes"]
	return pathwayGuide

##################################################################################################################################
################################################  CREATE ANNOTATION DATAFRAMES  ##################################################
##################################################################################################################################

def annotateGenes(SNPList, geneList, buffer=50000):
	"""
	Generates SNP-gene annotation dataframe for the first hidden layer connections.
	Groups all intergenic/intronic SNPs into one "unannotated" group.
	If want to have intergenic regions, use annotateGenesIntergenic() function.
	"""
	dfAnnotation= pd.DataFrame(geneList["GeneID"], columns=["GeneID"])
	dfAnnotation=dfAnnotation.append(pd.Series(["UnAnnotated"], index=['GeneID'], name=len(dfAnnotation)))
	dfAnnotation["SNPindex"]= [[] for _ in range(len(dfAnnotation))]

	for index, row in SNPList.iterrows():
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
	
def annotatePathways(df_geneAnnotation, pathwayList):
	dfAnnotation=pd.DataFrame(pathwayList["Pathway"], columns=["Pathway"])
	dfAnnotation=dfAnnotation.append(pd.Series(["UnAnnotated"],index=["Pathway"], name=len(dfAnnotation)))
	dfAnnotation["GeneIndex"]= [[] for _ in range(len(dfAnnotation))]

	print("Annotating gene-pathway mapping")
	# pbar = tqdm(total = len(df_geneAnnotation))
	for index,row in df_geneAnnotation.iterrows():
		geneID=row["GeneID"]
		pathwayIdx=pathwayList.index[pd.DataFrame(pathwayList.Genes.tolist()).isin([geneID]).any(1)].tolist()
		if pathwayIdx==[]:
			dfAnnotation.at[len(dfAnnotation)-1, "GeneIndex"].append(index)
		else:
			for i in pathwayIdx:
				dfAnnotation.at[i,"GeneIndex"].append(index)
		# pbar.update(1)

	dfAnnotation=dfAnnotation[dfAnnotation.astype(str)["GeneIndex"] != '[]']
	dfAnnotation=dfAnnotation.sort_index()
	dfAnnotation.index = range(len(dfAnnotation))
	return dfAnnotation
