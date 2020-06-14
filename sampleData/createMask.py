import pandas as pd
import numpy as np 
# from tqdm import tqdm #just for progress bars 
import natsort as ns #natural sorting 



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

def annotateGenes(SNPList, geneList, bufferSz=50000):
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
		ind=geneList.index[((geneList["Chromosome"]==SNPchr) & (geneList["Start"]-bufferSz <=SNPpos) &  (geneList["End"]+bufferSz >=SNPpos))].tolist()
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

def getGeneMask(geneAnnotationDF,N,p):
	mask=np.zeros(shape=[N,p])
	for index, row in geneAnnotationDF.iterrows():
		SNPidx=row["SNPindex"]
		for i in SNPidx:
			mask[i,index]=1
	return mask

path_SNPList="/Users/pinardemetci/Desktop/BANNs/Data/Chr1_10000_map.txt"
path_geneGuide="/Users/pinardemetci/Desktop/BANNs/Data/glist-hg19.tsv"
SNPdf=read_SNP_file(path_SNPList)
geneDF=read_gene_file(path_geneGuide)
print(SNPdf)
print(geneDF)
annotationDF=annotateGenes(SNPdf, geneDF, bufferSz=50000)
print(annotationDF)
annotationDF.to_pickle("/Users/pinardemetci/Desktop/BANNs/Data/Chr1_annotaionDF_5kb.pkl")


