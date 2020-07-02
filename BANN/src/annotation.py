"""
This program is free software: you can redistribute it under the
terms of the GNU General Public License; either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANY; without even the implied warranty of
MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

@author: Pinar Demetci

Script for generating annotation masks used for BANNs model.

If using gene-level inference, we need two files:
1) PLINK-formatted .map or .bim file (tab-delimited), specifying the SNP list, with the format detailed here: 
https://www.cog-genomics.org/plink2/formats#map
https://www.cog-genomics.org/plink2/formats#bim

2) Gene range list file (tab-delimited), e.g. with the format of glist-hg19 or glist-hg38: https://www.cog-genomics.org/plink/1.9/resources

Note1: It is important that the ordering of SNPs in the .map/.bim file is consistent with the columns of the genotype (X) matrix,
as in, the order of SNPs should match between SNP List and genotype matrix X. 

Note2: The annotation matrix that this script outputs will be of size p by g, where p corresponds to the number of SNPs and g corresponds to the number of SNP-sets.
The order of SNPs in the annotation mask will be the same as the order in SNP List and the order of SNPsets will be based on chromosomal location
"""

# Packages we depend on
import pandas as pd
import numpy as np 
from tqdm import tqdm #just for progress bars 
import natsort as ns #natural sorting
import sys
import time 
import math

##################################################################################################################################
###################################  READING IN GUIDE FILES AND CONVERTING INTO DATA FRAMES  #####################################
##################################################################################################################################

def read_SNP_file(path_SNPList):
	"""
	Helper function for annotate() and getMaskMatrix()
	--------------------------------------------------
	Takes in
	path_SNPList: path to the SNP list file. SNP List file needs to be in either a ".map"  or a ".bim" format.
	--------------------------------------------------
	Returns
	SNPList: a pandas dataframe object containing SNP information.
	--------------------------------------------------
	Notes:
	If there are 4 columns in the SNP list, we assume this is .map format based on the file format descriptions above
	If there are 6 columns, we assume this is .bim format based o nthe file format descriptions above.
	If there are a different number of columns, we warn the user about the file format and quit annotation procedure.
	SNPList dataframe is sorted by chromosome and then location for ease of search, but keeping the original indices for order match with the genotype data.
	"""
	SNPList=pd.DataFrame(pd.read_csv(path_SNPList, sep='\t', header=None))
	if len(SNPList.columns)==4: #The input is a .map file
		SNPList.columns=["Chromosome","VariantID","Morgans","Position"]
	elif len(SNPList.columns)==6: #The input is a .bim file
		SNPList.columns=["Chromosome","VariantID","Morgans","Position","Minor","Major"]
		SNPList=SNPList.drop(columns=["Minor","Major"]) #We don't need this information, so no need to store it
	else:
		sys.exit("Wrong file format for SNP List file. Please make sure to provide a tab-delimited .map or .bim file with no headers. \n We expect .map file to have 4 fields: Chromosome, Variant ID, Morgans, Position. https://www.cog-genomics.org/plink2/formats#map \n We expect .bim file to have 6 fields: Chromosome, Variant ID, Morgans, Position, Minor, Major. For more information, visit https://www.cog-genomics.org/plink2/formats#bim ")
	
	SNPList.Chromosome = SNPList.Chromosome.astype('str') #For chromosomes like X, y etc, this has to be a string for ease of comparison
	SNPList.Position = SNPList.Position.astype('int32') # Locations should be read as integers for ease of comparison in annotation
	
	# We will sort the dataframe based on first chromosome and then position for ease of search when annotating 
	# BUT we will not update the indices to make sure they match up with the input genotype matrix.
	# We use natsort to make sure in sorting pandas understands 2<10 (without natural sorting, it thinks 10<2)
	# For that, we will copy Chromosome and Position fields as categorical variables and sort based on them, then drop these columns
	# natsort requires turning data into Categorical data:
	SNPList['Chromosome']= pd.Categorical(SNPList['Chromosome'], ordered=True, categories= ns.natsorted(SNPList['Chromosome'].unique()))
	SNPList['Position'] = pd.Categorical(SNPList['Position'], ordered=True, categories= ns.natsorted(SNPList['Position'].unique()))
	SNPList = SNPList.sort_values(['Chromosome','Position'])

	# Fix datatypes from categorical:
	SNPList.Chromosome = SNPList.Chromosome.astype('str') #For chromosomes like X, y etc, this has to be a string for ease of comparison
	SNPList.Position = SNPList.Position.astype('int32') # Locations should be read as integers for ease of comparison in annotation
	return SNPList

def read_gene_file(path_geneGuide):
	"""
	Helper function for annotate()
	--------------------------------------------------
	Takes in
	path_geneGuide: path to gene range list file
	--------------------------------------------------
	Returns
	geneGuide: a data frames object containing gene range information.
	--------------------------------------------------
	Notes:
	geneGuide dataframe is sorted by chromosome and then location for ease of search.
	"""
	geneGuide=pd.DataFrame(pd.read_csv(path_geneGuide, sep='\t', header=None)) #Useful note: the gene names do not repeat, they are unique
	if len(geneGuide.columns)==4:
		geneGuide.columns=["Chromosome","Start","End","GeneID"]
	else:
		sys.exit("Wrong file format for gene range file. Please make sure to provide a tab-delimited file with no headers and with 4 fields: Chromosome, Start, End, GeneID. \n You can check the following link for more information: https://www.cog-genomics.org/plink/1.9/resources")
	# Fix datatypes:
	geneGuide.Chromosome = geneGuide.Chromosome.astype('str') #For chromosomes like X, y etc, this has to be a string for ease of comparison
	geneGuide.Start = geneGuide.Start.astype('int32') # Locations should be read as integers for ease of comparison in annotation
	geneGuide.End = geneGuide.End.astype('int32') # Locations should be read as integers for ease of comparison in annotation
	
	# Sort dataframe for ease of search. Using natsort to make sure pandas understands 2<10 (without natural sorting, it thinks 10<2):
	# natsort requires turning data into Categorical data:
	geneGuide['Chromosome'] = pd.Categorical(geneGuide['Chromosome'], ordered=True, categories= ns.natsorted(geneGuide['Chromosome'].unique()))
	geneGuide['Start'] = pd.Categorical(geneGuide['Start'], ordered=True, categories= ns.natsorted(geneGuide['Start'].unique()))
	geneGuide['End'] = pd.Categorical(geneGuide['End'], ordered=True, categories= ns.natsorted(geneGuide['End'].unique()))
	geneGuide.sort_values(by=["Chromosome","Start","End"],ascending=[True,True,True],inplace=True)
	geneGuide= geneGuide.reset_index(drop=True) #update indices

	# Fix datatypes from categorical:
	geneGuide.Chromosome = geneGuide.Chromosome.astype('str') #For chromosomes like X, y etc, this has to be a string for ease of comparison
	geneGuide.Start = geneGuide.Start.astype('int32') # Locations should be read as integers for ease of comparison in annotation
	geneGuide.End = geneGuide.End.astype('int32') # Locations should be read as integers for ease of comparison in annotation
	
	return geneGuide

##################################################################################################################################
################################################  CREATE ANNOTATION DATAFRAMES  ##################################################
##################################################################################################################################

def generate_intergenicDF(SNPList,geneList):
	"""
	Helper function for annotate(). Generates a dataframe for intergenic regions (including end and start of a chromosome upstream/downstream of genes) 
	to be merged with the gene dataframe. Used for intergenic annotations
	--------------------------------------------------
	Takes in
	SNPList: pandas dataframe object containing SNP list information
	geneList: pandas dataframe object containing gene range information
	--------------------------------------------------
	Returns
	geneDF: pandas dataframe object containing chromosomal location information on genes and intergenic SNP-sets
	"""
	prevChr="-1"
	intergenicDF=pd.DataFrame(columns=["Chromosome","Start","End","GeneID"])
	maxEnd=np.amax(SNPList["Position"].tolist())

	print("Creating Intergenic SNP-sets")
	with tqdm(total=len(geneList)) as pbar:
		for index, row in geneList.iterrows():
			pbar.update(1)
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
	geneDF.Chromosome = geneDF.Chromosome.astype('str') #For chromosomes like X, Y etc, this has to be a string for ease of comparison
	geneDF.Start = geneDF.Start.astype('int32') # Locations should be read as integers for ease of comparison in annotation
	geneDF.End = geneDF.End.astype('int32') # Locations should be read as integers for ease of comparison in annotation

	# Sort dataframe for ease of search. Using natsort to make sure pandas understands 2<10 (without natural sorting, it thinks 10<2):
	# natsort requires turning data into Categorical data:
	geneDF['Chromosome'] = pd.Categorical(geneDF['Chromosome'], ordered=True, categories= ns.natsorted(geneDF['Chromosome'].unique()))
	geneDF['Start'] = pd.Categorical(geneDF['Start'], ordered=True, categories= ns.natsorted(geneDF['Start'].unique()))
	geneDF['End'] = pd.Categorical(geneDF['End'], ordered=True, categories= ns.natsorted(geneDF['End'].unique()))
	geneDF.sort_values(by=["Chromosome","Start","End"],ascending=[True,True,True],inplace=True)
	geneDF= geneDF.reset_index(drop=True) #update indices

	# Fix datatypes again:
	geneDF.Chromosome = geneDF.Chromosome.astype('str') #For chromosomes like X, Y etc, this has to be a string for ease of comparison
	geneDF.Start = geneDF.Start.astype('int32') # Locations should be read as integers for ease of comparison in annotation
	geneDF.End = geneDF.End.astype('int32') # Locations should be read as integers for ease of comparison in annotation

	return geneDF

def annotateSets(SNPList, geneList, buffer=0):
	"""
	Helper function for annotate()
	Carries out SNP - to - SNP-set annotation for the first hidden connections in BANNs model.
	Groups all intergenic/intronic SNPs into one "Unannotated" group.
	--------------------------------------------------
	Takes in
	SNPList: pandas dataframe object containing SNP list information
	geneList: pandas dataframe object containing gene range information
	--------------------------------------------------
	Returns
	dfAnnotation: pandas dataframe object containing SNP - to - SNP-set annotation information
	"""
	dfAnnotation= geneList
	geneIDs=dfAnnotation["GeneID"]
	dfAnnotation.drop(labels=['GeneID'], axis=1,inplace = True)
	dfAnnotation.insert(0, 'GeneID', geneIDs)
	dfAnnotation=dfAnnotation.append(pd.Series(["UnAnnotated"], index=['GeneID'], name=len(dfAnnotation)))
	dfAnnotation["SNPindex"]= [[] for _ in range(len(dfAnnotation))]
	dfAnnotation["VariantID"]= [[] for _ in range(len(dfAnnotation))]

	print("Annotating SNP-sets with the corresponding SNPs")
	with tqdm(total=len(SNPList)) as pbar:
		for index, row in SNPList.iterrows():
			SNPidx=index #integer
			VariantID=row["VariantID"]
			SNPchr=row["Chromosome"]#string
			SNPpos=row["Position"] #integer
			ind=geneList.index[((geneList["Chromosome"]==SNPchr) & (geneList["Start"]-buffer <=SNPpos) &  (geneList["End"]+buffer >=SNPpos))].tolist()
			if ind==[]:
				#This means no matching genes were found for this SNP in annotation step.
				dfAnnotation.at[len(dfAnnotation)-1, "SNPindex"].append(SNPidx)
				dfAnnotation.at[len(dfAnnotation)-1, "VariantID"].append(VariantID)
			else:
				for i in ind:
					dfAnnotation.at[i,"SNPindex"].append(SNPidx)
					dfAnnotation.at[i,"VariantID"].append(VariantID)
			pbar.update(1)
	dfAnnotation=dfAnnotation[dfAnnotation.astype(str)["SNPindex"] != '[]'] #Drop SNP-sets with no SNPs in them
	dfAnnotation=dfAnnotation.sort_index()
	dfAnnotation.index = range(len(dfAnnotation))  #Reassign indices
	return dfAnnotation

def dropSingletonSets(annotationDF, SNPList, geneList, buffer):
	"""
	Helper function for annotate()
	After an initial step of annotations, drops SNPsets with only one SNP in them and re-annotates the SNPs based on the remaining SNP-sets
	--------------------------------------------------
	Takes in
	annotationDF: pandas dataframe object containing SNP - to - SNP-set annotation information from the annotation step before singleton SNP-sets are dropped
	SNPList: pandas dataframe object containing SNP list information
	geneList: pandas dataframe object containing gene range information
	buffer: parameter that determines how much buffer (in basepairs) to allow around SNP-set boundaries during the last step of annotation.
	--------------------------------------------------
	Returns
	annotationDF: pandas dataframe object containing the final SNP - to - SNP-set annotation that do not include singleton SNP-sets
	"""
	annotationDF_pruned=annotationDF[annotationDF['SNPindex'].map(len) > 1] #Get rid of SNP-sets with only one SNPs in them
	SNPsets=annotationDF_pruned["GeneID"].tolist() #Get the resulting genes and SNPsets
	geneList_pruned=geneList[geneList["GeneID"].isin(SNPsets)] #Only keep these SNPsets in the geneList
	annotationDF=annotateSets(SNPList, geneList, buffer=buffer)
	return annotationDF

def annotate(path_SNPList, path_geneGuide,  outputFile, intergenic=False, buffer=0, dropSingletons=False):
	"""
	Function to create a SNP-to-SNPset annotation dataframe. 
	Wrapper around the previous functions. Called by the user.
	--------------------------------------------------
	Takes in
	path_SNPList: (string) Path to the SNP list file (file needs to be in either .map or .bim format, check docstring of function read_SNP_file() for more information)
	path_geneGuide: (string) Path to the gene range list file (check docstring of function read_gene_file() for more information)
	outputFile: (string) Path to the .txt file (containing the filename) where the final annotation should be saved in a tab-delimited format.
	intergenic: (boolean) True or False. A parameter of the annotation. Determines whether intergenic SNP-sets will be considered or not in annotation.
		If considered (True), intergenic regions between genes are created and SNPs are annotated accordingly. 
		If not considered (False), all SNPs in intergenics regions are grouped into one group called "Unannotated", which corresponds to the last SNP-set in the annotation dataframe.
	buffer: (integer) A parameter of the annotation. Determines how much of a buffer (in basepairs bp) to allow for around SNP-set boundaries when considering whether a SNP should be annotated in a SNPset
	dropSingletons: (boolean) True or False. A parameter of the annotation. Determines whether to keep SNPsets that are annotated only with one SNP or not. 
		If True, they are included in the annotation. If False, they are dropped from the annotation and the SNPs are re-annotated with the remaining SNPsets
	--------------------------------------------------
	Returns 
	dfAnnotation: pandas dataframe object that conatins the final SNP - to - SNP-set annotations. It is saved in tab-delimited format to the file specified by "outputFile" argument
	
	"""
	SNPList=read_SNP_file(path_SNPList) #Read in the SNP list as a dataframe
	geneList=read_gene_file(path_geneGuide) #Read in the gene list as a dataframe

	if intergenic==False:
		message="You have chosen to annotate SNP-sets without intergenic regions and with a buffer of " +str(buffer)+"bp"
		print(message)
		dfAnnotation=annotateSets(SNPList,geneList, buffer=buffer)
	elif intergenic==True:
		message="You have chosen to annotate SNP-sets with intergenic regions and with a buffer of " +str(buffer)+"bp"
		print(message)
		geneList=generate_intergenicDF(SNPList,geneList)
		dfAnnotation=annotateSets(SNPList, geneList, buffer=buffer)
	if dropSingletons==True:
		print("Dropping SNP-sets that are singletons (containing only one SNP) and re-annotating SNPs without them")
		dfAnnotation=dropSingletonSets(dfAnnotation, SNPList, geneList, buffer=buffer)

	# Save the resulting annotationDF to a file for future reference:
	message="Saving annotation results to file "+outputFile
	print(message)
	dfAnnotation.to_csv(outputFile, sep="\t")
	return dfAnnotation

def getMaskMatrix(path_SNPList, annotationDF, outputFile):
	"""
	Function to create he corresponding SNP - to- SNP-set annotation mask, which is a sparse matrix that guides the connections of the hidden layer of BANN model. 
	Called by the user.
	--------------------------------------------------
	Takes in
	path_SNPList: (string) Path to the SNP list file (file needs to be in either .map or .bim format, check docstring of function read_SNP_file() for more information)
	annotationDF: (pandas dataframe) annotation dataframe created and output by the annotate() function.
	outputFile: (string) Path to the .txt file (containing the filename) where the final mask matrix should be saved in a tab-delimited format.
	--------------------------------------------------
	Returns 
	mask: (numpy array) SNP - to- SNP-set annotation matrix. This is a sparse matrix of 0s and 1s and of size (Number of SNPs by Number of SNP-sets), where rows correspond to SNPs and columns correspond to SNP-sets. 
	If an entry is 1, it means the corresponding SNP is annotated within the corresponding SNPset.
	"""
	p=len(annotationDF) #The number of SNP-sets in annotation data frame
	N=len(pd.DataFrame(pd.read_csv(SNPList_path, sep='\t', header=None)))
	mask=np.zeros(shape=[N,p]) #initialize the mask matrix
	print("creating mask")
	with tqdm(total=len(annotationDF)) as pbar:
		for index, row in annotationDF.iterrows():
			#Iterating over all SNP-sets in the annotationDF (which will correspond to columns in the mask matrix)
			SNPidx=row["SNPindex"] #Get the SNP indices that will be annotated within this SNP-set (these indices will correspond to rows in the mask matrix)
			for i in SNPidx:
				#Change mask matrix values to 1 from 0 for the corresponding SNP - SNP-set pairs:
				mask[i,index]=1
			pbar.update(1)
	message="Saving annotation mask to file "+outputFile+" in tab-delimited format"
	np.savetxt(outputFile, mask, delimiter="\t")
	print(message)
	return mask
