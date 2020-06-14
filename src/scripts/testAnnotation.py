from annotation import *
from utils import *
import os
import numpy as np

os.chdir("/Users/pinar/Desktop/ML-GSEA/Data/")


mapFile="TestSNPList.txt"
geneFile="TestGeneList.txt"
SNPList=read_SNP_file(mapFile)
geneList=read_gene_file(geneFile)

mappingDF=annotateGenes(SNPList,geneList,500)
print(mappingDF)
N=len(SNPList)
