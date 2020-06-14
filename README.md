# ML-GSEA
Note: This README is a work in progress as well as the rest of the repository. I'll try to keep things as organized and easy to understand as possible.

### ORGANIZATION OF THE REPO
- Most of the code is in the folder "src". 
- /scripts has useful code that isn't directly related to the tool (e.g. experiments, streamlining stuff, code for formatting data to use other tools)
- I removed our data from the /Data folder and won't be pushing it to GitHub. But I kept the guide files and test files there since that's public info anyway. 

### DEPENDENCIES
The packages the current code depends on are listed in requirements.txt 

### INPUT FILES
Depending on whether one is working with gene-level inference or pathway-level inference, there are either 4 or 5 files required:

1) X, the genotype matrix: .csv file
2) y, the phenotype matrices: .csv file   [can do .bed as well]
(I generate X and y by writing RData to tab delimeted files in R)
3) .map file for SNP List: PLINK formatted .map file (https://www.cog-genomics.org/plink2/formats#map)
4) Gene range list file (e.g. glist_hg19), format described here: https://www.cog-genomics.org/plink/1.9/resources
5) If working with pathway-level inference: pathway guide file, MSigDB format. The default for this file is set to None so if it doesn't exist, we can carry out gene-level inference without needing it.



