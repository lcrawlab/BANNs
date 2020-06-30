# Biologically Annoted Nerual Networks (BANNs)

**BANNs** are a class of feedforward Bayesian models with partially connected architectures that are guided by predefined SNP-set annotations.
 
 ## Installation and Dependencies
 
 We implement BANNs in three different software packages. The first two are implemented in Python using Tensorflow and numpy, respectively. The third version is implemented in R. The dependencies and requirements needed to install and run each version of the BANN software may be found in the README of the corresponding subdirectories. 
 
 ## Tutorial and Examples
 
 For each version of the software, we also provide example code and a toy dataset which illustrate how to use BANNs and conduct multi-scale genomic inference. 

## Background 

The BANN framework simply requires individual-level genotype/phenotype data and a predefined list of SNP-set annotations (see schematic below). 

* `X`: Genotype matrix of size N by P where N is the number of individuals and P is the number of SNPs.
* `y`: Phenotype file of N rows where N is the number of individuals and each row stores the continuous phenotype value. 
* `mask`: Mask matrix of size P by G where P is the number of SNPs and G is the number of SNP sets (genes). Each column is a vector filled with 0s and 1s with 1 indicates the appearance of the corresponding SNP of that row within the gene of that column and vice versa.  

![alt text](misc/Fig1.pdf)

# TUTORIAL
For each version, we provide an example code and a toy example data in the corresponding subdirectory to illustrate how to use BANNs. Please check accordingly.

# NOTES
* Please make sure that the individual order (rows) of the genotype matrix X is the same with phenotype file y.
* Please make sure that the SNP order (columns) of genotype matrix X is the same with the mask file (rows). 
* We report the results according to the order of the input files. For example, the Posterior Inclusion Probabilities (PIPs) of genes are ordered in the same way to the order of the maks file (columns). 

# RELEVANT CITATIONS


# QUESTIONS AND FEEDBACK
For questions or concerns with BANNs, please contact [Pinar Demetci](mailto:pinar_demetci@brown.edu), [Wei Cheng](mailto:wei_cheng1@brown.edu), [Lorin Crawford](mailto:lorin_crawford@brown.edu).

We appreciate any feedback you may have with our software and/or instructions.








