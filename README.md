# BANN
Multi-scale Genomic Inference using Biologically Annotated Neural Networks with Variational Expectation-Maximization Algorithm.
 
# DEPENDENCIES
We implement BANNs using Tensorflow, numpy and R. For each version, please check the dependencies requirements in each subdirectory. 

# INSTALLATION
For each version, please check the installation instructions in each subdirectory.

# INPUT FILES
* `X`: Genotype matrix of size N by P where N is the number of individuals and P is the number of SNPs.
* `y`: Phenotype file of N rows where N is the number of individuals and each row stores the continuous phenotype value. 
* `mask`: Mask matrix of size P by G where P is the number of SNPs and G is the number of SNP sets (genes). Each column is a vector filled with 0s and 1s with 1 indicates the appearance of the corresponding SNP of that row within the gene of that column and vice versa.  

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








