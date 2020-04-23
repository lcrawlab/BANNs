GBJ is an R package and the point of it is to carry out multiple hypothesis test using a Berk-Jones test, modified for GWAS settings. The reason why they choose Berk-Jones test is because they claim this is "optimal in some sense" to detect rare and weak signals among a set of independent factors. 

Their p-value calculations don't depend on permutations (good for LD case)

For conducting gene-set analysis, we group SNPs into genes and then run GBJ on them.

Input similar to SKAT, no parameter tuning (?)


