# ML-GSEA
Note: This README is a work in progress as well as the rest of the repository. I'll try to keep things as organized and easy to understand as possible.

### ORGANIZATION OF THE REPO
- Most of the code is in the folder "src". 
- /scripts has useful code that isn't directly related to the tool (e.g. experiments, streamlining stuff, code for formatting data to use other tools)
- I removed our data from the /Data folder and won't be pushing it to GitHub. But I kept the guide files and test files there since that's public info anyway. 

### DEPENDENCIES
The packages the current code depends on are listed in requirements.txt [Note: I need to update this!]

### INPUT FILES
Depending on whether one is working with gene-level inference or pathway-level inference, there are either 4 or 5 files required:

1) X, the genotype matrix: .csv file
2) y, the phenotype matrices: .csv file   [can do .bed as well]
(I generate X and y by writing RData to tab delimeted files in R)
3) .map file for SNP List: PLINK formatted .map file (https://www.cog-genomics.org/plink2/formats#map)
4) Gene range list file (e.g. glist_hg19), format described here: https://www.cog-genomics.org/plink/1.9/resources
5) If working with pathway-level inference: pathway guide file, MSigDB format. The default for this file is set to None so if it doesn't exist, we can carry out gene-level inference without needing it.

### GUIDED SPARSE CONNECTIONS
Throughout the semester, I looked into a few different ways to implement this:
1) Creating sub-networks and connecting them together in one big final neural network:
This was the approach we emplyed in the beginning of the semester as well as last semester with RATE. The problem is, I don't think this allows for overlaps. The sub-networks are created by splitting SNP matrix into matrices of SNPs that belong to the same gene. NNs are created for them. Then the output at the gene level is combined 

2) Specifying trainable variables for the trainable weights in the W matrices, and declaring the weights of the "dropped-out" connections as constants with value "0" and concatenating all these variables to get the W matrix for each layer:
I was working with this earlier in April 2019 but changed my approach since this section of the code takes long (inefficient) and code was messy. I will upload my past attempts in the "old" folder in case we want to compare things.

3)Creating a "drop-out" matrix of 0s and 1s as constants that are element-wise multiplied with the weight matrix in each layer:
This is the current approach because it is fast to do.
The reason I hadn't started with this approach was because the weights that get multiplied with 0 still get updated in backpropagation. However, from a toy dataset I played with, their updates seemed small, likely because ... and the NN still updated the causal weights to something close to the original effect sizes (most causal weight was still the highest weight). So I think this works but I might compare the results of this approach to #2 and see if there is any significant difference.

4) Overwriting the dropout method native to tensorflow. This is actually a viable option and probably one of the most "elegant" ways of implementation along with #3). When I first attempted, I broke a few things, got scared and decided not to follow this route but I think it is probably worth a try.


### VARIATIONAL INFERENCE
There are a few potential ways for this as well:
We no longer use Edward for variational inference but instead use customized tensorflow_probability layers.
Something similar can be done with pytorch too.

### TOGGLING BETWEEN GENE-LEVEL AND PATHWAY-LEVEL INFERENCE
My idea for user-friendliness was to run a GUI at the start, where the user can select the files needed and indicate with a check button whether to do gene-level or pathway-level work.
This is not currently integrated with the rest of the code because when I tried to use it on Oscar with British N=10000 dataset in an interactive job and my connection to the cluster was cut (time-out?) and the job didn't complete running. But we could activate this if wanted. If not, the toggle will likely stay as a command line argument.

#### Quick note: 
If you send /subscribe github lcrawlab/ML-GSEA in our Slack channel, you'll subscribe to updates on this repo.

#### To-do:
Create a "how-to" documentation to describe how to use the code.

