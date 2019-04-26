# ML-GSEA
Note: This README is a work in progress as well as the rest of the repository. I'll try to keep things as organized and easy to understand as possible.

### ORGANIZATION OF THE REPO
- Most of the code is in the folder "src". 
- I am turning my simulation runs into ipython notebooks, stored in the "Notebook" folder instead of making changes to the src code. If one wants to run these on Oscar, though, it will require running the job as an "interactive job". The Caveat is that if the connection is lost, the job will be terminated.
Alternatively, one can turn it into a regular python code and run as a batch job. The advantage is to be able to display results in the notebook along with the code without having to run it again. Will be good for the paper.
- venv contains the virtual environment I work with, where I experimented with packages and their different versions without having to change packages in my own computer or Oscar user environment.  
I am not sure if one can activate it right away and work in the same environment but I think it is possible with the command $source venv/bin/activate as long as virtualenv package is installed (update: tried, doesn't work. But activating the venv and then installing the requirements with pip install requirements.txt works).
- Deleted simulation data from Simulation_data/ folder due to UKBiobank data privacy but the guide files for sparse connections are in Simulation_data folder . The glist-hg19 (SNP-to-gene guide) and KEGG_guide (gene-pathway guide) will stay the same but the .map file will change based on what data we are working with.

### DEPENDENCIES
The packages the current code depends on are listed in requirements.txt

### INPUT FILES
Depending on whether one is working with gene-level inference or pathway-level inference, there are either 4 or 5 files required:

1) X, the genotype matrix: .csv file
2) y, the phenotype matrices: .csv file
(I generate X and y by writing RData to .csv files in R)
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
1) Edward KLqp() is what I have been working with. It works with tensors so defining the NN model in tensorflow and feeding it into Edward works pretty well. Evaluating posteriors is very easy. 
Drawback: Does not work with the latest version of tensorflow so I had to downgrade tensorflow. This did not affect the syntax or the NN. However, when we release it to public use, if we want researchers to use it easily, it might be a good idea to make sure it works with the more recent versions of tf. I have been more focused on getting results so far but now that I am trying to optimize things, I am looking into the next option as well: 


2) The latest version of tensorflow_probability.vi has tensorflow_probability.vi.KL_reverse(), which computes the ELBO in the NN model. Alternatively "autograd" library in python helps computing integrals, derivatives etc. Using one of these (currently trying tfp.vi.KL_reverse() ), we could feed the ELBO loss into the stochastic gradient descent optimizer in tensorflow (e.g. AdamOptimizer()) and train the network that way, then evaluate the means of the weights or the probability of the Bernoulli variables (spike var).  
Caveat: I have read that "stochastic variational optimizer" in Variational Bayes is a modified verions of stochastic gradient descent optimizer but I don't yet know what this modification is. So I am not fully sure that what I describe here is the same things as stochastic variational inference, need to do a few readings.

3) Other options: pytorch-Pyro, BayesPy, PyMC3, Stan etc. Probably won't look into these, though.

### TOGGLING BETWEEN GENE-LEVEL AND PATHWAY-LEVEL INFERENCE
My idea for user-friendliness was to run a GUI at the start, where the user can select the files needed and indicate with a check button whether to do gene-level or pathway-level work.
This is not currently integrated with the rest of the code because when I tried to use it on Oscar with British N=10000 dataset in an interactive job and my connection to the cluster was cut (time-out?) and the job didn't complete running.
I just don't like running interactive jobs on the cluster. But we could activate this if wanted. If not, the toggle will likely stay as a command line argument.

#### Quick note: 
If you send /subscribe github lcrawlab/ML-GSEA in our Slack channel, you'll subscribe to updates on this repo.

