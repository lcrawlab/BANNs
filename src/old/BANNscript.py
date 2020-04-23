#Import tensorflow related packages
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K

# Import other packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os,sys,inspect

# Import scripts from our model
from BANN import BANN_Quantitative
from utils import *
from customModel import *
from annotation import *
from evaluate import * 


## Set directory to where to read and write files. This sets to the directory of the current file
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
os.chdir(current_dir)


###########################
##  Read in input files  ##
###########################
X=np.genfromtxt("testData/XSimu.csv",delimiter=",")
y=np.genfromtxt("testData/ySimu.csv")
print("X shape", X.shape)
print("y shape", y.shape)


###########################
##      Annotation       ##
###########################

# First create a dataframe of annotations:
# mapFile="Br19_10k_map.txt"
# geneFile="glist-hg19Real.txt"
# SNPList=read_SNP_file(mapFile)
# geneList=read_gene_file(geneFile)
# annotationDF=annotateGenes(SNPList, geneList)

# # Then construct the gene-level mask based on these annotations:
# p0=X.shape[1]
# p1=len(annotationDF)
# geneMask=getGeneMask(annotationDF,p0,p1)

# In the case of this example, we have the mask saved as .npy already 
# (because I didn't want to deal with simulating annotation files)
geneMask = np.load("testData/mask.npy")


###########################
##  Set hyperparameters  ##
###########################
n_epochs = 500 #Set to 500-1000 for whole chromosomes
batch_size = 1000 #Can be set to 1000-5000 for whole chromosomes
learning_rate=0.1 #learning rate is best 1e-3 when using whole chromosome. Have not tried for whole genome. 
# The model is pretty sensitive to learningRate/n_epochs balance. 
# The higher the number of parameters, the lower the learning rate needs to be and as a consequence, the higher the n_epochs need to be.
# batch_size is not as important but affects variance of the loss. 

###########################
##    Build the model   ##
###########################
ngenes=geneMask.shape[1]  #or len(annotationDF), The number of genes can be retrieved from the shape of the mask or the annotation DF length
p1 = ngenes #This will always be the number of genes
p2 =1 #This will always be 1 for quantitative models
layers=buildModel(p1,p2,geneMask,activation_fn="relu")
bnn = BANN_Quantitative(layers, l_rate=learning_rate) # Create and train network


###########################
##     Fit the model     ##
###########################

# Based on the loss plot, if you think we are overfitting, you can decrease the patience parameter 
earlystopper = EarlyStopping(monitor='val_loss', patience=50, verbose=1)
fit_history = bnn.fit(X,y, batch_size=batch_size, epochs=n_epochs,
                        validation_split=0.2,
                        callbacks=[earlystopper],
                        verbose=1) #verbose can be set to 0 if we don't want 


                        # callbacks=[EarlyStopping(monitor="val_acc", patience=5)],

###############################################
##    Visualize model training at the end    ##
###############################################
training_loss = fit_history.history['loss']
test_loss = fit_history.history['val_loss']
epoch_count=np.arange(len(training_loss))
plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, test_loss, 'b-')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()


#################################################
##    Infer PIPs for the SNP and gene levels   ##
#################################################
# Note: In the model
# This approximation works with log odds ratio.
# To get the posterior inclusion probabilities, we convert log odds to probabilities
genePIPs=logits2pip(K.eval(bnn.model.layers[-1].variables[0]))
SNPpips=logits2pip(np.sum(K.eval(bnn.model.layers[-2].variables[0])*geneMask, axis=1))


####################################################################
##    Plot PIPs: First graph SNP-level, Second graph gene-level   ##
####################################################################
xaxsSNPs=np.arange(len(SNPpips))
xaxsGenes=np.arange(len(genePIPs))
plt.scatter(xaxsSNPs,SNPpips)
plt.show()
plt.scatter(xaxsGenes,genePIPs)
plt.show()


############################################################################
##    Plot evaluation: First graph ROC, Second graph Precision - Recall   ##
############################################################################
# NOTE: I am using R to plot ROC etc now, included in the folder.








