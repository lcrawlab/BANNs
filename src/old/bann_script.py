## Import needed packages:
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
# import matplotlib.pyplot as plt

## Import our software:
from annotation import *
from utils import *
from customLayers import *
from bann_keras import BANN_Quantitative


#######################################################
#################### GET DATA #########################
#######################################################
os.chdir("/Users/pinar/Desktop/ML-GSEA/Data/")
#X=np.load("PickledData_Simulation/Br19_5k_X.npy")
#y=np.load("PickledData_Simulation/Br19_5k_0.6_0.08_0.1_y1.npy")
start=time.time()

mapFile="Br19_10k_map.txt"
geneFile="glist-hg19.tsv"
end=time.time()
########################################################
#################### ANNOTATION ########################
########################################################
SNPList=read_SNP_file(mapFile)
geneList=read_gene_file(geneFile)
annotationDF=annotateGenes(SNPList, geneList)
gmas

#geneMask=np.load("PickledData_Simulation/Br19_5k_0.6_0.08_0.1_geneMask.npy")
N1=X.shape[0]
p1=X.shape[1]
p2=geneMask.shape[1]

########################################################
#################### BUILD MODEL #######################
########################################################

# Training settings
n_epochs = 100000
batch_size = 4000
l_rate=1e-3

# Network architecture
# layers = []
# layers.append(Deterministic(p2, geneMask, activation='relu',input_shape=(p1,)))
# # layers.append(tfp.layers.DenseLocalReparameterization(1))
# layers.append(Probabilistic(1))
layers=build_geneNetwork(p1,p2,geneMask)
bnn = BANN_Quantitative(layers, p1, l_rate)

########################################################
#################### TRAIN MODEL #######################
########################################################

history = bnn.fit(X,y, batch_size=batch_size, epochs=n_epochs,
                        validation_split=0.2,
                        # callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_acc", patience=4)],
                        callbacks=[],
                        verbose=1)

yplot=bnn.var_params()[0]
print(yplot)
# xplot=range(1,len(yplot)+1)
# ydf=pd.DataFrame(yplot)
# ydf.columns=["bernoulli"]
# ydf["gIndex"]= range(len(ydf))
# ydf.sort_values(by=["bernoulli"], ascending=[False], inplace=True)
# print(ydf)
# plt.scatter(xplot,yplot)
# plt.show()

# Get training and test loss histories
# training_loss = history.history['loss']
# test_loss = history.history['val_loss']

# # Create count of the number of epochs
# epoch_count = range(1, len(training_loss) + 1)

# # Visualize loss history
# plt.plot(epoch_count, training_loss, 'r--')
# plt.plot(range(1, len(training_loss)), test_loss, 'b-')
# plt.legend(['Training Loss', 'Test Loss'])
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.show()





