import os
import numpy as np

os.chdir("/Users/pinar/Desktop/ML-GSEA/Data/TextData_Simulation")
X=np.genfromtxt("Br19_5k_X.txt", delimiter="\t")
np.save("Br19_5k_X",X)