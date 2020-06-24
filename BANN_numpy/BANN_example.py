#Please make sure you are under the BANN_numpy directory so you can load the following modulus and data with the correct path!!!!!
import sys
from BANNs_iterative import *
import csv
import numpy as np

#load in data
X = np.loadtxt("example_data/Xtest.txt")
y = np.loadtxt("example_data/ytest.txt")
mask = np.loadtxt("example_data/masktest.txt")

#run BANN
res = BANN(X, mask, y, centered=True, numModels=20, tol=1e-4, maxiter=1e4, show_progress = True)

#access results
#SNP layer posterior inclusion probability
#res["SNP_res"]["pip"]
#SNP layer PVE
#res["SNP_res"]["pve"]

#gene layer posterior inclusion probability
# res["SNPset_res"]["pip"]
#gene layer PVE
# res["SNPset_res"]["pve"]