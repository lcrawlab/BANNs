from BANNs import *
import csv
import numpy as np

X=np.genfromtxt("/Users/pinardemetci/Desktop/X_TOY.csv",delimiter=",")
y=np.genfromtxt("/Users/pinardemetci/Desktop/y_TOY.csv",delimiter=",")
mask=np.genfromtxt("/Users/pinardemetci/Desktop/mask_TOY.csv",delimiter=",")
print(diagsq(X))

results=BANN(X,y,mask)

# SNP_results=results["SNP_res"]
# SNPset_results=results["SNPset_res"]

# SNP_csv_columns=SNP_results.keys()
# SNPset_csv_columns=SNPset_results.keys()

# SNP_csv_file = "SNP_results.csv"
# SNPset_csv_file = "SNPset_results.csv"
# try:
#     with open(SNP_csv_file, 'w') as csvfile:
#         writer = csv.DictWriter(csvfile, fieldnames=SNP_csv_columns)
#         writer.writeheader()
#         for data in SNP_results:
#             writer.writerow(data)
# except IOError:
#     print("I/O error")

# try:
#     with open(SNPset_csv_file, 'w') as csvfile:
#         writer = csv.DictWriter(csvfile, fieldnames=SNPset_csv_columns)
#         writer.writeheader()
#         for data in SNPset_results:
#             writer.writerow(data)
# except IOError:
#     print("I/O error")