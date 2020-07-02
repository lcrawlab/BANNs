from utils import *
from customModel_iterative import *
import time

def BANNvarEM(X, y, centered=False, numModels=20, tol=1e-4, maxiter=1e4, show_progress = True):
	'''
	X is the genotype matrix with n by p
	y is the phenotype with dimension of n
	numModels is the number of initialized models for computing LB 
	tol is the convergence tolerance
	maxiter is the maximum iterations
	'''
	### convert data to numpy array
	X=np.asarray(X)
	y=np.asarray(y)
	### normalizing inputs
	if centered==False:
		y = (y-np.mean(y))/np.std(y)
		for i in range(X.shape[1]):
			X[:,i] = (X[:,i]- np.mean(X[:,i]))/np.std(X[:,i])
	### number of samples
	n=X.shape[0]
	### number of features
	p=X.shape[1]
	### precompute xy
	xy=np.matmul(y,X)
	### precompute diagonal elements of X^TX
	d=diagsq(X)
	print("initializing variables")
    ### initialize variance hyper-param
	tau=np.repeat(np.var(y),numModels)*1.0
	sigma=np.repeat(1,numModels)*1.0
	### fixed hyper-parameter (pi_k in the paper)
	logodds=np.linspace(-np.log10(p),-1, num=numModels).reshape(1,numModels)
	alpha=rand(p,numModels)
	alpha=alpha/rep_row(np.sum(alpha,axis=0),p)
	mu=randn(p,numModels)
	update_order=np.arange(p)
    ### normalized importance weights for SNP layer
	logw=np.repeat(0, numModels)*1.0
	s=np.zeros((p,numModels))
	b=np.zeros((1,numModels))
	I=np.ones((n,1))
	SIy= np.matmul(y,I)*1.0/n
	SIX= np.matmul(np.transpose(I),X)*1.0/n
	### Finding the best initialization
	for i in range(0,numModels):
		if show_progress == True:
			print("Initialize model " + str(i+1) + "/" + str(numModels))
		out=outerloop(X, I, y, xy, d, SIy, SIX, tau[i],sigma[i],logodds[:,i],alpha[:,i],mu[:,i],update_order,
					 tol,maxiter,i)
		logw[i]  = out["logw"]
		tau[i]  = out["tau"]
		sigma[i]  = out["sigma"]
		b[:,i] = out["b"]
		alpha[:,i] = out["alpha"]
		mu[:,i] = out["mu"]
		s[:,i] = out["s"]
	i = np.argmax(logw)
	alpha = rep_col(alpha[:,i],numModels)
	mu = rep_col(mu[:,i],numModels)
	tau = np.repeat(tau[i],numModels)*1.0
	sigma = np.repeat(sigma[i],numModels)*1.0
	### Loop for optimizing
	for i in range(0,numModels):
		if show_progress == True:
			print("Updating model " + str(i+1) + "/" + str(numModels))
		out=outerloop(X, I, y, xy, d, SIy, SIX, tau[i],sigma[i],logodds[:,i],alpha[:,i],mu[:,i],update_order,
					 tol,maxiter,i)
		logw[i]  = out["logw"]
		tau[i]  = out["tau"]
		sigma[i]  = out["sigma"]
		b[:,i] = out["b"]
		alpha[:,i] = out["alpha"]
		mu[:,i] = out["mu"]
		s[:,i] = out["s"]
	### normalize weights etc
	w = normalizelogweights(logw)
	pip = np.matmul(alpha,w)
	beta = np.matmul((alpha*mu),w)
	beta_cov = np.matmul(b,w)
	### summarize results
	temp_res = {"b":b,"logw":logw, "w":w, "tau":tau, "sigma":sigma,"logodds":logodds, "alpha":alpha,
			  "mu":mu, "s":s, "pip":pip, "beta":beta, "beta_cov":beta_cov}
	### estiamte model PVE
	pve = estimatePVE(temp_res, X, nr = 1000)
	results = {"b":b,"logw":logw, "w":w, "tau":tau, "sigma":sigma,"logodds":logodds, "alpha":alpha,
			  "mu":mu, "s":s, "pip":pip, "beta":beta, "beta_cov":beta_cov, "pve":pve}
	return results


def BANN(X, mask, y, centered=False, numModels=20, tol=1e-4, maxiter=1e4, show_progress = True):
	'''
	X is the genotype matrix with dimension n-by-p
	mask is the annotation file with dimension p-by-g
	y is the phenotype with dimension n
	numModels is the number of initialized models for computing LB 
	tol is the convergence tolerance
	maxiter is the maximum iterations
	show_progress is the indicator for printing out the progress
	'''
	### number of SNPs
	p = X.shape[1]
	### number of genes
	g = mask.shape[1]
	### optimizing SNP layer
	SNP_res=BANNvarEM(X = X, y = y, centered = centered, numModels = numModels, tol = tol, maxiter = maxiter, show_progress = show_progress)
	### summarize weights
	w = rep_row(SNP_res["w"], p)
	bH=rep_col(np.sum(SNP_res["w"]*SNP_res["mu"]*SNP_res["alpha"], axis = 1),mask.shape[1])*mask
	G = leakyrelu(np.matmul(X, bH))
	### optimizing gene layer with normalization
	SNPset_res = BANNvarEM(X = G, y = y, centered = False, numModels = numModels, tol = tol, maxiter = maxiter, show_progress = show_progress)
	### summarize results
	results = {"SNP_res":SNP_res, "SNPset_res": SNPset_res}
	return results






