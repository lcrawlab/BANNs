from utils import *
from customModel import *


def BANN(X,y,mask, centered=False, numModels=20, tol=1e-4, maxiter=1e4):
	X=np.asarray(X)
	y=np.asarray(y)
	mask=np.asarray(mask)

	n=X.shape[0]
	p=X.shape[1]
	g=X.shape[1]
	xy=np.matmul(y,X)
	d=diagsq(X)

	tau_snp=np.repeat(np.var(y),numModels)
	sigma_snp=np.repeat(1,numModels)
	logodds_snp=np.linspace(-np.log10(p),-1, num=numModels).reshape(1,numModels)
	alpha_snp=rand(p,numModels)
	alpha_snp=alpha_snp/rep_row(np.sum(alpha_snp,axis=0),p)
	mu_snp=randn(p,numModels)
	update_order_snp=np,arange(p)

	tau_set=np.repeat(np.var(y),numModels)
	sigma_set=np.repeat(1,numModels)
	logodds_set=np.linspace(-np.log10(g),-1, num=numModels).reshape(1,numModels)
	alpha_set=rand(g,numModels)
	alpha_set=alpha_snp/rep_row(np.sum(alpha_snp,axis=0),g)
	mu_set=randn(g,numModels)
	update_order_set=np,arange(g)

	logw_snp=np.repeat(0,numModels)
	s_snp=np.zeros((p,numModels))
	b_snp=np.zeros((1,numModels))

	logw_set=np.repeat(0,numModels)
	s_set=np.zeros((g,numModels))
	b_set=np.zeros((1,numModels))

	I_snp=np.ones((n,1))
	SIy_snp = np.matmul(np.linalg.solve(n,y),I_snp) 
	SIX_snp =np.matmul(np.linalg.solve(n,I_snp,X))


	I_set=np.ones((n,1))
	SIy_set =np.matmul(np.linalg.solve(n,y),I_set)
	SIX_set =np.matmul(np.linalg.solve(p,I_set),X) 

	for i in range(0,numModels):
		out=outerloop(X, I_snp, y, mask,xy, d, SIy_snp, SIX_snp, I_set,SIy_set, SIX_set,
					 tau_snp[i],sigma_snp[i],logodds_snp[,i],alpha_snp[,i],mu_snp[,i],update_order_snp,
					 tau_set[i],sigma_set[i],logodds_set[,i],alpha_set[,i],mu_set[,i],update_order_set,
					 tol,maxiter,i )

		  ### Finding the best initialization of h
		logw_snp[i]  = out["logw_snp"]
		tau_snp[i]  = out["tau_snp"]
		sigma_snp[i]  = out["sigma_snp"]
		b_snp[,i] = out["b_snp"]
		alpha_snp[,i] = out["alpha_snp"]
		mu_snp[,i] = out["mu_snp"]
		s_snp[,i] = out["s_snp"]
		
		logw_set[i]  = out["logw_set"]
		tau_set[i]   = out["tau_set"]
		sigma_set[i] = out["sigma_set"]
		b_set[,i] = out["b_set"]
		alpha_set[,i] = out["alpha_set"]
		mu_set[,i]= out["mu_set"]
		s_set[,i] = out["s_set"]

	i_snp = np.argmax(logw_snp)
	alpha_snp = rep_col(alpha_snp[,i_snp],numModels)
	mu_snp = rep_col(mu_snp[,i_snp],numModels)
	tau_snp = np.repeat(tau_snp[i_snp],numModels)
	sigma_snp = np.repeat(sigma_snp[i_snp],numModels)
	  
	i_set = np.argmax(logw_set)
	alpha_set = rep_col(alpha_set[,i_set],numModels)
	mu_set = rep_col(mu_set[,i_set],numModels)
	tau_set = np.repeat(tau_set[i_set],numModels)
	sigma_set <- np.repeat(sigma_set[i_set],numModels)


	for i in range(0,numModels):
		out=outerloop(X, I_snp, y,mask, xy, d, SIy_snp, SIX_snp, I_set,SIy_set, SIX_set,
					 tau_snp[i],sigma_snp[i],logodds_snp[,i],alpha_snp[,i],mu_snp[,i],update_order_snp,
					 tau_set[i],sigma_set[i],logodds_set[,i],alpha_set[,i],mu_set[,i],update_order_set,
					 tol,maxiter,i )

		  ### Finding the best initialization of h
		logw_snp[i]  = out["logw_snp"]
		tau_snp[i]  = out["tau_snp"]
		sigma_snp[i]  = out["sigma_snp"]
		b_snp[,i] = out["b_snp"]
		alpha_snp[,i] = out["alpha_snp"]
		mu_snp[,i] = out["mu_snp"]
		s_snp[,i] = out["s_snp"]
		
		logw_set[i]  = out["logw_set"]
		tau_set[i]   = out["tau_set"]
		sigma_set[i] = out["sigma_set"]
		b_set[,i] = out["b_set"]
		alpha_set[,i] = out["alpha_set"]
		mu_set[,i]= out["mu_set"]
		s_set[,i] = out["s_set"]


	w_snp = normalizelogweights(logw_snp)
	pip_snp = np.matmul(alpha_snp,w_snp)
	beta_snp = np.matmul((alpha_snp*mu_snp),w_snp)
	beta_cov_snp = np.matmul(b_snp,w_snp)
	  
	w_set = normalizelogweights(logw_set)
	pip_set = np.matmul(alpha_set,w_set)
	beta_set = np.matmul((alpha_set*mu_set),w_set)
	beta_cov_set = np.matmul(b_set,w_set)

#########

	SNP_res = {"b":b_snp,"logw":logw_snp, "w":w_snp, "tau":tau_snp, "sigma":sigma_snp,"logodds":logodds_snp, "alpha":alpha_snp,
			  "mu":mu_snp, "s":s_snp, "pip":pip_snp, "beta":beta_snp, "beta_cov":beta_cov_snp, "y":y}


	SNPset_res = {"b":b_set,"logw":logw_set, "w":w_set, "tau":tau_set, "sigma":sigma_set,"logodds":logodds_set, "alpha":alpha_set,
			  "mu":mu_set, "s":s_set, "pip":pip_set, "beta":beta_set, "beta_cov":beta_cov_set, "y":y}


	results = {"SNP_res":SNP_res, "SNPset_res":SNPset_res}
	results["model_pve"]= estimatePVE(results["SNP_res"], X)

	return results

