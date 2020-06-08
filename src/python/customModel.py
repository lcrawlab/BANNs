
import numpy as np


def outerloop(X, I_snp, y,mask, xy,d,SIy_snp,SIX_snp,I_set,SIy_set,SIX_set,tau_snp,sigma_snp,logodds_snp,alpha_snp,mu_snp,update_order_snp,
        tau_set,sigma_set,logodds_set,alpha_set,mu_set,update_order_set,tol,maxiter,outer_iter):
  p=X.shape[1]
  g=SIX_set.shape[1]

  if(len(logodds_snp)==1):
    logodds_snp=np.repeat(logodds_snp,p)
  if(len(logodds_set)==1):
    logodds_set=np.repeat(logodds_set,g)
  
  res=innerLoop(X,y,mask,xy,d,tau_snp,sigma_snp,np.log(10)*logodds_snp,alpha_snp,mu_snp,update_order_snp,tau_set,sigma_set,np.log(10)*logodds_set,alpha_set,mu_set,update_order_set,
                   tol,maxiter,outer_iter)

  out={}
  out["logw.snp"] = out["logw.snp"] - np.linalg.det(np.cross(I_snp))
  out["logw.set"] = out["logw.set"]  - np.linalg.det(np.cross(I_set))
  
  out["b_snp"] =np.asarray(SIy_snp - np.matmul(SIX_snp,(alpha.snp*mu.snp)))
  out["b_set"]=np.asarray(SIy_set - np.matmul(SIX_set,(alpha.set*mu.set)))
  
  numiter_snp  = len(out["logw_snp"])
  out["logw_snp"]= out["logw_snp"][numiter_snp]
  numiter_set  =len(out["logw_set"])
  out["logw_set"]= out["logw_set"][numiter_set]
  return out 


def linear_predictors(X, Z, b, alpha, mu):
  numModels=alpha.shape[1]
  Y=np.matmul(Z,b)+np.matmul(X,(alpha*mu))
  return Y 
