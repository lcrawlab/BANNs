
import numpy as np
from utils import *

def innerLoop(X,y,mask,xy,d,tau_snp,sigma_snp,np.log(10)*logodds_snp,alpha_snp,mu_snp,update_order_snp,tau_set,sigma_set,np.log(10)*logodds_set,alpha_set,mu_set,update_order_set,
                   tol,maxiter,outer_iter):
  n=X.shape[0]
  p=X.shape[1]
  g=mask.shape[1]

  Xr = np.ndarray.flatten(np.matmul(X,(alpha_snp * mu_snp)))
  s_snp=sigma_snp*tau_snp/(sigma_snp*d+1)
  err_snp=np.repeat(0,maxiter)
  sigma0_snp=1
  n0_snp=10

  bH=np.matmul(rep_col(mu_snp * alpha_snp,p),mask)
  H=np.matmul(X, bH)
  Hr = np.ndarray.flatten(np.matmul(H, alpha_set*mu_set))
  Hd=diagsq(H)
  s_set = sigma_set*tau_set/(sigma_set*Hd+1)
  err_set=np.repeat(0,maxiter)
  sigma0_set=1
  n0_set=10
  logw_snp=np.repeat(0,maxiter)
  logw_set=np.repeat(0, maxiter)

  for i in range(0,maxiter):
    alpha0_snp = alpha_snp
    mu0_snp = mu_snp
    s0_snp=s_snp
    tau0_snp=tau_snp
    sigma_old_snp = sigma_snp

    alpha0_set = alpha_set
    mu0_set = mu_set
    s0_set=s_set
    tau0_set = tau_set
    sigma_old_set = sigma_set

    logw0_snp = varLoss(Xr,d,y,tau_snp,alpha_snp,mu_snp,s_snp,tau_snp*sigma_snp, logodds_snp)
    logw0_set = varLoss(Hr,Hd,y,tau_set,alpha_set,mu_set,s_set,tau_set*sigma_set, logodds_set)

    res   = varParamUpdate(X,mask, tau_snp, sigma_snp, logodds_snp, xy, d, alpha_snp, mu_snp, Xr, update_order_snp,
                            tau_set, sigma_set, logodds_set, alpha_set, mu_set, Hr, update_order_set) 
    
    alpha_snp = res["alpha_snp"]
    mu_snp    = res["mu_snp"]
    Xr    = res["Xr"]
    alpha_set = res["alpha_set"]
    mu_set    = res["mu_set"]
    Hr    = res["Hr"]
    del(res)
    
    logw_snp[i] = varLoss(Xr,d,y,tau_snp,alpha_snp,mu_snp,s_snp,tau_snp*sigma_snp, logodds_snp)
    logw_set[i] = varLoss(Hr,Hd,y,tau_set,alpha_set,mu_set,s_set,tau_set*sigma_set, logodds_set)

    tau_snp= (norm2(y-Hr)**2+dot(Hd, betavar(alpha_set, mu_set, s_set))+ dot(alpha_set, (s_set+m_set**2)/sigma_set))/(n+np.sum(alpha_set))
    s_set   =sigma_set*tau_set/(sigma_set*Hd + 1)
    sigma_set = (sigma0_set*n0_set + dot(alpha_set,s_set + mu_set**2))/(n0_set + tau_set*np.sum(alpha_set))
    s_set  = sigma.set*tau.set/(sigma.set*Hd + 1)

    #CONVERGENCE CHECK:
    err_snp[i] = np.max(numpy.absolute(alpha_snp - alpha0_snp))
    err_set[i] = np.max(numpy.absolute(alpha_set - alpha0_set))

    if (logw_snp[i]+logw_set[i] < logw0_snp+logw0_set):
      logw_snp[i] = logw0_snp
      logw_set[i] = logw0_set
      err_snp[i]  = 0
      err_set[0]  = 0
      
      tau_snp = tau0_snp
      sigma_snp = sigma_old_snp
      alpha_snp = alpha0_snp
      mu_snp = mu0_snp
      s_snp = s0_snp
      
      tau_set = tau0_set
      sigma_set = sigma_old_set
      alpha_set = alpha0_set
      mu_set = mu0_set
      s_set = s0_set
      break
    elif((err_snp[i] < tol)&(err_set[i] < tol)):
      break
    
    res={}
    res["logw_snp"]=logw_snp[1:i]
    res["logw_set"]=logw_set[1:i]
    res["err_snp"]=err_snp[1:i]
    res["err_set"]=err_set[1:i]
    res["tau_snp"]=tau_snp[1:i]
    res["sigma_snp"]=sigma_snp[1:i]
    res["alpha_snp"]=alpha_snp[1:i]
    res["mu_snp"]=mu_snp[1:i]
    res["s_snp"]=s_snp[1:i]

    res["tau_set"]=tau_set[1:i]
    res["sigma_set"]=sigma_set[1:i]
    res["alpha_set"]=alpha_set[1:i]
    res["mu_set"]=mu_set[1:i]
    res["s_set"]=s_set[1:i]

    return res

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

  res["logw.snp"] = res["logw.snp"] - np.linalg.det(np.cross(I_snp))
  res["logw.set"] = res["logw.set"]  - np.linalg.det(np.cross(I_set))
  
  res["b_snp"] =np.asarray(SIy_snp - np.matmul(SIX_snp,(alpha.snp*mu.snp)))
  res["b_set"]=np.asarray(SIy_set - np.matmul(SIX_set,(alpha.set*mu.set)))
  
  numiter_snp  = len(res["logw_snp"])
  res["logw_snp"]= res["logw_snp"][numiter_snp]
  numiter_set  =len(res["logw_set"])
  res["logw_set"]= res["logw_set"][numiter_set]
  return res


def linear_predictors(X, Z, b, alpha, mu):
  numModels=alpha.shape[1]
  Y=np.matmul(Z,b)+np.matmul(X,(alpha*mu))
  return Y 
