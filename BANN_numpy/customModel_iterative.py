import numpy as np
from utils import *

def varParamUpdate(X, tau, sigma, logodds, xy, d, alpha0, mu0, Xr0, updates):
  '''
  X is the genotype matrix.
  tau, sigma are the variance hyperparameters.
  logodds is the weights for fixed hyper-prior.
  xy, d, Xr0 are precomputed statistics.
  alpha0, mu0 are free parameters.
  updates is the update order
  '''
  #number of individuals
  n = X.shape[0]
  #number of featuress
  p=X.shape[1]
  #make sure it's a vector of alpha and mus
  alpha = np.ndarray.flatten(np.asarray(alpha0))
  mu=np.ndarray.flatten(np.asarray(mu0))
  Xr = Xr0
  #update each feature
  for i in updates:
    #update variance
    s = sigma*tau/(sigma*d[i] + 1)
    #posterior mean coefficeint
    r  = alpha[i] * mu[i]
    #update mean
    mu[i] = s/tau * (xy[i] + d[i]*r - np.sum(X[:,i]*Xr))
    #update individual weight
    alpha[i] = sigmoid(logodds[i] + (np.log(s/(sigma*tau)) + mu[i]**2/s)/2)
    #update Xr as r is updated
    Xr = Xr + (alpha[i]*mu[i] - r) * X[:,i]
  #summarize results
  res={}
  res["alpha"] = alpha
  res["mu"] = mu
  res["Xr"] = Xr
  return res 

def innerLoop(X,y,xy,d,tau,sigma,logodds,alpha,mu,update_order,tol,maxiter,outer_iter):
  #number of individuals
  n=X.shape[0]
  #number of SNPs
  p=X.shape[1]
  #trick for avoiding compute X^TX matrix
  Xr = np.ndarray.flatten(np.matmul(X,(alpha * mu)))
  s=sigma*tau/(sigma*d+1)
  logw=np.repeat(0.0,maxiter)*1.0
  err=np.repeat(0.0,maxiter)*1.0
  sigma0=1
  n0=10
  #loop until converge
  for i in range(0,int(maxiter)):
    # print('Iteration:', i)
    alpha0 = alpha
    mu0 = mu
    s0=s
    tau0=tau
    sigma_old = sigma
    #current lower bound
    logw0 = varLoss(Xr,d,y,tau,alpha,mu,s,tau*sigma, logodds)
    res = varParamUpdate(X,tau, sigma, logodds, xy, d, alpha, mu, Xr, update_order) 
    alpha = res["alpha"]
    mu    = res["mu"]
    Xr    = res["Xr"]
    del(res)
    logw[i] = varLoss(Xr,d,y,tau,alpha,mu,s,tau*sigma, logodds)
    # print('LB', logw[i])
    #Updating Variance Hyper-Parameters
    tau= (norm2(y-Xr)**2+dot(d, betavar(alpha, mu, s))+ dot(alpha, (s+mu**2)/sigma))/(n+np.sum(alpha))
    s = sigma*tau/(sigma*d + 1)
    sigma = (sigma0*n0 + dot(alpha,s + mu**2))/(n0 + tau*np.sum(alpha))
    s = sigma*tau/(sigma*d + 1)

    #Convergence check based on maximum difference between the posteriro probabilities between iterations
    #or when the lower bound decreases after first update.
    err[i] = np.max(np.absolute(alpha - alpha0))
    if ((logw[i] < logw0) and (i>1)):
      logw[i] = logw0
      err[i]  = 0
      tau = tau0
      sigma = sigma_old
      alpha = alpha0
      mu = mu0
      s = s0
      break
    elif((err[i] < tol) and (i>1)):
      break
  #return result
  res={}
  res["logw"]=logw[0:(i+1)]
  res["err"]=err[0:(i+1)]
  res["tau"]=tau
  res["sigma"]=sigma
  res["alpha"]=alpha
  res["mu"]=mu
  res["s"]=s
  return res

def estimatePVE(model, X):
  '''
  estimate model PVE
  '''
  p=X.shape[1]
  numModels=len(model["logw"])
  pve=np.repeat(0.0,100)
  for i in range(0,100):
    j = np.random.choice(numModels,1,p=model["w"])
    b=model["mu"][:,j]+np.sqrt(model["s"][:,j])*np.random.normal(0,1,p)
    b=b*(np.random.uniform(0,1,p)<model["alpha"][:,j])
    sz=var1(np.matmul(X, b))
    pve[i]=sz/(sz+model["tau"][j])
  return np.mean(pve)
  
def outerloop(X, I, y, xy, d, SIy, SIX, tau, sigma, logodds, alpha, mu, update_order,tol,maxiter,outer_iter):
  #number of features
  p=X.shape[1]
  if(len(logodds)==1):
    logodds=np.repeat(logodds,p)*1.0
  #rescale logodds 
  logodds = np.log(10)*logodds
  #inner loop for Variational EM update
  res=innerLoop(X,y,xy,d,tau,sigma,logodds,alpha,mu,update_order,tol,maxiter,outer_iter)
  #subtract intercept to derive the weights
  res["logw"] = res["logw"] - np.dot(np.transpose(I), I)[0][0]*0.5
  res["b"] =np.asarray(SIy - np.matmul(SIX,(alpha*mu)))
  numiter  = len(res["logw"])
  res["logw"]= res["logw"][numiter-1]
  return res


 
