
import numpy as np

def softplus(x):
  return np.log(1+np.exp(x))

def leakyrelu(x):
  return np.where(x > 0, x, x * 0.01)  

def relu(x):
  return np.where(x > 0, x, 0)  

def logpexp(x):
  y=x
  indices = [i for i,v in enumerate(x >= 4) if v]
  for i in indices:
    y[i] <- np.log(1 + np.exp(x[i]))
  return y

def sigmoid(x):
  return (1/(1+np.exp(-x)))

def logsigmoid(x):
  return -logpexp(-x)

def var1(x):
  n=len(x)
  res=np.var(x)*(n-1)/n
  return res

def var1_cols(x): ### NOTE: instead of "var1.cols" !!!
  n=x.shape[1]
  return np.var(x,axis=1)*(n-1)/n

def rep_col(x,n):
  return np.repeat(x, n).reshape(len(x),n)

def rep_row(x,n):
  return np.tile(x,(n,1))

def rand(m,n):
  return np.random.uniform(0, 1, size=(m, n))

def randn(m,n):
  return np.random.normal(0, 1, size=(m, n))

def dot(x,y):
  return np.sum(x*y)

def norm2(x):
  return np.sqrt(dot(x,x))

def betavar(p,mu,s):
  return p*(s+(1-p)*(mu**2))

def diagsq(X, a=None):
  m=X.shape[0]
  n=X.shape[1]
  if a==None:
    a=np.repeat(1,m)
  y=np.repeat(0,n)
  for j in range(0,n):
    for i in range(0,m):
      t=X[i,j]
      y[j]=y[j]+(t*t*a[i])
  return y 

def varLoss(Xr, d, y, sigma, alpha, mu, s, sa, logodds):
  linearLoss=(len(y)/2*np.log(2*pi*sigma) - norm2(y - Xr)^2/(2*sigma)-dot(d,betavar(alpha,mu,s))/(2*sigma))
  kleffect=((sum(alpha) + dot(alpha,np.log(s/sa)) - dot(alpha,s + mu^2)/sa)/2 -dot(alpha,np.log(alpha+5e-52)) - dot(1 - alpha,log(1 - alpha+5e-52)))
  loss=linearLoss+kleffect+np.sum((alpha-1)*logodds + logsigmoid(logodds))
  return loss

def normalizeLogWeights(logw):
  c=np.max(logw)
  w=np.exp(logw-c)
  return (w/np.sum(w))


def linear_predictors(X, Z, b, alpha, mu):
  numModels=alpha.shape[1]
  Y=np.matmul(Z,b)+np.matmul(X,(alpha*mu))
  return Y 


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