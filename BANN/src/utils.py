import math
import numpy as np

def softplus(x):
  return np.log(1+np.exp(x))

def leakyrelu(x):
  return np.where(x > 0, x, x * 0.01)  

def relu(x):
  return np.where(x > 0, x, 0)  

def sigmoid(x):
  return (1/(1+np.exp(-x)))

def logpexp(x):
  y=x
  indices=np.argwhere(x < 16)
  for j in indices:
    for i in j:
      y[i]= float(np.log(1 + np.exp(x[i])))
  return y

def logsigmoid(x):
  return -logpexp(-x)

def var1(x):
  res=np.var(x)*(len(x)-1)/len(x)
  return res

def var1_cols(x): 
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

def betavar(p,mu,s):
  return p*(s+(1-p)*(mu**2))

def dot(x,y):
  if x.shape!=y.shape:
    y=np.ones(x.shape)
  return np.dot(x,y)

def norm2(x):
  return np.sqrt(dot(x,x))
  
def diagsq(X, a=None):
  m=X.shape[0]
  n=X.shape[1]
  if a==None:
    a=np.repeat(1.0,m)
  y=np.repeat(0.0,n)

  for j in range(0,n):
    for i in range(0,m):
      t=X[i,j]
      y[j]+=t*t*a[i]
  return y 

def elbo(Xr, d, y, sigma, alpha, mu, s, sa, logodds):
  pi=math.pi
  linearLoss= -len(y)*0.5*np.log(2* pi *sigma) - np.sqrt(np.dot(y - Xr,y - Xr))**2/(2*sigma)-np.dot(d,betavar(alpha,mu,s))/(2*sigma)
  kleffect=(sum(alpha) + np.dot(alpha,np.log(s/sa)) - np.dot(alpha, s+ mu**2)/sa)/2 - np.dot(alpha,np.log(alpha+5e-52)) - np.dot(1 - alpha,np.log(1 - alpha+5e-52))
  loss=linearLoss+kleffect+np.sum((alpha-1)*logodds + logsigmoid(logodds))
  return loss

def normalizelogweights(logw):
  w=np.exp(logw-np.max(logw))
  return (w/np.sum(w))

def gradients( X, tau, sigma, logodds, xy, d, alpha, mu, Xr,p):
    #update each feature
    for i in range(0, p):
      s = sigma*tau/(sigma*d[i] + 1)
      r  = alpha[i] * mu[i]
      mu[i] = s/tau * (xy[i] + d[i]*r - np.sum(X[:,i]*Xr))
      alpha[i] = sigmoid(logodds[i] + (np.log(s/(sigma*tau)) + mu[i]**2/s)/2)
      Xr = Xr + (alpha[i]*mu[i] - r) * X[:,i]
    return alpha,mu,Xr 
