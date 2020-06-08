
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
