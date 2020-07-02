import tensorflow as tf
import numpy as np
from utils import *

class HiddenLayer(object):
	def __init__(self, X, y, models):
		print(y.shape)
		self.N=X.shape[0]
		self.p=X.shape[1]
		self.models=models

		self.kernel=randn(self.p, self.models) #mu
		self.pip=rand(self.p, self.models) #alpha
		self.pip=self.pip/rep_row(np.sum(self.pip, axis=0),self.p) #alpha
		self.logp=np.linspace(-np.log10(self.p), -1, num=self.models).reshape(1,self.models) #logodds
		self.bias=np.zeros((1,self.models)) #b

	def feedforward(self,X,i):
		return np.matmul(X,(self.pip[:,i] * self.kernel[:,i]))

	def build(self, X, y,epochs):
		self.tau=np.repeat(np.var(y), self.models)*1.0
		self.sigma=np.repeat(1,self.models)*1.0
		self.logw=np.repeat(0, self.models)*1.0
		self.s=np.zeros((self.p,self.models))
		self.I=np.ones((self.N,1))
		self.SIy= np.matmul(y,self.I)*1.0/self.N
		self.SIX= np.matmul(np.transpose(self.I),X)*1.0/self.N

		self.optimize(X, y, epochs)
		i = np.argmax(self.logw)
		self.pip = rep_col(self.pip[:,i],self.models)
		self.kernel = rep_col(self.kernel[:,i],self.models)
		self.tau = np.repeat(self.tau[i],self.models)
		self.sigma = np.repeat(self.sigma[i],self.models)

	def optimize(self, X, y, epochs):
		progbar = tf.keras.utils.Progbar(self.models)
		for i in range(0,self.models): #ADD TQDM HERE!!
			xy=np.matmul(y,X)
			d=diagsq(X)
			logw, err, s = self.gradient_updates(X,y,xy,d,epochs,i)
			logw = logw - np.log(np.absolute(np.linalg.det(np.dot(np.transpose(self.I),self.I))))/2 
			self.logw[i]= logw[len(logw)-1]
			self.bias[:,i] =np.asarray(self.SIy - np.matmul(self.SIX,(self.pip[:,i]*self.kernel[:,i])))
			self.s[:,i]=s
			progbar.update(i+1)

	def train(self, X, y,epochs):
		self.build(X, y, epochs)
		self.optimize(X, y, epochs)


	def gradient_updates(self, X,y,xy,d,epochs,i):
		if(len(self.logp[:,i])==1):
			logodds=np.repeat(self.logp[:,i],self.p)
		logp10=np.log(10)*logodds
		Xr = np.ndarray.flatten(np.matmul(X,(self.pip[:,i] * self.kernel[:,i])))
		s=self.sigma[i]*self.tau[i]/(self.sigma[i]*d+1)
		logw=np.repeat(0.0,epochs)
		err=np.repeat(0.0,epochs)
		for e in range(0,int(epochs)):
			pip0 = self.pip[:,i]
			kernel0 = self.kernel[:,i]
			s0=s
			tau0=self.tau[i]
			sigma_old = self.sigma[i]
			logw0 = elbo(Xr,d,y,self.tau[i],self.pip[:,i],self.kernel[:,i],s,self.tau[i]*self.sigma[i], logp10) #FIX THIS! 

			self.pip[:,i],self.kernel[:,i],Xr=gradients(X, self.tau[i], self.sigma[i], logp10, xy, d, self.pip[:,i], self.kernel[:,i], Xr,self.p)
			logw[e] = elbo(Xr,d,y,self.tau[i],self.pip[:,i],self.kernel[:,i],s,self.tau[i]*self.sigma[i], logp10) #FIX THIS!
			self.tau[i]= (norm2(y-Xr)**2+dot(d, betavar(self.pip[:,i], self.kernel[:,i], s))+ dot(self.pip[:,i], (s+self.kernel[:,i]**2)/self.sigma[i]))/(self.N+np.sum(self.pip[:,i]))
			s = self.sigma[i]*self.tau[i]/(self.sigma[i]*d + 1)
			self.sigma[i] = (10 + dot(self.pip[:,i],s + self.kernel[:,i]**2))/(10 + self.tau[i]*np.sum(self.pip[:,i]))
			s = self.sigma[i]*self.tau[i]/(self.sigma[i]*d + 1)

			#Check convergence
			err[e] = np.max(np.absolute(self.pip[:,i] - pip0))
			if ((logw[e] < logw0) and (e>1)):
			  logw[e] = logw0
			  err[e]  = 0
			  self.tau[i] = tau0
			  self.sigma[i] = sigma_old
			  self.pip[:,i] = pip0
			  self.kernel[:,i] = kernel0
			  s = s0
			  break
			elif((err[e] < 1e-4) and (e>1)):
			  break
		return logw[0:(i+1)], err[0:(i+1)], s






