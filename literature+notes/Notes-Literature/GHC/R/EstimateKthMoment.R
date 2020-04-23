EstimateKthMoment <-
function(Y,G,k=4,X=NULL,N.res = 2000,m=30){  
  t_mesh = qnorm(1-0.05)+(qnorm(1-10^-5)-qnorm(1-0.05))*(0:(m-1))/(m-1)
  t_dif = t_mesh[2]-t_mesh[1]
  N = length(Y)
  P = ncol(G)
  X = cbind(rep(1,N),X)
  Q = ncol(X)
  XXXX = X %*%solve(t(X)%*% X) %*% t(X)
  H = diag(N)-XXXX
  V = t(G) %*% H
  sigma2 = (Y %*% t(H) %*% H %*% Y)/(N-Q)
  for(i in 1:P){V[i,] = V[i,]/sqrt(sum(V[i,]^2)*sigma2)}
  resid=H%*%Y
  Yres = matrix(XXXX%*%Y,nrow=N,ncol=N.res) + apply(matrix(resid,nrow=N,ncol=N.res),2,sample)  
  Smat = floor((abs(V %*% Yres)-t_mesh[1])/t_dif)+1
  Smat[which(Smat < 1)]=0
  x = 1:floor(m/2)
  Summat_k = matrix(,nrow=length(x),ncol=N.res)
  Mean_v=P*2*pnorm(t_mesh,lower.tail=FALSE)
  for(i in x){Summat_k[i,]=(apply(Smat,2,function(x) length(which(x>=i)))-Mean_v[i])^k}
  Momk=apply(Summat_k,1,mean)  
  k_train = log(Momk)
  coef = as.numeric(lm(k_train~x)$coef)
  CDFretvals = exp(coef[1] + coef[2]*(1:m))
  Summat_k2 = matrix(,nrow=length(x),ncol=N.res)
  for(i in x){Summat_k2[i,]=(apply(Smat,2,function(x) length(which(x==i)))-Mean_v[i]+Mean_v[i+1])^k}
  Momk=apply(Summat_k2,1,mean)  
  k2_train = log(Momk)  
  plot(k2_train)
  coef = as.numeric(lm(k2_train~x)$coef)
  PDFretvals = exp(coef[1] + coef[2]*(1:m))
  retval = list(TailK = CDFretvals,BinK = PDFretvals)
  return(retval)
}
