
# Single step of coordinate ascent updating variational parameters to maximize the variational lower bound when a mixture of normals is used:
singlemixupdate<-function(X, xy, d, sigma, sa, q, alpha, mu, Xr, s, logw, n, k){
  mu[0]   = 0;
  s[0]    = 0;
  logw[0] = log(q[0] + 5e-52);
  
  for (i in 1:k){
    print("s")
    print(s)
    print(sigma*sa[i]/(sa[i]*d + 1);)
    s[i] = sigma*sa[i]/(sa[i]*d + 1);
  }
  r = dot(alpha,mu,k);
  t = xy + d*r - dot(x,Xr,n); #used for mu
  for (i in 1:k){
    mu[i] = s[i]/sigma*t; 
  }
  for (i in 1:k){
    SSR     = mu[i]*mu[i]/s[i]; 
    logw[i] = log(q[i] + 5e-52) + (log(s[i]/(sigma*sa[i])) + SSR)/2;
  }
  w<-normalizelogweights(logw,alpha,k);
  rnew = dot(alpha,mu,k);
  add(Xr,rnew - r,x,n);
  
  return(list(Xr = Xr, mu=mu, alpha = alpha,logw=logw,w=w,s=s,sigma=sigma, sa=sa ))
}


BANNmixUpdate<-function(X, sigma, sa,w,xy,d, alpha, mu, Xr, update.order){
    
  n <- nrow(X)
  k <- length(w)
  numiter <- length(update.order)
  
  #initializing variables for storage:
  s <- numeric(k)
  logw<-numeric(k)
  
  # Iterate through coordinate updates
  for (j in 0:numiter){
    print(j)
    i=update.order[j]
    x= getColumn(X,i,n)
    outSingle<-singlemixupdate(x,xy[i],d[i],sigma,sa,w,getColumn(alpha,i,k),getColumn(mu,i,k),Xr,s,logw,n,k)
    Xr=outSingle$Xr
    mu=outSingle$mu
    alpha=outSingle$alpha
    logw=outSingle$logw
    w=outSingle$w
    s=outSingle$s
    sigma=outSingle$sigma
    sa=outSingle$sa
  }
 return(list(Xr = Xr, mu=mu, alpha = alpha,logw=logw,w=w,s=s,sigma=sigma, sa=sa ))
}
  