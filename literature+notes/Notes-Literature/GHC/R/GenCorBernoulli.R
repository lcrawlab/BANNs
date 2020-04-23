GenCorBernoulli <-
function(n,mu,rhomat){
  k = length(mu)
  sigmamat = matrix(1,nrow=k,ncol=k)
  for(i in 1:(length(mu)-1)){
    for(j in (i+1):length(mu)){
      mui = mu[i]
      muj = mu[j]
      sigmamat[i,j]=uniroot(function(x) pmvnorm(upper=qnorm(c(mui,muj)),corr=matrix(c(1,x,x,1),nrow=2))[[1]]-rhomat[i,j]*sqrt(mui*(1-mui)*muj*(1-muj))-mui*muj,interval=c(0,1))[[1]]
      sigmamat[j,i]=sigmamat[i,j]
    }
  }
  normsims = rmvnorm(n,sigma=sigmamat)
  outmat = matrix(0,nrow=n,ncol=k)
  for(i in 1:k){
    outmat[which(normsims[,i]<qnorm(mu[i])),i]=1
  }
  return(outmat)
}
