HCindep <-
function(S,Sigmamat,just_stat=F,maxcorr=.95,p_thresh=100){
  if(length(S)==1){return(2*pnorm(abs(S),lower.tail=F))}
  out=MakeFullRank(S,Sigmamat,maxcorr)
  S=out[[1]];Sigmamat=out[[2]]
  P = length(S)
  m = P
  if(P==1){return(2*pnorm(abs(S),lower.tail=F))}
  #Sigmawiggle = as.matrix(nearPD(rWishart(1,n,diag(P))[,,1]/n,corr=T)$mat)#adjust for unknown covariance  
  #S = t(solve(chol(Sigmawiggle))) %*% t(solve(chol(Sigmamat))) %*% S  
  S =  t(solve(chol(Sigmamat))) %*% S
  if(P > p_thresh){
    g = function(tt){
      tailprob=2*pnorm(tt,lower.tail=F)
      return(P*tailprob*(1-tailprob))      
    }
    pval = GHCsimulated(S,diag(P),g,1000)
    if(pval>0.1){return(pval)}
    pval = GHCsimulated(S,diag(P),g,10000)
    if(pval>0.01){return(pval)}
    pval = GHCsimulated(S,diag(P),g,1000000)
    return(pval)
  }  
  t_mesh = sort(abs(S))
  GHC.mesh = rep(0,m)
  tailprob = 2*pnorm(t_mesh,lower.tail=F)
  for(i in 1:m){
    if(tailprob[i]==1){
      GHC.mesh[i]=0
    } else{
      GHC.mesh[i]=(length(which(abs(S)>=t_mesh[i]))-P*tailprob[i])/sqrt(P*tailprob[i]*(1-tailprob[i]))    
    }
  }
  GHCstat = max(GHC.mesh)
  if(just_stat){return(GHCstat)}
  if(GHCstat<=0){return(1)}
  t_mesh = rep(0,P)
  for(k in 1:P){
    t_mesh[k]=uniroot(function(x) GHCstat*sqrt(P*2*pnorm(x,lower.tail=F)*(1-2*pnorm(x,lower.tail=F)))+P*2*pnorm(x,lower.tail=F)-(P-k+1),interval=c(10^-10,50))$root
  }
  m=P;tailprob = 2*pnorm(t_mesh,lower.tail=F);cv=(P-1):0
  pval_prod=rep(0,m)
  PM = matrix(0,nrow=cv[1]+1,ncol=m)
  PM[,1] = dbinom(0:cv[1],P,tailprob[1])
  pval_prod[1]=sum(PM[1:(cv[1]+1),1])
  for(j in 2:m){
    PM=PM_updateindep(j,t_mesh,cv,PM,tailprob)
    pval_prod[j]=sum(PM[1:(cv[j]+1),j])
  }
  return(1-prod(pval_prod))
}
