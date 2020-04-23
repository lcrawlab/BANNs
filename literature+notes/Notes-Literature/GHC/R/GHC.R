GHC <-
function(S,Sigmamat,just_stat=F,maxcorr=.95,t_0=0,p_thresh = 100,restricted=FALSE){
  #  out=MakeFullRank(S,Sigmamat,maxcorr)
  #  S=out[[1]];Sigmamat=out[[2]]
  S[which(abs(S)<10^-4)]=10^-4
  t_mesh = sort(abs(S))
  P = length(S)
  m = P
  if(P==1){return(2*pnorm(abs(S),lower.tail=F))}
  #   S = t(solve(chol(Sigmamat))) %*% S
  #   Sigma = GetNearIndepSigma(Sigmamat,n)
  #  Sigma=Sigma[lower.tri(Sigma)]
  Sigma = Sigmamat[lower.tri(Sigmamat)]
  num_approx_sum=40
  rho_mom = sapply(1:num_approx_sum,function(x) mean(Sigma^(2*x)))
  t_meshh = c(1:50/10)
  VarFunc_v = rep(0,length(t_meshh))
  for(i in 1:length(t_meshh)){
    VarFunc_v[i]=VarFunc(t_meshh[i],t_meshh[i],P,rho_mom,len_Sigma=length(Sigma),indep_tests=FALSE)[1]    
  }
  g = ApproxVarFunc(t_meshh,VarFunc_v)
  if(P > p_thresh){
    Sigmamat = as.matrix(nearPD(Sigmamat,corr=TRUE)$mat)
    pval = GHCsimulated(S,Sigmamat,g,1000)
    if(pval>0.1){return(pval)}
    pval = GHCsimulated(S,Sigmamat,g,10000)
    if(pval>0.01){return(pval)}
    pval = GHCsimulated(S,Sigmamat,g,1000000)
    return(pval)
  }
  GHC.mesh = rep(0,m)
  tailprob = 2*pnorm(t_mesh,lower.tail=F)
  for(i in 1:m){
    GHC.mesh[i]=(length(which(abs(S)>=t_mesh[i]))-P*tailprob[i])/sqrt(g(t_mesh[i]))
    if(t_mesh[i]<t_0){GHC.mesh[i]=0} ##new
    if(restricted){
      if(t_mesh[i]>qnorm(1-0.1/(P)) || t_mesh[i] < qnorm(1-.5/2)){GHC.mesh[i]=0}
    }
  }
  GHCstat = max(GHC.mesh)
  if(just_stat){return(GHCstat)}
  if(GHCstat<=0){return(1)}
  t_mesh = rep(0,P)
  optimout = optimize(function(x) GHCstat*sqrt(g(x))+P*2*pnorm(x,lower.tail=F),interval=c(0,10),maximum=TRUE,tol=10^-10)
  low_cut = optimout$maximum
  for(k in 1:P){
    if(k == 1 && optimout$objective <=P){
      t_mesh[k]=10^-10
    } else{
      t_mesh[k]=uniroot(function(x) GHCstat*sqrt(g(x))+P*2*pnorm(x,lower.tail=F)-(P-k+1),interval=c(low_cut,30))$root
      #t_mesh[k] = invnorm(1-(2*(P-k+1)+GHCstat^2-GHCstat*sqrt(GHCstat^2+4*(P-k+1)-4*(P-k+1)^2/P))/(4*(GHCstat^2+P)))      
    }
  }
  if(t_0>0){t_mesh = c(t_0,t_mesh[which(t_mesh>t_0)])}
  V = GetVarMat(t_mesh,P,rho_mom,length(Sigma),correct_sum=F,indep_tests=F)  
  m=length(t_mesh);tailprob = 2*pnorm(t_mesh,lower.tail=F);cv=(m-1):0
  SRv = GetCondVarTerms(Sigma,t_mesh,numterms=40)
  pval_prod=rep(0,m)
  PM = matrix(0,nrow=cv[1]+1,ncol=m)
  PM[,1] = CountPDF(0:cv[1],P,P*tailprob[1],V[1,1])
  pval_prod[1]=sum(PM[1:(cv[1]+1),1])
  for(j in 2:m){
    PM=PM_update(j,t_mesh,cv,PM,tailprob,V,SRv,P)
    pval_prod[j]=sum(PM[1:(cv[j]+1),j])
  }
  return(1-prod(pval_prod))
}
