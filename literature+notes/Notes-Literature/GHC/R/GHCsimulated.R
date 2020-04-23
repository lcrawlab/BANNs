GHCsimulated <-
function(S,Sigmamat,g,numperms=1000){
  p=length(S)
  f = function(t,p,s){
    tailprob=pnorm(t,lower.tail=F)
    return((s-2*p*tailprob)/sqrt(g(t)))
  }
  nloops = ceiling(numperms/1000)
  GHCstats = rep(0,nloops*1000)
  for(j in 1:nloops){
    x = rmvnorm(1000,sigma=Sigmamat)
    y = apply(abs(x),1,function(xx) sort(xx,decreasing=T))
    y2 =y
    for(i in 1:p){
      y2[i,]=f(y[i,],p,i)
    }
    GHCstats[(1+(j-1)*1000):(j*1000)] = apply(y2,2,max)
  }
  GHCstarvec = rep(0,p)
  for(i in 1:p){
    GHCstarvec[i] = f(sort(abs(S),decreasing=T)[i],p,i)
  }
  GHCstar = max(GHCstarvec)
  pval = length(which(GHCstats>=GHCstar))/length(GHCstats)
  return(pval)
}
