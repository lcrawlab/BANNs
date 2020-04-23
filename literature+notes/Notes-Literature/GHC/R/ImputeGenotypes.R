ImputeGenotypes <-
function(G){
  p=ncol(G)
  n=nrow(G)
  inds=intersect(intersect(which(G!=0),which(G!=1)),which(G!=2))
  G[inds]=NA
  inds = which(is.na(G))
  if(length(inds)==0){return(G)}
  for(ind in inds){
    cind=ceiling(ind/n)
    rind=ind%%n
    if(ind%%n==0){rind=n}
    G[rind,cind] = rbinom(1,2,mean(G[,cind],na.rm=TRUE)/2)
  }
  return(G)
}
