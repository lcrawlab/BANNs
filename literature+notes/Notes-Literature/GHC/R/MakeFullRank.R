MakeFullRank <-
function(S,Sigmamat,maxcorr=0.95){
  if(length(S)<2){return(list(S,Sigmamat))}
  while(TRUE){
    ranks = sapply(1:ncol(Sigmamat),function(x) qr(Sigmamat[,-x])$rank)
    if(var(ranks)==0){break}
    indremove = order(ranks,decreasing=T)[1]
    S = S[-indremove]
    Sigmamat=Sigmamat[-indremove,-indremove]
  }
  while(TRUE){
    indrm=which(abs(Sigmamat)>maxcorr,arr.ind=T)
    indrm = indrm[-which(as.numeric(indrm[,1])-as.numeric(indrm[,2])==0),]
    if(nrow(indrm)==0){break}
    indrm = indrm[1]
    S=S[-indrm]
    Sigmamat=Sigmamat[-indrm,-indrm]
    if(length(S)==1){return(list(S,as.matrix(Sigmamat)))}
  }
  return(list(S,Sigmamat))
}
