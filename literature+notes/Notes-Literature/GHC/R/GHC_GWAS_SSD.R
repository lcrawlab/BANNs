GHC_GWAS_SSD <-
function(SSD.Info,Y,X,list_restrict=NULL){
  N.Set = SSD.Info$nSets
  res.mat = matrix(NA,nrow=N.Set,ncol=3)  
  for(i in 1:N.Set){
    res.mat[i,1] = SSD.Info$SetInfo$SetID[i]
    if(is.null(list_restrict)==FALSE){
      if(length(intersect(res.mat[i,1],list_restrict))==0){
        next
      }
    }
    cat(res.mat[i,1],"\n")
    try1 = try(Get_Genotypes_SSD(SSD.Info,i),silent=TRUE)
    if(class(try1) != "try-error"){
      G = ImputeGenotypes(try1)
      G=as.matrix(G[,which(colMeans(G)/2>0.05)])
    } else{next}
    p=ncol(G)
    if(p==0){next}
    res.mat[i,2] = p
    objout=LinearModelScoreTest(Y,G,X)
    S = objout[[1]];Sigmamat=objout[[3]]
    try2 = try(GHC(S,Sigmamat),silent=TRUE)
    if(class(try2)!="try-error"){
      res.mat[i,3] = try2
    }
    ProgressBar(N.Set,i)
  }
  res.mat=res.mat[order(res.mat[,3]),]
  return(res.mat)
}
