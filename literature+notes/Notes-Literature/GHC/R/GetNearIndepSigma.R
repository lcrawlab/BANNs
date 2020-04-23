GetNearIndepSigma <-
function(Sigmamat,n,nsum=51){
  U = solve(chol(Sigmamat))
  matall = rWishart(nsum,n,Sigmamat)
  rho2mean=rep(0,nsum)
  for(i in 1:nsum){
    mat=matall[,,i]/n
    Sigadj = t(U)%*%mat%*%U
    Sigmaadj=Sigadj[lower.tri(Sigadj)]    
    rho2mean[i]=mean(Sigmaadj^2)
  }
  return(t(U)%*%(matall[,,order(rho2mean)[(nsum+1)/2]]/n)%*%U)
}
