GetCondVarTerms <-
function(Sigma,tm,numterms=40){
  m=length(tm)
  s_v = pnorm(tm,lower.tail=FALSE)
  p_v = dnorm(tm)
  retval = rep(0,m)
  Hmat = GetHermiteTermsCond(tm,numterms=numterms)^2
  Smat = matrix(,nrow=numterms,ncol=length(Sigma))
  for(i in 1:numterms){
    Smat[i,] = Sigma^(2*i)/factorial(2*i)
  }
  HSmat = Hmat %*% Smat
  ratvals = rep(0,m)
  for(j in 2:m){
    ratvals[j] = sum((s_v[j]^2+p_v[j]^2*(HSmat[j,]))/(s_v[j-1]^2+p_v[j-1]^2*(HSmat[j-1,])))
  }
  return(ratvals[-1])
}
