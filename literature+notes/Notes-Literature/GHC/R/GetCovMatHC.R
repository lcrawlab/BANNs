GetCovMatHC <-
function(V){
  m = nrow(V)
  C = matrix(0,nrow=m,ncol=m)
  for(i in 1:m){
    for(j in i:m){
      C[i,j]=V[i,j]/sqrt(V[i,i]*V[j,j])
      C[j,i]=C[i,j]
    }
  }
  return(C)
}
