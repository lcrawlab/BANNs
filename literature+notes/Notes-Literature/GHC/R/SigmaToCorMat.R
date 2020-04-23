SigmaToCorMat <-
function(Sigma,P,w=NULL){
  if(is.null(w)){
    w=P-1
  }
  ret_mat = matrix(0,nrow=P,ncol=P)
  for(i in 1:w){
    for(j in 1:(P-i)){
      ret_mat[j+i,j] = Sigma[sum((P-i+1):P)-P+j]
    }
  }
  ret_mat = ret_mat + t(ret_mat) + diag(P)
  return(ret_mat)
}
