GetVarMat <-
function(t_mesh,P,rho_mom,len_Sigma,correct_sum,indep_tests){
  m=length(t_mesh)
  quotient = rep(0,m)
  V = matrix(0,nrow=m,ncol=m)
  for(i in 1:m){
    for(j in i:m){
      retval=VarFunc(t_mesh[i],t_mesh[j],P,rho_mom,len_Sigma,indep_tests)
      if(i==j){quotient[i]=retval[2]}
      V[i,j] = retval[1]
      V[j,i] = V[i,j]
    }
  }
  if(correct_sum){
    return(V*mean(quotient))    
  }
  else{
    return(V)    
  }
}
