PM_updateindep <-
function(ind,t_mesh,cv,PM,tailprob){
  denom=sum(PM[1:(cv[ind-1]+1),ind-1])
  for(a_k in 0:cv[ind]){
    for(a_k1 in 0:cv[ind-1]){
      PM[a_k+1,ind] = PM[a_k+1,ind] + dbinom(a_k,a_k1,tailprob[ind]/tailprob[ind-1])*PM[a_k1+1,ind-1]
    }
  }
  PM[1:(cv[ind]+1),ind] = PM[1:(cv[ind]+1),ind]/denom  
  return(PM)
}
