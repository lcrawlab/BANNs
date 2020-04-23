PM_update <-
function(ind,t_mesh,cv,PM,tailprob,V,SRv,P){
  denom=sum(PM[1:(cv[ind-1]+1),ind-1])
  for(a_k in 0:cv[ind]){
    for(a_k1 in 0:cv[ind-1]){
      #      PM[a_k+1,ind] = PM[a_k+1,ind] + dbinom(a_k,a_k1,tailprob[ind]/tailprob[ind-1])*PM[a_k1+1,ind-1]
      Mu=a_k1*tailprob[ind]/tailprob[ind-1]
      V = 2*a_k1*(a_k1-1)*SRv[ind-1]/(P*(P-1)) + Mu-(Mu)^2
      PM[a_k+1,ind] = PM[a_k+1,ind] + CountPDF(a_k,a_k1,Mu,V)*PM[a_k1+1,ind-1]
    }
  }
  PM[1:(cv[ind]+1),ind] = PM[1:(cv[ind]+1),ind]/denom  
  return(PM)
}
