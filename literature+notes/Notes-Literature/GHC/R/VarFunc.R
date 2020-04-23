VarFunc <-
function(t1,t2,P,rho_mom,len_Sigma,indep_tests){
  VarTerm=P*(2*pnorm(max(t1,t2),lower.tail=FALSE)-4*pnorm(t2,lower.tail=FALSE)*pnorm(t1,lower.tail=FALSE))
  H_sum = HermitePoly(t1,t2,rho_mom)
  CovTerm_noextra=2*(P*(P-1))*dnorm(t1)*dnorm(t2)*H_sum[1]
  CovTerm_extra=2*(P*(P-1))*dnorm(t1)*dnorm(t2)*H_sum[2]
  if(indep_tests){return(c(VarTerm,1))}
  return(c(VarTerm+CovTerm_noextra,(VarTerm+CovTerm_extra)/(VarTerm+CovTerm_noextra)))
}
