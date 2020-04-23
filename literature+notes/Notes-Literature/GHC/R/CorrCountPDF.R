CorrCountPDF <-
function(Mu_nb,Var_nb,a,P,Peff,AVGLD,kmom,weight_matchp,indep_tests=F){
  if(indep_tests){return(dbinom(a,P,Mu_nb/P))}
  bb.alpha= optimize(function(x) (P*x*((P*x-Mu_nb*x)/Mu_nb)*(x+P+(P*x-Mu_nb*x)/Mu_nb)/((x+(P*x-Mu_nb*x)/Mu_nb)^2*(x+1+(P*x-Mu_nb*x)/Mu_nb))-Var_nb)^2,lower=10^-49,upper=10^5)$minimum
  bb.beta = (P*bb.alpha-Mu_nb*bb.alpha)/Mu_nb
  return(dbetabinom.ab(a,P,bb.alpha,bb.beta))
}
