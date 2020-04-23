CorrCountCDF <-
function(Mu_nb,Var_nb,cutoff,P,Peff,AVGLD,kmom,weight_matchp,indep_tests=F){
  if(indep_tests){return(pbinom(ceiling(cutoff)-1,P,Mu_nb/P))}
  bb.alpha= optimize(function(x) (P*x*((P*x-Mu_nb*x)/Mu_nb)*(x+P+(P*x-Mu_nb*x)/Mu_nb)/((x+(P*x-Mu_nb*x)/Mu_nb)^2*(x+1+(P*x-Mu_nb*x)/Mu_nb))-Var_nb)^2,lower=10^-49,upper=10^7)$minimum
  bb.beta = (P*bb.alpha-Mu_nb*bb.alpha)/Mu_nb
  return(pbetabinom.ab(ceiling(cutoff)-1,P,bb.alpha,bb.beta))
}
