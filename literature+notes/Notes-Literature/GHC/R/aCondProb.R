aCondProb <-
function(a,t_0,t_1,c_1,V0,V1,Mu_0,Mu_1,P,SR,PDF = FALSE,indep_tests=F){  
  if(c_1 <= 0){return(0)}
  if(ceiling(c_1-1)>=a){return(1)}
  Mu = a*pnorm(t_1,lower.tail=FALSE)/pnorm(t_0,lower.tail=FALSE)
  if(a==1){return(1-Mu)}
  V = 2*a*(a-1)*SR/(P*(P-1)) + Mu-(Mu)^2
  if(indep_tests || V<Mu){
    if(PDF){
      return(dbinom(c_1,a,Mu/a))
    } else{
      return(pbinom(ceiling(c_1-1),a,Mu/a))
    }
  }
  bb.alpha = optimize(function(aa) (a*aa*(aa*(a-Mu)/Mu)*(aa+(aa*(a-Mu)/Mu)+a)/((aa+(aa*(a-Mu)/Mu))^2*(aa+(aa*(a-Mu)/Mu)+1)) - V)^2,lower=10^-49,upper=10^7)$minimum
  bb.beta = bb.alpha*(a-Mu)/Mu
  if(PDF){return(dbetabinom.ab(c_1,a,bb.alpha,bb.beta))}
  return(pbetabinom.ab(ceiling(c_1-1),a,bb.alpha,bb.beta))
}
