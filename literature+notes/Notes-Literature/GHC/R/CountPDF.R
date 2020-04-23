CountPDF <-
function(a,P,Mu1,Var){
  if(P<=1){return(dbinom(a,P,Mu1))}
  Mu2 = Var + Mu1^2
  bb.alpha = (P*Mu1-Mu2)/(P*(Mu2/Mu1-Mu1-1)+Mu1) 
  bb.beta = (P-Mu1)*(P-Mu2/Mu1)/(P*(Mu2/Mu1-Mu1-1)+Mu1)
  return(dbetabinom.ab(a,P,bb.alpha,bb.beta))
}
