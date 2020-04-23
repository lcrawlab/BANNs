CorrCountCDFCond2 <-
function(t_0,t_1,t_2,c_0,c_1,c_2,V0,V1,V2,V01,V12,P,Peff,AVGLD,kmoms,weight_matchp,SR01,SR12,indep_tests){
  if(ceiling(c_2)>min(P,ceiling(c_1)-1)){return(1)}
  ret_sum=0;numer=0
  Mu_0 = P*2*pnorm(t_0,lower.tail=FALSE)
  Mu_1 = P*2*pnorm(t_1,lower.tail=FALSE)
  Mu_2 = P*2*pnorm(t_2,lower.tail=FALSE)
  numer=0
  for(a_1 in ceiling(c_2):min(P,ceiling(c_1)-1)){
    temp = 0
    for(a_0 in a_1:min(P,ceiling(c_0)-1)){
      temp = temp + aCondProb(a_0,t_0,t_1,a_1,V0,V1,Mu_0,Mu_1,P,SR01,PDF=TRUE,indep_tests)*CorrCountPDF(Mu_0,V0,a_0,P,Peff,AVGLD,kmoms[1],weight_matchp,indep_tests)
    }
    numer = numer+temp*aCondProb(a_1,t_1,t_2,c_2,V1,V2,Mu_1,Mu_2,P,SR12,PDF=FALSE,indep_tests)
  }
  for(a_0 in 0:min(P,ceiling(c_0)-1)){
    numer = numer + aCondProb(a_0,t_0,t_1,c_2,V0,V1,Mu_0,Mu_1,P,SR01,PDF=FALSE,indep_tests)*CorrCountPDF(Mu_0,V0,a_0,P,Peff,ANGLD,kmoms[1],weight_matchp,indep_tests)
  }
  denom = 0
  for(a_0 in 0:min(P,ceiling(c_0)-1)){
    denom = denom + aCondProb(a_0,t_0,t_1,c_1,V0,V1,Mu_0,Mu_1,P,SR01,PDF=FALSE,indep_tests)*CorrCountPDF(Mu_0,V0,a_0,P,Peff,ANGLD,kmoms[1],weight_matchp,indep_tests)
  }
  return(numer/denom)
}
