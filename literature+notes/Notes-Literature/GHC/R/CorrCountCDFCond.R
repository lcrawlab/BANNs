CorrCountCDFCond <-
function(t_0,t_1,c_0,c_1,V0,V1,V01,P,Peff,AVGLD,kmoms,weight_matchp,SR,indep_tests){
  if(c_1 >P){return(1)}
  if(c_0 <= 1){return(1)}
  if(ceiling(c_0-1)<=ceiling(c_1-1)){return(1)}
  ret_sum = 0 
  Mu_0 = P*2*pnorm(t_0,lower.tail=FALSE)
  Mu_1 = P*2*pnorm(t_1,lower.tail=FALSE)
  denom=CorrCountCDF(Mu_0,V0,ceiling(c_1-1)+.01,P,Peff,AVGLD,kmoms[1],weight_matchp,indep_tests)
  ret_sum=denom
  for(a in ceiling(c_1):min((ceiling(c_0)-1),P)){
    temp = CorrCountPDF(Mu_0,V0,a,P,Peff,AVGLD,kmoms[1],weight_matchp,indep_tests)
    denom = denom + temp
    #    ret_sum = ret_sum + pbinom(ceiling(c_1-1),a,Mu_1/Mu_0) * temp
    ret_sum = ret_sum + aCondProb(a,t_0,t_1,c_1,V0,V1,Mu_0,Mu_1,P,SR,PDF=FALSE,indep_tests) * temp
  }  
  return(ret_sum/denom)
}
