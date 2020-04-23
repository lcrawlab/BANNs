CondBinProb <-
function(t_1,t_0,c_1,c_0,P){
  if(ceiling(c_1)==ceiling(c_0)){return(1)}
  tailprob0 = 2*pnorm(t_0,lower.tail=FALSE)
  tailprob1 = 2*pnorm(t_1,lower.tail=FALSE)
  denom = pbinom(ceiling(c_0-1),P,tailprob0)
  numer=pbinom(ceiling(c_1)-1,P,tailprob0)
  for(j in ceiling(c_1):ceiling(c_0-1)){
    numer = numer + pbinom(ceiling(c_1-1),j,tailprob1/tailprob0)*dbinom(j,P,tailprob0)
  }
  return(numer/denom)
}
