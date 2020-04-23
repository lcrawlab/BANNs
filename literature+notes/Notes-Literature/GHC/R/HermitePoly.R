HermitePoly <-
function(t1,t2,rho_mom){
  Hpoly_t1_prev=1
  Hpoly_t2_prev=1
  Hpoly_t1 = t1
  Hpoly_t2 = t2
  terms = rep(0,length(rho_mom))
  for(i in 1:length(rho_mom)){
    terms[i] = 2*Hpoly_t1*Hpoly_t2*rho_mom[i]/factorial(2*i)
    temp = Hpoly_t1
    Hpoly_t1 = Hpoly_t1*t1-(2*i-1)*Hpoly_t1_prev
    Hpoly_t1_prev = temp
    temp = Hpoly_t1
    Hpoly_t1 = Hpoly_t1*t1-(2*i)*Hpoly_t1_prev
    Hpoly_t1_prev = temp    
    temp = Hpoly_t2
    Hpoly_t2 = Hpoly_t2*t2-(2*i-1)*Hpoly_t2_prev
    Hpoly_t2_prev = temp
    temp = Hpoly_t2
    Hpoly_t2 = Hpoly_t2*t2-(2*i)*Hpoly_t2_prev
    Hpoly_t2_prev = temp
  }
  #  retvals = ExtrapolateEndTerms_HermitePoly(terms)
  retvals = c(0,0)
  return(c(sum(terms)+retvals[1],sum(terms)+retvals[2]))  
}
