ExtrapolateEndTerms_HermitePoly <-
function(y){
  x = 1:length(y)
  a=as.numeric(uniroot(function(a) sum(y/((x-a)^2))/sum(1/((x-a)^4))-sum(y/((x-a)^3))/sum(1/((x-a)^5)),interval=c(-2000,1000))[1])
  b=sum(y/((x-a)^2))/sum(1/((x-a)^4))
  return(c(0,b/(length(y)-a)))
}
