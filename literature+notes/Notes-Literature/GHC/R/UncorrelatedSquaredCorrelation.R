UncorrelatedSquaredCorrelation <-
function(N,num_iter){
  cor_vec = rep(0,num_iter)
  for(i in 1:num_iter){
    cor_vec[i]=sqrt((cor(rnorm(N),rnorm(N)))^2)
  }
  return(mean(cor_vec))
}
