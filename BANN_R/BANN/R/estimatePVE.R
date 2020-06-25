#' BANN function for estimating proportiona of variance explained (PVE).
#' @fit stores the results for all the parameters of the neural network.
#' @X is the input matrix with dimensionality n by p.
#' @nr stores the number of iterations.
#' @example
#' pve = estimatePVE(fit, X)
estimatePVE <- function (fit, X, nr = 1000) {
  #number of feature
  p  <- ncol(X)
  #number of models fitted
  numModels <- length(fit$logw)
  #store each pve
  pve <- rep(0,nr)
  for (i in 1:nr) {
    j <- sample(numModels,1,prob = fit$w)
    b <- with(fit,mu[,j] + sqrt(s[,j]) * rnorm(p))
    b <- b * (runif(p) < fit$alpha[,j])

    sz     <- c(var1(X %*% (b)))
    pve[i] <- sz/(sz + (fit$tau[j]))
  }
  return(mean(pve))
}
