#' BANN function for updating free parameters and variance parameters.
#' @X is the genotype matrix with dimensionality n-by-p.
#' @tau, @sigma are variance parameters.
#' @logodds is the weight for fixed hyper-param,
#' @xy, @d, @Xr0 are precomputed helper statistics.
#' @alpha0, @mu0 are free parameters.
#' @update is the update order.
#' out   <- varParamUpdate(X,tau,sigma,logodds,xy,d,alpha,mu,Xr,update.order)
varParamUpdate <- function (X, tau, sigma, logodds, xy, d, alpha0, mu0, Xr0, updates) {
  #Number of individuals
  n <- nrow(X)
  #Number of features
  p <- ncol(X)

  #Initialize storage:
  alpha <- c(alpha0)
  mu    <- c(mu0)
  Xr    <- c(Xr0)

  #Loop for updating free parameters and Xr
  for (j in updates) {
    s <- sigma*tau/(sigma*d[j] + 1)
    r     <- alpha[j] * mu[j]
    mu[j] <- s/tau * (xy[j] + d[j]*r - sum(X[,j]*Xr))
    alpha[j] <- sigmoid(logodds[j] + (log(s/(sigma*tau)) + mu[j]^2/s)/2)
    Xr <- Xr + (alpha[j]*mu[j] - r) * X[,j]
  }
  #Return results
  return(list(alpha = alpha,mu = mu,Xr = Xr))
}
