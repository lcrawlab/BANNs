#' BANN function for outer loop of all models to update free parameters and variance parameters.
#' @X is the genotype matrix with dimensionality n-by-p.
#' @y is the phenotype vector with length n.
#' @tau, sigma are variance parameters.
#' @logodds is the weight for fixed hyper-parameters.
#' @xy, @d, @I, @SIy, @SIX are precomputed helper statistics.
#' @alpha0, @mu0 are free parameters.
#' @update.order is the feature update order.
#' @tol is the tolerance for checking convergence.
#' @maxiter is the maximum iteration for updating parameters.
#' @outer.iter is the outer iter index
#' @example
#' out <- outerloop(X, I, y,xy,d,SIy,SIX,tau[i], sigma[i],logodds[,i],alpha[,i],mu[,i], update.order,tol,maxiter,i)
outerloop <- function(X, I, y, xy, d, SIy, SIX, tau,
                      sigma, logodds, alpha, mu, update.order, tol, maxiter,
                      outer.iter){
  #Number of features
  p <- ncol(X)
  if (length(logodds) == 1)
    logodds <- rep(logodds,p)
  #Call inner loop for updating parameters
  out <- innerloop(X,y,xy,d,tau,sigma,log(10)*logodds,alpha,mu,update.order,
                   tol,maxiter,outer.iter)
  #Subtract off the effect of the intercept and summarize results
  out$logw <- out$logw - determinant(crossprod(I),logarithm = TRUE)$modulus/2
  out$b <- c(with(out,SIy - SIX %*% (alpha*mu)))
  numiter  <- length(out$logw)
  out$logw <- out$logw[numiter]
  return(out)
}
