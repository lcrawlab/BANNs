#' BANN function for inner loop within each one model to update free parameters and variance parameters.
#' @X is the genotype matrix with dimensionality n by p.
#' @y is the phenotype vector with length n.
#' @tau, @sigma are variance parameters.
#' @logodds is the weight for fixed hyper-paran.
#' @xy, @d, @Xr0 are precomputed helper statistics.
#' @alpha0, @mu0 are free parameters.
#' @update.order is the feature update order.
#' @tol is the tolerance for checking convergence.
#' @maxiter is the maximum iteration for updating parameters.
#' @outer.iter is the outer iter index.
#' @example
#' out <- innerloop(X,y,xy,d,tau,sigma,log(10)*logodds,alpha,mu,update.order,tol,maxiter,outer.iter)
innerloop <-function (X, y,xy,d, tau, sigma, logodds, alpha, mu, update.order,
                      tol = 1e-4, maxiter = 1e4,
                      outer.iter = NULL){
  n<-nrow(X)
  p<-ncol(X)
  #Precomputed statistics
  Xr <- c(X %*% (alpha*mu))
  #Free parameter: variance for individual features
  s <- sigma*tau/(sigma*d + 1)
  logw <- rep(0,maxiter)
  err  <- rep(0,maxiter)
  sigma0 = 1
  n0 = 10
  #Loop until converge
  for (iter in 1:maxiter) {

    #Save the current variational and model parameters.
    alpha0 <- alpha
    mu0    <- mu
    s0     <- s
    tau0 <- tau
    sigma.old <- sigma

    #Current variational lowerbound
    logw0 <- varLoss(Xr,d,y,tau,alpha,mu,s,tau*sigma, logodds)
    out   <- varParamUpdate(X,tau,sigma,logodds,xy,d,alpha,mu,Xr,update.order)
    alpha <- out$alpha
    mu    <- out$mu
    Xr    <- out$Xr
    rm(out)

    #Variational lowerbound after updates
    logw[iter] <- varLoss(Xr,d,y,tau,alpha,mu,s,tau*sigma, logodds)
    #print(paste("LB:",logw[iter]))

    #Update variance paraemters
    tau <- (norm2(y - Xr)^2 + dot(d,betavar(alpha,mu,s)) +
              dot(alpha,(s + mu^2)/sigma))/(n + sum(alpha))
    s     <- sigma*tau/(sigma*d + 1)
    sigma <- (sigma0*n0 + dot(alpha,s + mu^2))/(n0 + tau*sum(alpha))
    s  <- sigma*tau/(sigma*d + 1)

    # Check the convergence with posterior mean (stop if change is smaller than the tolerance) or the lower bound (stop if start decreasing after first update)
    err[iter] <- max(abs(alpha - alpha0))
    if (logw[iter] < logw0  & iter > 2) {
      logw[iter] <- logw0
      err[iter]  <- 0
      tau      <- tau0
      sigma        <- sigma.old
      alpha      <- alpha0
      mu         <- mu0
      s          <- s0
      break
    } else if (err[iter] < tol & iter > 2)
    {
      break
    }
  }
  #Return results
  return(list(logw = logw[1:iter],err = err[1:iter],tau = tau,sigma = sigma,
              alpha = alpha,mu = mu,s = s))
}
