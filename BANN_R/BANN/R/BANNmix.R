#' This program is free software: you can redistribute it under the
#' terms of the GNU General Public License; either version 3 of the
#' License, or (at your option) any later version.
#'
#' This program is distributed in the hope that it will be useful, but
#' WITHOUT ANY WARRANY; without even the implied warranty of
#' MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE. See the GNU
#' General Public License for more details.
#'
#' BANN function for free parameters and variance parameters updates with mixture model of three components.
#' @X is the input matrix of dimension n-by-p where n is the number of samples and p is the number of input variables (e.g., SNPs).
#' @y is the phenotype vector of length n.
#' @centered is used to check whether the X matrix is normalized.
#' @tol is the tolerance for checking convergence.
#' @maxiter is the maximum iteration for updating parameters.
#' @show_progress is the indicator for whether the software should print out the progress of the model fitting.
#' @example
#' SNP_res = BANNvarEM(X, y)
#' 
BANNmix<-function(X, y, numModels=20, tol = 1e-4, maxiter = 1e4, verbose = TRUE){
  n<-nrow(X)
  p<-ncol(X)
  update.order= 1:p
  I=matrix(1,n,1)
  SIy <- as.vector(solve(n,c(y) %*% I))
  SIX <- as.matrix(solve(n,t(I) %*% X))
  X <- X + I %*% SIX
  y <- y + c(I %*% SIy)
  
  xy <- drop(y %*% X)
  d  <- diagsq(X)
  
  # Initialize latent variables
  sigma<-var(y)
  sa<-20
  sa <- selectmixsd(xy/d,sa)^2
  K<-length(sa)
  w<-rep(1/K,K)
  w.penalty<-rep(1,K)
  alpha<- rand(p,K)+K*1e-8
  alpha<-alpha/rep.col(rowSums(alpha),K)
  mu<- randn(p,K)
  mu[,1] <- 0 #one of them will be at 0
  
  
  # For each variable and mixture component, calculate the variance of the regression coefficient conditioned on beingdrawn from the kth mixture component. The first column of "s" is always zero
  s <- matrix(0,p,K)
  for (i in 2:K){
    s[,i] <- sigma*sa[i]/(sa[i]*d + 1)
  }
  
  # Initialize storage for outputs logZ, err and nzw.
  logZ <- rep(0,maxiter)
  err  <- rep(0,maxiter)
  nzw  <- rep(0,maxiter)
  
  # Initialize the "inactives"; that is the mixture components with weights of zero:
  inactive   <- 1:K
  K0         <- K
  w0.penalty <- w.penalty
  sa0        <- sa
  
  # Initialize the Xr:
  Xr <- drop(X %*% rowSums(alpha*mu))
  
  for (iter in 1:maxiter) {
    # Save the current variational parameters and model parameters.
    alpha0 <- alpha
    mu0    <- mu
    s0     <- s
    sigma0 <- sigma
    w0     <- w
    
    # Compute the current variational lower bound to the marginal log-likelihood.
    logZ0 <- computevarlbmix(I,Xr,d,y,sigma,sa,w,alpha,mu,s)
    
    # Update variational parameters
    out   <- BANNmixUpdate(X,sigma,sa,w,xy,d,alpha,mu,Xr,update.order) ### 
    alpha <- out$alpha
    mu    <- out$mu
    Xr    <- out$Xr
    rm(out)
    
    # Compute the current variational lower bound to the marginal log-likelihood.
    logZ[iter] <- computevarlbmix(I,Xr,d,y,sigma,sa,w,alpha,mu,s)

    # Compute the approximate maximum likelihood estimate of the residual variable (sigma) and recalculate the variance of the regression coefficients
    sigma <- (norm2(y - Xr)^2 + dot(d,betavarmix(alpha,mu,s)) + sum(colSums(as.matrix(alpha[,-1]*(s[,-1] + mu[,-1]^2)))/sa[-1]))/(n + sum(alpha[,-1]))
      for (i in 2:K){
        s[,i] <- sigma*sa[i]/(sa[i]*d + 1)
      }
    
    # Compute the approximate penalized maximum likelihood estimate of the mixture weights (w)
    w <- colSums(alpha) + w.penalty - 1
    w <- w/sum(w)

    
    # (5f) CHECK CONVERGENCE
    # ----------------------
    # Print the status of the algorithm and check the convergence criterion. 
    # Convergence is reached when the maximum difference between the posterior mixture assignment probabilities at two
    # successive iterations is less than the specified tolerance, or when the variational lower bound has decreased.
    err[iter] <- max(abs(alpha - alpha0))
    nzw[iter] <- K0 - K
    if (verbose) {
      progress.str <- ####### CHANGE THIS!!!!
        sprintf("%04d %+13.6e %0.1e %0.1e %13s [%0.3f,%0.3f] (%d)",
                iter,logZ[iter],err[iter],sigma,
                sprintf("[%0.1g,%0.1g]",sqrt(min(sa[-1])),sqrt(max(sa))),
                min(w),max(w),nzw[iter])
      cat(progress.str,"\n")
    }
    if (logZ[iter] < logZ0) {
      logZ[iter] <- logZ0
      err[iter]  <- 0
      sigma      <- sigma0
      w          <- w0
      alpha      <- alpha0
      mu         <- mu0
      s          <- s0
      break
    } else if (err[iter] < tol)
      break
    
    # Check if any mixture components should be dropped based on threshold. Note that the first mixture component should never be droped.
    keep    <- apply(alpha,2,max) >= 1e-8
    keep[1] <- TRUE
    if (!all(keep)) {
      
    # At least one of the mixture components satisfy the criterion for being dropped, so adjust the inactive set.
      inactive  <- inactive[keep]
      sa        <- sa[keep]
      w         <- w[keep]
      w0        <- w0[keep]
      w.penalty <- w.penalty[keep]
      alpha     <- alpha[,keep]
      alpha0    <- alpha0[,keep]
      mu        <- mu[,keep]
      mu0       <- mu0[,keep]
      s         <- s[,keep]
      s0        <- s0[,keep]
      K         <- length(inactive)
    }
  }
  if (verbose)
    cat("\n")
  
  # (6) CREATE FINAL OUTPUT
  # -----------------------
  K   <- K0
  fit <- list(n = n,mu.cov = NULL,w.penalty = w0.penalty,drop.threshold = 1e-8,
              sigma = sigma,sa = sa0,w = rep(0,K),alpha = matrix(0,p,K), ### CHANGE!! ADD PIP HERE!!!
              mu = matrix(0,p,K),s = matrix(0,p,K),lfsr = NULL,
              logZ = logZ[1:iter],err = err[1:iter],nzw = nzw[1:iter])
  fit$w[inactive]      <- w
  fit$alpha[,inactive] <- alpha
  fit$mu[,inactive]    <- mu
  fit$s[,inactive]     <- s
  
  # Compute the posterior mean estimate of the regression
  # coefficients for the covariates under the current variational
  # approximation.
  fit$mu.cov <- c(SIy - SIX %*% rowSums(alpha * mu))
  
  # Compute the local false sign rate (LFSR) for each variable.
  fit$lfsr <- computelfsrmix(alpha,mu,s)
  
  # Add row names to some of the outputs.
  rownames(fit$alpha) <- colnames(X)
  rownames(fit$mu)    <- colnames(X)
  rownames(fit$s)     <- colnames(X)
  names(fit$lfsr)     <- colnames(X)
  
  class(fit) <- c("BANNsmix","list")
  return(fit)
}
 

  