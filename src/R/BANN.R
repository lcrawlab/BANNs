
#########################################################################################################
########################################## LOAD IN DATA #################################################
#########################################################################################################
library(Matrix)
library(doParallel)
library(foreach)
source("/Users/pinardemetci/Desktop/BANN_dev/utils.R")
cores=detectCores()
registerDoParallel(cores)

X=as.matrix(read.csv("/Users/pinardemetci/Desktop/X_varbvsTOY.csv", sep=",", header=FALSE))
y=as.numeric(unlist(read.csv("/Users/pinardemetci/Desktop/y_varbvsTOY.csv", sep=",", header=FALSE)))
mask=as.matrix(read.csv("/Users/pinardemetci/Desktop/maskTOY.csv", sep=",", header=FALSE))
X=X[,1:100]
print(dim(X))
print(length(y))

#########################################################################################################
######################################### HELPER FUCTIONS ###############################################
#########################################################################################################

varParamUpdate <- function (X, sigma, sa, logodds, xy, d, alpha0, mu0, Xr0, updates) {
  n <- nrow(X)
  p <- ncol(X)
  
  # Check inputs sigma and sa.
  if (length(sigma) != 1 | length(sa) != 1)
    stop("Inputs sigma and sa must be scalars")
  
  # Check input Xr0.
  if (length(Xr0) != n)
    stop("length(Xr0) must be equal to nrow(X)")
  
  # Check input "updates".
  if (sum(updates < 1 | updates > p) > 0)
    stop("Input \"updates\" contains invalid variable indices")
  
  # Initialize storage for the results.
  alpha <- c(alpha0)
  mu    <- c(mu0)
  Xr    <- c(Xr0)
  
  
  for (j in updates) {
    
    # Compute the variational estimate of the posterior variance.
    s <- sa*sigma/(sa*d[j] + 1)
    
    # Update the variational estimate of the posterior mean.
    r     <- alpha[j] * mu[j]
    mu[j] <- s/sigma * (xy[j] + d[j]*r - sum(X[,j]*Xr))
    
    # Update the variational estimate of the posterior inclusion
    # probability.
    alpha[j] <- sigmoid(logodds[j] + (log(s/(sa*sigma)) + mu[j]^2/s)/2)
    
    # Update Xr = X*r.
    Xr <- Xr + (alpha[j]*mu[j] - r) * X[,j]
  }
  return(list(alpha = alpha,mu = mu,Xr = Xr))
}

innerLoop <-function (X, y, sigma, sa, logodds, alpha, mu, update.order,
                      tol = 1e-4, maxiter = 1e4,
                      outer.iter = NULL, n0 = 10, sa0 = 1){
  n<-nrow(X)
  p<-ncol(X)
  
  xy <- c(y %*%X)
  d  <- diagsq(X)
  Xr <- c(X %*% (alpha*mu))
  s <- sa*sigma/(sa*d + 1)
  logw <- rep(0,maxiter)
  err  <- rep(0,maxiter)
  
  for (iter in 1:maxiter) {
    
    # Save the current variational and model parameters.
    alpha0 <- alpha
    mu0    <- mu
    s0     <- s
    sigma0 <- sigma
    sa.old <- sa
    
    # COMPUTE CURRENT VARIATIONAL LOWER BOUND
    # --------------------------------------------
    # Compute the lower bound to the marginal log-likelihood.
    logw0 <- int.linear(Xr,d,y,sigma,alpha,mu,s) +
      int.gamma(logodds,alpha) +
      int.klbeta(alpha,mu,s,sigma*sa)
    # UPDATE VARIATIONAL APPROXIMATION
    # -------------------------------------
    out   <- varParamUpdate(X,sigma,sa,logodds,xy,d,alpha,mu,Xr,update.order)
    alpha <- out$alpha
    mu    <- out$mu
    Xr    <- out$Xr
    rm(out)
    
    # COMPUTE UPDATED VARIATIONAL LOWER BOUND
    # --------------------------------------------
    # Compute the lower bound to the marginal log-likelihood.
    logw[iter] <- int.linear(Xr,d,y,sigma,alpha,mu,s) +
      int.gamma(logodds,alpha) +
      int.klbeta(alpha,mu,s,sigma*sa)
    
    # UPDATE RESIDUAL VARIANCE
    # -----------------------------
    # Compute the maximum likelihood estimate of sigma, if requested.
    # Note that we must also recalculate the variance of the regression
    # coefficients when this parameter is updated.
    sigma <- (norm2(y - Xr)^2 + dot(d,betavar(alpha,mu,s)) +
                dot(alpha,(s + mu^2)/sa))/(n + sum(alpha))
    s     <- sa*sigma/(sa*d + 1)
    
    # (2e) UPDATE PRIOR VARIANCE OF REGRESSION COEFFICIENTS
    # -----------------------------------------------------
    # Compute the maximum a posteriori estimate of sa, if requested.
    # Note that we must also recalculate the variance of the
    # regression coefficients when this parameter is updated.
    sa <- (sa0*n0 + dot(alpha,s + mu^2))/(n0 + sigma*sum(alpha))
    s  <- sa*sigma/(sa*d + 1)
    
    # (2f) CHECK CONVERGENCE
    # ----------------------
    # Print the status of the algorithm and check the convergence
    # criterion. Convergence is reached when the maximum difference
    # between the posterior probabilities at two successive iterations
    # is less than the specified tolerance, or when the variational
    # lower bound has decreased.
    err[iter] <- max(abs(alpha - alpha0))
    #print(logw[iter])
    #print(logw0)
    if (logw[iter] < logw0) {
      logw[iter] <- logw0
      err[iter]  <- 0
      sigma      <- sigma0
      sa         <- sa.old
      alpha      <- alpha0
      mu         <- mu0
      s          <- s0
      break
    } else if (err[iter] < tol)
      break
  }
  
  return(list(logw = logw[1:iter],err = err[1:iter],sigma = sigma,sa = sa,
              alpha = alpha,mu = mu,s = s))
}

estimatePVE <- function (fit, X, nr = 1000) {
  
  p  <- ncol(X)
  numModels <- length(fit$logw)
  
  pve <- rep(0,nr)
  
  # For each sample, compute the proportion of variance explained.
  for (i in 1:nr) {
    j <- sample(numModels,1,prob = fit$w)
    b <- with(fit,mu[,j] + sqrt(s[,j]) * rnorm(p))
    b <- b * (runif(p) < fit$alpha[,j])
    sz     <- c(var1(X %*% (b)))
    pve[i] <- sz/(sz + (fit$sigma[j]))
    pve_estimate=mean(pve)
  }
  
  return(pve_estimate)
}

outerloop <- function(X, Z, y, SZy, SZX, sigma,
                      sa, logodds, alpha, mu, update.order, tol, maxiter,
                      outer.iter,n0, sa0){
  p <- ncol(X)
  if (length(logodds) == 1)
    logodds <- rep(logodds,p)
  out <- innerLoop(X,y,sigma,sa,log(10)*logodds,alpha,mu,update.order,
                   tol,maxiter,outer.iter,
                   n0,sa0)
  out$logw <- out$logw - determinant(crossprod(Z),logarithm = TRUE)$modulus/2

  out$b <- c(with(out,SZy - SZX %*% (alpha*mu)))
  numiter  <- length(out$logw)
  out$logw <- out$logw[numiter]
  return(out)
}

BANNvarEM<-function(X, y, centered=FALSE, numModels=20, nr = 100, sa0 = 1, n0 = 10, tol = 1e-4,maxiter = 1e4){
  n=nrow(X)
  p=ncol(X)
  
  Z=matrix(1,n,1)
  SZy <- as.vector(solve(n,c(y) %*% Z))
  SZX <- as.matrix(solve(n,t(Z) %*% X))
  
  if (centered==FALSE){
    X <- scale(X,center = TRUE,scale = FALSE)
    y <- y - mean(y)
  }
  if (is.null(rownames(X)))
    rownames(X) <- 1:n
  if (is.null(colnames(X)))
    colnames(X) <- paste0("X",1:p)
  colnames(Z)[1] <- "(Intercept)"
  
  ### Initialize latent variables:
  sigma=rep(var(y),numModels)
  sa=rep(1,numModels)
  logodds=t(matrix(seq(-log10(p), -1, length.out=numModels)))
  alpha=rand(p,numModels)
  alpha=alpha/rep.row(colSums(alpha),p)
  mu=randn(p, numModels)
  update.order= 1:p
  
  #initialize storage for the outputs
  logw   <- rep(0, numModels)
  s      <- matrix(0,p,numModels)
  b <- matrix(0,ncol(Z),numModels)
  
  print("Finding the best initialization of hyperparameters")
  for (i in 1:numModels) {
    out <- outerloop(X, Z, y,SZy,SZX,sigma[i],
                     sa[i],logodds[,i],alpha[,i],mu[,i],
                     update.order,tol,maxiter,i,n0,sa0)
    
    logw[i]    <- out$logw
    sigma[i]   <- out$sigma
    sa[i]      <- out$sa
    b[,i] <- out$b
    alpha[,i]  <- out$alpha
    mu[,i]     <- out$mu
    s[,i]      <- out$s
  }
  
  i     <- which.max(logw)
  alpha <- rep.col(alpha[,i],numModels)
  mu    <- rep.col(mu[,i],numModels)
  sigma <- rep(sigma[i],numModels)
  sa <- rep(sa[i],numModels)

  print("Computing marginal likelihood")
  for (i in 1:numModels) {
    out <- outerloop(X,Z,y,SZy,SZX,sigma[i],
                     sa[i],logodds[,i],alpha[,i],mu[,i],
                     update.order,tol,maxiter,i,n0,sa0)
    logw[i]    <- out$logw
    sigma[i]   <- out$sigma
    sa[i]      <- out$sa
    b[,i] <- out$b
    alpha[,i]  <- out$alpha
    mu[,i]     <- out$mu
    s[,i]      <- out$s
  }
  
  w        <- normalizelogweights(logw)
  pip      <- c(alpha %*% w)
  beta     <- c((alpha*mu) %*% w)
  beta.cov <- c(b %*% w)
  
  fit <- list(n0 = n0,sa0 = sa0,b = b,
              logw = logw, w = w, sigma = sigma, sa = sa, logodds = logodds,alpha = alpha,
              mu = mu,s = s,pip = pip,beta = beta, beta.cov = beta.cov,y = y)
  class(fit) <- c("varEM","list")
  fit$model.pve <- estimatePVE(fit,X,nr)
  
  fit$pve           <- matrix(0,p,numModels)
  rownames(fit$pve) <- colnames(X)
  sx                <- var1.cols(X)
  for (i in 1:numModels)
    fit$pve[,i] <- sx*(mu[,i]^2 + s[,i])/var1(y)
  X <- X + Z %*% SZX
  y <- y + c(Z %*% SZy)
  fit$fitted.values <- linear.predictors(X,Z,b,alpha,mu)
  fit$residuals <- y - fit$fitted.values
  fit$residuals.response
  
  hyper.labels                = paste("theta",1:numModels,sep = "_")
  rownames(fit$alpha)         = colnames(X)
  rownames(fit$mu)            = colnames(X)
  rownames(fit$s)             = colnames(X)
  names(fit$beta)             = colnames(X)
  names(fit$pip)              = colnames(X)
  rownames(fit$b)        = colnames(Z)
  names(fit$beta.cov)         = colnames(Z)
  rownames(fit$fitted.values) = rownames(X)
  colnames(fit$b)        = hyper.labels
  colnames(fit$alpha)         = hyper.labels
  colnames(fit$mu)            = hyper.labels
  colnames(fit$s)             = hyper.labels
  colnames(fit$fitted.values) = hyper.labels
  rownames(fit$residuals) = rownames(X)
  colnames(fit$residuals) = hyper.labels
  fit$logodds = c(fit$logodds)
  colnames(fit$pve) = hyper.labels
  return(fit)
}

N=nrow(X)
p=ncol(X)
g=ncol(mask)




BANN <-function(X, mask, y, centered=FALSE, numModels=20, tol = 1e-4,maxiter = 1e4){
  ########  Input data quality control ###############
  
  ### Check input variables:
  if (!(is.matrix(X)) | !(is.numeric(X)) | sum(is.na(X))> 0){
    stop("The genotype data passed in, X, needs to be a numeric matrix")
  }
  if (!(is.numeric(y)) | sum(is.na(y))>0){
    stop("The phenotype data passed in, y, needs to be a numeric vector with the same number of samples (columns) as X, the genotype matrix")
  }
  
  if (!( (is.matrix(X)) & (is.numeric(X)) & (sum(is.na(X))==0)) ){
    stop("The genotype data passed in, X, needs to be an all numeric matrix")
  }
  if (!(is.finite(tol)& (tol>0))){
    stop("tol parameter needs to be a finite positive number")
  }
  SNP_res<-BANNvarEM(X, y, centered=TRUE)
  G=(X*(rep.row(SNP_res$beta,N)))
  G=softplus(G%*%mask)
  SNPset_res<-BANNvarEM(G, y, centered=TRUE)
  SNP_pve<-mean(SNP_res$model.pve)
  SNPset_pve<-mean(SNPset_res$model.pve)
  print("Estimated SNP level heritability:")
  print(SNP_pve)
  print("Estimated SNP-set level heritability:")
  print(SNPset_pve)
}

BANN(X, mask, y, centered=TRUE, numModels=20, tol = 1e-4,maxiter = 10000)
