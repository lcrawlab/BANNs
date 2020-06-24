#' BANN function for free parameters and variance parameters updates.
#' @X is the input matrix with n by p.
#' @y is the phenotype vector with length n.
#' @centered is used to check whether the X is normalized.
#' @tol is the tolerance for checking convergence.
#' @maxiter is the maximum iteration for updating parameters.
#' @show_progress is the indicator for whether printing out the progress.
#' @example
#' SNP_res = BANNvarEM(X, y)
BANNvarEM<-function(X, y, centered=FALSE, numModels=20, tol = 1e-4,maxiter = 1e4, show_progress = TRUE){
  #normalizing data
  if (centered==FALSE){
    X <- scale(X,center = TRUE,scale = TRUE)
    y <- (y - mean(y))/sd(y)
  }
  #num of inds and features
  n=nrow(X)
  p=ncol(X)
  #precomputed statistics
  xy <- c(y %*%X)
  d  <- diagsq(X)
  # Initialize latent variables:
  tau=rep(var(y),numModels)
  sigma=rep(1,numModels)
  logodds=t(matrix(seq(-log10(p), -1, length.out=numModels)))
  alpha=rand(p,numModels)
  alpha=alpha/rep.row(colSums(alpha),p)
  mu=randn(p, numModels)
  update.order= 1:p

  #start storage for the optimization params:
  logw   <- rep(0, numModels)
  s      <- matrix(0,p,numModels)
  b <- matrix(0,1,numModels)

  ### For intercept, b:
  I=matrix(1,n,1)
  SIy <- as.vector(solve(n,c(y) %*% I))
  SIX <- as.matrix(solve(n,t(I) %*% X))

  #Finding the best initialization of hyperparameters
  for (i in 1:numModels) {
    if(show_progress == TRUE){
      print(paste("Initialize model ", i, "/", numModels, sep=""))
    }
    out <- outerloop(X, I, y,xy,d,SIy,SIX,tau[i],
                     sigma[i],logodds[,i],alpha[,i],mu[,i],
                     update.order,tol,maxiter,i)

    logw[i]    <- out$logw
    tau[i]   <- out$tau
    sigma[i]      <- out$sigma
    b[,i] <- out$b
    alpha[,i]  <- out$alpha
    mu[,i]     <- out$mu
    s[,i]      <- out$s
  }

  i     <- which.max(logw)
  alpha <- rep.col(alpha[,i],numModels)
  mu    <- rep.col(mu[,i],numModels)
  tau <- rep(tau[i],numModels)
  sigma <- rep(sigma[i],numModels)

  #Computing marginal likelihood
  for (i in 1:numModels) {
    if(show_progress == TRUE){
      print(paste("Updating model ", i, "/", numModels, sep=""))
    }
    out <- outerloop(X,I,y,xy,d,SIy,SIX,tau[i],
                     sigma[i],logodds[,i],alpha[,i],mu[,i],
                     update.order,tol,maxiter,i)
    logw[i]    <- out$logw
    tau[i]   <- out$tau
    sigma[i]      <- out$sigma
    b[,i] <- out$b
    alpha[,i]  <- out$alpha
    mu[,i]     <- out$mu
    s[,i]      <- out$s
  }

  w        <- normalizelogweights(logw)
  pip      <- c(alpha %*% w)
  beta     <- c((alpha*mu) %*% w)
  beta.cov <- c(b %*% w)
  fit <- list(b = b,
              logw = logw, w = w, tau = tau, sigma = sigma, logodds = logodds,alpha = alpha,
              mu = mu,s = s,pip = pip,beta = beta, beta.cov = beta.cov,y = y)
  class(fit) <- c("BANNvarEM","list")
  #estimate pve
  fit$model.pve <- estimatePVE(fit,X)
  fit$pve           <- matrix(0,p,numModels)
  rownames(fit$pve) <- colnames(X)
  sx                <- var1.cols(X)
  for (i in 1:numModels){
    fit$pve[,i] <- sx*(mu[,i]^2 + s[,i])/var1(y)
  }
  X <- X + I %*% SIX
  y <- y + c(I %*% SIy)
  fit$fitted.values <- linear.predictors(X,I,b,alpha,mu)
  fit$residuals <- y - fit$fitted.values
  fit$residuals.response
  hyper.labels                = paste("theta",1:numModels,sep = "_")
  rownames(fit$alpha)         = colnames(X)
  rownames(fit$mu)            = colnames(X)
  rownames(fit$s)             = colnames(X)
  names(fit$beta)             = colnames(X)
  names(fit$pip)              = colnames(X)
  rownames(fit$b)        = colnames(I)
  names(fit$beta.cov)         = colnames(I)
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
