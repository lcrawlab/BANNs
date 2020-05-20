# @Author: Pinar Demetci, February 2020
# Crawford Lab, Brown University CCMB
# Variational EM optimization for BANN framework

#########################################################################################################
########################################## LOAD IN DATA #################################################
#########################################################################################################
library(Matrix)
library(foreach)
library(doParallel)

cores = detectCores()
registerDoParallel(cores=cores)

path_to_utils="/Users/pinardemetci/Desktop/deneme/BANNs/utils.R"
Xfilepath="/Users/pinardemetci/Desktop/deneme/BANNs/X_varEMTOY.csv"
yfilepath="/Users/pinardemetci/Desktop/deneme/BANNs/y_varEMTOY.csv"
maskFile="/Users/pinardemetci/Desktop/deneme/BANNs/maskTOY3.csv"
#outputfile="/Users/pinardemetci/Desktop/BANN_Res.RData"

source(path_to_utils) 
X=as.matrix(read.csv(Xfilepath, sep=",", header=FALSE))
y = as.numeric(unlist(read.csv(yfilepath, sep=",", header=FALSE)))
mask =  as.matrix(read.table(maskFile, sep=","))
# mask = mask[,-1] #get rid of the last gene which has all intergenic SNPs
# delete_index = which(apply(mask, 2, sum)==1)
# mask = as.matrix(mask[,-delete_index])
print(dim(X))
print(length(y))
print(dim(mask))

#########################################################################################################
######################################### HELPER FUCTIONS ###############################################
#########################################################################################################

varParamUpdate <- function (X, tau, sigma, logodds, xy, d, alpha0, mu0, Xr0, updates) {
  n <- nrow(X)
  p <- ncol(X)
  
  # Initialize storage:
  alpha <- c(alpha0)
  mu    <- c(mu0)
  Xr    <- c(Xr0)
  
  for (j in updates) {
    
    s <- sigma*tau/(sigma*d[j] + 1)
    r     <- alpha[j] * mu[j]
    mu[j] <- s/tau * (xy[j] + d[j]*r - sum(X[,j]*Xr))
    alpha[j] <- sigmoid(logodds[j] + (log(s/(sigma*tau)) + mu[j]^2/s)/2)
    Xr <- Xr + (alpha[j]*mu[j] - r) * X[,j]
  }
  return(list(alpha = alpha,mu = mu,Xr = Xr))
}

innerLoop <-function (X, y,xy,d, tau, sigma, logodds, alpha, mu, update.order,
                      tol = 1e-4, maxiter = 1e4,
                      outer.iter = NULL){
  n<-nrow(X)
  p<-ncol(X)
  Xr <- c(X %*% (alpha*mu))
  s <- sigma*tau/(sigma*d + 1)
  logw <- rep(0,maxiter)
  err  <- rep(0,maxiter)
  sigma0 = 1
  n0 = 10
  
  for (iter in 1:maxiter) {
    
    # Save the current variational and model parameters.
    alpha0 <- alpha
    mu0    <- mu
    s0     <- s
    tau0 <- tau
    sigma.old <- sigma
  
    logw0 <- varLoss(Xr,d,y,tau,alpha,mu,s,tau*sigma, logodds) #current variational lowerbound
    out   <- varParamUpdate(X,tau,sigma,logodds,xy,d,alpha,mu,Xr,update.order)
    alpha <- out$alpha
    mu    <- out$mu
    Xr    <- out$Xr
    rm(out)
    logw[iter] <- varLoss(Xr,d,y,tau,alpha,mu,s,tau*sigma, logodds) #variational lowerbound after updates
    tau <- (norm2(y - Xr)^2 + dot(d,betavar(alpha,mu,s)) +
                dot(alpha,(s + mu^2)/sigma))/(n + sum(alpha))
    s     <- sigma*tau/(sigma*d + 1)
    sigma <- (sigma0*n0 + dot(alpha,s + mu^2))/(n0 + tau*sum(alpha))
    s  <- sigma*tau/(sigma*d + 1)
    
    # CHECK CONVERGENCE
    err[iter] <- max(abs(alpha - alpha0))
    if (logw[iter] < logw0) {
      logw[iter] <- logw0
      err[iter]  <- 0
      tau      <- tau0
      sigma        <- sigma.old
      alpha      <- alpha0
      mu         <- mu0
      s          <- s0
      break
    } else if (err[iter] < tol)
      break
  }
  
  return(list(logw = logw[1:iter],err = err[1:iter],tau = tau,sigma = sigma,
              alpha = alpha,mu = mu,s = s))
}
  
estimatePVE <- function (fit, X, nr = 1000) {
  
  p  <- ncol(X)
  numModels <- length(fit$logw)
  
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

outerloop <- function(X, I, y,xy,d, SIy, SIX, tau,
                      sigma, logodds, alpha, mu, update.order, tol, maxiter,
                      outer.iter){
  p <- ncol(X)
  if (length(logodds) == 1)
    logodds <- rep(logodds,p)

  out <- innerLoop(X,y,xy,d,tau,sigma,log(10)*logodds,alpha,mu,update.order,
                   tol,maxiter,outer.iter)
  out$logw <- out$logw - determinant(crossprod(I),logarithm = TRUE)$modulus/2
  
  out$b <- c(with(out,SIy - SIX %*% (alpha*mu)))
  numiter  <- length(out$logw)
  out$logw <- out$logw[numiter]
  return(out)
}

BANNvarEM<-function(X, y, centered=FALSE, numModels=20, tol = 1e-4,maxiter = 1e4){
  
  if (centered==FALSE){
    X <- scale(X,center = TRUE,scale = FALSE)
    y <- y - mean(y)
  }
  
  n=nrow(X)
  p=ncol(X)
  xy <- c(y %*%X) 
  d  <- diagsq(X)
  ### Initialize latent variables:
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
    print(i)
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
  
  N=nrow(X)
  p=ncol(X)
  g=ncol(mask)
  SNP_res=BANNvarEM(X, y, centered=centered)
  w<-rep.row(SNP_res$w, p)
  b<-rep.col(rowSums(w*SNP_res$mu*SNP_res$alpha),p)%*%mask
  G<-X%*%b
  G<-leakyrelu(G)
  SNPset_res=BANNvarEM(G, y, centered=TRUE, numModels=30)
  results=list(SNP_level=SNP_res, SNPset_level=SNPset_res)
  return(results)
}

start_time <- Sys.time()
res<-BANN(X, mask, y, centered=FALSE, numModels=20, tol = 1e-10,maxiter = 10000)
#save(res, file = outputfile)
end_time <- Sys.time()


