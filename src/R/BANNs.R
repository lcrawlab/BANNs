# @Author: Pinar Demetci, May 2020
# Crawford Lab, Brown University CCMB
# Variational EM optimization for BANN Framework
library(Matrix)
library(foreach)
library(doParallel)

cores = detectCores()
registerDoParallel(cores=cores)

####### Read in data #########

# X=as.matrix(read.csv("/Users/pinardemetci/Desktop/X_TOY.csv", header=FALSE))
# y=as.numeric(unlist(read.csv("/Users/pinardemetci/Desktop/y_TOY.csv", header=FALSE)))
# mask=as.matrix(read.csv("/Users/pinardemetci/Desktop/mask_TOY.csv", header=FALSE))
# path_to_utils="/Users/pinardemetci/Desktop/utils.R"
# outputfile="filepath"
source(path_to_utils) 

varParamUpdate<-function(X,mask, tau.snp, sigma.snp, logodds.snp, xy, d, alpha0.snp, mu0.snp, Xr0, updates.snp,
                         tau.set, sigma.set, logodds.set, alpha0.set, mu0.set, Hr0, updates.set) {
  n <- nrow(X)
  p <- ncol(X)
  g<- ncol(mask)
    
  # Initialize storage:
  alpha.snp <- c(alpha0.snp)
  mu.snp    <- c(mu0.snp)
  Xr    <- c(Xr0)
  alpha.set <- c(alpha0.set)
  mu.set    <- c(mu0.set)
  Hr    <- c(Hr0)
  
  for (j in updates.snp) {
    
    s.snp <- sigma.snp*tau.snp/(sigma.snp*d[j] + 1)
    r.snp     <- alpha.snp[j] * mu.snp[j]
    mu.snp[j] <- s.snp/tau.snp * (xy[j] + d[j]*r.snp - sum(X[,j]*Xr))
    alpha.snp[j] <- sigmoid(logodds.snp[j] + (log(s.snp/(sigma.snp*tau.snp)) + mu.snp[j]^2/s.snp)/2)
    Xr <- Xr + (alpha.snp[j]*mu.snp[j] - r.snp) * X[,j]
  }
  
  for (j in updates.set) {
    #bH<-rep.col(mu.snp*alpha.snp,p)%*%mask
    bH<-rep.col(mu.snp*ifelse(alpha.snp<=0.5,0,alpha.snp),p)%*%mask
    H<-X%*%bH
    Hy <- c(y %*%H) 
    Hd<-diagsq(H)
    s.set <- sigma.set*tau.set/(sigma.set*Hd[j] + 1)
    r.set     <- alpha.set[j] * mu.set[j]
    mu.set[j] <- s.set/tau.set * (Hy[j] + Hd[j]*r.set - sum(H[,j]*Hr))
    alpha.set[j] <- sigmoid(logodds.set[j] + (log(s.set/(sigma.set*tau.set)) + mu.set[j]^2/s.set)/2)
    Hr <- Hr + (alpha.set[j]*mu.set[j] - r.set) * H[,j]
  }
  return(list(alpha.snp = alpha.snp,mu.snp = mu.snp,Xr = Xr,alpha.set = alpha.set,mu.set = mu.set,Hr = Hr))
}

innerLoop <-function(X, y,mask,xy,d, tau.snp, sigma.snp, logodds.snp, alpha.snp, mu.snp, update.order.snp,
                     tau.set, sigma.set, logodds.set, alpha.set, mu.set, update.order.set,
                     tol = 1e-4, maxiter = 1e4, outer.iter = NULL){
  n<-nrow(X)
  p<-ncol(X)
  g<-ncol(mask)
  
  Xr <- c(X %*% (alpha.snp*mu.snp))
  s.snp <- sigma.snp*tau.snp/(sigma.snp*d + 1)
  err.snp  <- rep(0,maxiter)
  sigma0.snp = 1
  n0.snp = 10
  
  #bH<-rep.col(mu.snp*alpha.snp,p)%*%mask
  bH<-rep.col(mu.snp*ifelse(alpha.snp<=0.5,0,alpha.snp),p)%*%mask
  H<-X%*%bH
  Hr <- c(H%*% (alpha.set*mu.set))
  Hd<-diagsq(H)
  s.set <- sigma.set*tau.set/(sigma.set*Hd + 1)
  err.set <- rep(0,maxiter)
  sigma0.set = 1
  n0.set = 10
  logw.snp <- rep(0,maxiter)
  logw.set <- rep(0,maxiter)
  
  
  
  
  for(iter in 1:maxiter){
    # Save the current variational and model parameters.
    alpha0.snp <- alpha.snp
    mu0.snp    <- mu.snp
    s0.snp     <- s.snp
    tau0.snp <- tau.snp
    sigma.old.snp <- sigma.snp
    
    alpha0.set <- alpha.set
    mu0.set    <- mu.set
    s0.set     <- s.set
    tau0.set <- tau.set
    sigma.old.set <- sigma.set
    
    logw0.snp <- varLoss(Xr,d,y,tau.snp,alpha.snp,mu.snp,s.snp,tau.snp*sigma.snp, logodds.snp) 
    logw0.set <- varLoss(Hr,Hd,y,tau.set,alpha.set,mu.set,s.set,tau.set*sigma.set, logodds.set)
    
    out   <- varParamUpdate(X,mask, tau.snp, sigma.snp, logodds.snp, xy, d, alpha.snp, mu.snp, Xr, update.order.snp,
                            tau.set, sigma.set, logodds.set, alpha.set, mu.set, Hr, update.order.set) 
    
    alpha.snp <- out$alpha.snp
    mu.snp    <- out$mu.snp
    Xr    <- out$Xr
    alpha.set <- out$alpha.set
    mu.set    <- out$mu.set
    Hr    <- out$Hr
    rm(out)
    
    logw.snp[iter] <- varLoss(Xr,d,y,tau.snp,alpha.snp,mu.snp,s.snp,tau.snp*sigma.snp, logodds.snp)
    logw.set[iter] <- varLoss(Hr,Hd,y,tau.set,alpha.set,mu.set,s.set,tau.set*sigma.set, logodds.set)
    
    tau.snp <- (norm2(y - Xr)^2 + dot(d,betavar(alpha.snp,mu.snp,s.snp)) +
                  dot(alpha.snp,(s.snp + mu.snp^2)/sigma.snp))/(n + sum(alpha.snp))
    s.snp     <- sigma.snp*tau.snp/(sigma.snp*d + 1)
    sigma.snp <- (sigma0.snp*n0.snp + dot(alpha.snp,s.snp + mu.snp^2))/(n0.snp + tau.snp*sum(alpha.snp))
    s.snp  <- sigma.snp*tau.snp/(sigma.snp*d + 1)
    
    tau.set <- (norm2(y - Hr)^2 + dot(Hd,betavar(alpha.set,mu.set,s.set)) +
                  dot(alpha.set,(s.set + mu.set^2)/sigma.set))/(n + sum(alpha.set))
    s.set     <- sigma.set*tau.set/(sigma.set*Hd + 1)
    sigma.set <- (sigma0.set*n0.set + dot(alpha.set,s.set + mu.set^2))/(n0.set + tau.set*sum(alpha.set))
    s.set  <- sigma.set*tau.set/(sigma.set*Hd + 1)
    
    # CHECK CONVERGENCE
    err.snp[iter] <- max(abs(alpha.snp - alpha0.snp))
    err.set[iter] <- max(abs(alpha.set - alpha0.set))
    if (logw.snp[iter]+logw.set[iter] < logw0.snp+logw0.set) {
      logw.snp[iter] <- logw0.snp
      logw.set[iter] <- logw0.set
      err.snp[iter]  <- 0
      err.set[iter]  <- 0
      
      tau.snp      <- tau0.snp
      sigma.snp       <- sigma.old.snp
      alpha.snp      <- alpha0.snp
      mu.snp         <- mu0.snp
      s.snp          <- s0.snp
      
      tau.set      <- tau0.set
      sigma.set        <- sigma.old.set
      alpha.set      <- alpha0.set
      mu.set         <- mu0.set
      s.set          <- s0.set
      break
    } else if ((err.snp[iter] < tol)&(err.set[iter] < tol))
      break
  }
  return(list(logw.snp = logw.snp[1:iter],logw.set = logw.set[1:iter],err.snp = err.snp[1:iter],err.set = err.set[1:iter],tau.snp = tau.snp,sigma.snp = sigma.snp,
              alpha.snp = alpha.snp,mu.snp = mu.snp,s.snp = s.snp,tau.set = tau.set,sigma.set = sigma.set, alpha.set = alpha.set,mu.set = mu.set,s.set = s.set))
  
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

outerloop <-function(X, I.snp, y,mask, xy,d,SIy.snp,SIX.snp,I.set,SIy.set,SIX.set,tau.snp,sigma.snp,logodds.snp,alpha.snp,mu.snp,update.order.snp,tau.set,sigma.set,logodds.set,alpha.set,mu.set,update.order.set,tol,maxiter,outer.iter){
  p<-ncol(X)
  g<-ncol(SIX.set)
  
  if (length(logodds.snp)==1)
    logodds.snp <- rep(logodds.snp,p)
  if (length(logodds.set)==1)
    logodds.set <- rep(logodds.set,g)
  
  out <- innerLoop(X,y,mask,xy,d,tau.snp,sigma.snp,log(10)*logodds.snp,alpha.snp,mu.snp,update.order.snp,tau.set,sigma.set,log(10)*logodds.set,alpha.set,mu.set,update.order.set,
                   tol,maxiter,outer.iter)
  
  out$logw.snp <- out$logw.snp - determinant(crossprod(I.snp),logarithm = TRUE)$modulus/2
  print(determinant(crossprod(I.snp),logarithm = TRUE)$modulus/2)
  
  out$logw.set <- out$logw.set - determinant(crossprod(I.set),logarithm = TRUE)$modulus/2
  print(determinant(crossprod(I.snp),logarithm = TRUE)$modulus/2)
  
  out$b.snp <- c(with(out,SIy.snp - SIX.snp %*% (alpha.snp*mu.snp)))
  out$b.set <- c(with(out,SIy.set - SIX.set %*% (alpha.set*mu.set)))
  
  numiter.snp  <- length(out$logw.snp)
  out$logw.snp <- out$logw.snp[numiter.snp]
  numiter.set  <- length(out$logw.set)
  out$logw.set <- out$logw.set[numiter.set]
  return(out)
}


BANNvarEM<-function(X,y, mask, centered=FALSE,numModels=30, tol=1e-4, maxiter=1e4){
  
  ### QC for input variables:
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
  
  if (centered==FALSE){
    X<-scale(X,center=TRUE,scale=FALSE)
    y<-y-mean(y)
  }
  
  ### Get dimensions and pre-compute large computes:
  n=nrow(X)
  p=ncol(X)
  g=ncol(mask)
  xy <- c(y %*%X) 
  d  <- diagsq(X)
  
  ### Initialize latent variables:
  tau.snp=rep(var(y),numModels)
  sigma.snp=rep(1,numModels)
  logodds.snp=t(matrix(seq(-log10(p), -1, length.out=numModels)))
  alpha.snp=rand(p,numModels)
  alpha.snp=alpha.snp/rep.row(colSums(alpha.snp),p)
  mu.snp=randn(p, numModels)
  update.order.snp= 1:p
  
  tau.set=rep(var(y),numModels)
  sigma.set=rep(1,numModels)
  logodds.set=t(matrix(seq(-log10(g), -1, length.out=numModels)))
  alpha.set=rand(g,numModels)
  alpha.set=alpha.set/rep.row(colSums(alpha.set),g)
  mu.set=randn(g, numModels)
  update.order.set= 1:g
  
  ### Start storage for the optimization params:
  logw.snp<- rep(0,numModels)
  s.snp <- matrix(0,p,numModels)
  b.snp <-matrix(0,1,numModels)
  
  logw.set<- rep(0,numModels)
  s.set <- matrix(0,g,numModels)
  b.set <-matrix(0,1,numModels)
  
  ### For intercepts:
  I.snp=matrix(1,n,1)
  SIy.snp <- as.vector(solve(n,c(y) %*% I.snp))  
  SIX.snp <- as.matrix(solve(n,t(I.snp) %*% X))
  
  I.set=matrix(1,n,1)
  SIy.set <- as.vector(solve(n,c(y) %*% I.set)) 
  SIX.set <- as.matrix(solve(p,t(I.set) %*% (X%*%mask)))
  
  ### Finding the best initialization of hyperparameters:
  foreach(i=1:numModels)%do%{
    print(i)
    out <- outerloop(X, I.snp, y, mask,xy, d, SIy.snp, SIX.snp, I.set,SIy.set, SIX.set,
                     tau.snp[i],sigma.snp[i],logodds.snp[,i],alpha.snp[,i],mu.snp[,i],update.order.snp,
                     tau.set[i],sigma.set[i],logodds.set[,i],alpha.set[,i],mu.set[,i],update.order.set,
                     tol,maxiter,i )
    logw.snp[i]    <- out$logw.snp
    tau.snp[i]   <- out$tau.snp
    sigma.snp[i]      <- out$sigma.snp
    b.snp[,i] <- out$b.snp
    alpha.snp[,i]  <- out$alpha.snp
    mu.snp[,i]     <- out$mu.snp
    s.snp[,i]      <- out$s.snp
    
    logw.set[i]    <- out$logw.set
    tau.set[i]   <- out$tau.set
    sigma.set[i]      <- out$sigma.set
    b.set[,i] <- out$b.set
    alpha.set[,i]  <- out$alpha.set
    mu.set[,i]     <- out$mu.set
    s.set[,i]      <- out$s.set
  }
  
  i.snp     <- which.max(logw.snp)
  alpha.snp <- rep.col(alpha.snp[,i.snp],numModels)
  mu.snp    <- rep.col(mu.snp[,i.snp],numModels)
  tau.snp <- rep(tau.snp[i.snp],numModels)
  sigma.snp <- rep(sigma.snp[i.snp],numModels)
  
  i.set     <- which.max(logw.set)
  alpha.set <- rep.col(alpha.set[,i.set],numModels)
  mu.set    <- rep.col(mu.set[,i.set],numModels)
  tau.set <- rep(tau.set[i.set],numModels)
  sigma.set <- rep(sigma.set[i.set],numModels)
  
  ### Compute marginal likelihood:
  foreach(i=1:numModels)%do%{
    print(i)
    out <- outerloop(X, I.snp, y,mask, xy, d, SIy.snp, SIX.snp, I.set,SIy.set, SIX.set,
                     tau.snp[i],sigma.snp[i],logodds.snp[,i],alpha.snp[,i],mu.snp[,i],update.order.snp,
                     tau.set[i],sigma.set[i],logodds.set[,i],alpha.set[,i],mu.set[,i],update.order.set,
                     tol,maxiter,i )
    logw.snp[i]    <- out$logw.snp
    tau.snp[i]   <- out$tau.snp
    sigma.snp[i]      <- out$sigma.snp
    b.snp[,i] <- out$b.snp
    alpha.snp[,i]  <- out$alpha.snp
    mu.snp[,i]     <- out$mu.snp
    s.snp[,i]      <- out$s.snp
    
    logw.set[i]    <- out$logw.set
    tau.set[i]   <- out$tau.set
    sigma.set[i]      <- out$sigma.set
    b.set[,i] <- out$b.set
    alpha.set[,i]  <- out$alpha.set
    mu.set[,i]     <- out$mu.set
    s.set[,i]      <- out$s.set
  }
  
  w.snp<- normalizelogweights(logw.snp)
  pip.snp    <- c(alpha.snp %*% w.snp)
  beta.snp     <- c((alpha.snp*mu.snp) %*% w.snp)
  beta.cov.snp <- c(b.snp %*% w.snp)
  
  w.set<- normalizelogweights(logw.set)
  pip.set    <- c(alpha.set %*% w.set)
  beta.set     <- c((alpha.set*mu.set) %*% w.set)
  beta.cov.set <- c(b.set %*% w.set)
  
  SNP_res <- list(b = b.snp,
              logw = logw.snp, w = w.snp, tau = tau.snp, sigma = sigma.snp, logodds = logodds.snp,alpha = alpha.snp,
              mu = mu.snp,s = s.snp,pip = pip.snp,beta = beta.snp, beta.cov = beta.cov.snp,y = y)
  SNPset_res <- list(b = b.set,
                 logw = logw.set, w = w.set, tau = tau.set, sigma = sigma.set, logodds = logodds.set,alpha = alpha.set,
                 mu = mu.set,s = s.set,pip = pip.set,beta = beta.set, beta.cov = beta.cov.set,y = y)
  
  fit <- list(SNP_res = SNP_res, SNPset_res = SNPset_res)
  class(fit) <- c("BANN","list")
  fit$model.pve <- estimatePVE(fit$SNP_res,X)
  fit$pve           <- matrix(0,p,numModels)
  rownames(fit$pve) <- colnames(X)
  sx                <- var1.cols(X)
  for (i in 1:numModels){
    fit$pve[,i] <- sx*(mu.snp[,i]^2 + s.snp[,i])/var1(y)
  }
  X <- X + I.snp %*% SIX.snp
  y <- y + c(I.snp %*% SIy.snp)
  fit$fitted.values <- linear.predictors(X,I.snp,b.snp,alpha.snp,mu.snp)
  fit$residuals <- y - fit$fitted.values
  fit$residuals.response
  
  hyper.labels                = paste("theta",1:numModels,sep = "_")
  rownames(fit$SNP_res$alpha)         = colnames(X)
  rownames(fit$SNP_res$mu)            = colnames(X)
  rownames(fit$SNP_res$s)             = colnames(X)
  names(fit$SNP_res$beta)             = colnames(X)
  names(fit$SNP_res$pip)              = colnames(X)
  rownames(fit$SNP_res$b)        = colnames(I.snp)
  names(fit$SNP_res$beta.cov)         = colnames(I.snp)
  rownames(fit$SNP_res$fitted.values) = rownames(X)
  rownames(fit$SNP_res$residuals) = rownames(X)
  fit$SNP_res$logodds = c(fit$SNP_res$logodds)
  
  ### Note to self: need to integrate SNP names and SNPset names into results, too. -Pinar.
  return(fit)
}

res<-BANNvarEM(X,y,mask,centered=FALSE)
save(res, file = outputfile)







