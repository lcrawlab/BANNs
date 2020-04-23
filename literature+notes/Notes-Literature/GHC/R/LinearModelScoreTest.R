LinearModelScoreTest <-
function(Y,G,X=NULL,type=NULL){
  IDrm = which(is.na(Y))
  if(length(IDrm)>0){
    Y = Y[-IDrm]
    X = X[-IDrm,]
    G = G[-IDrm,]
  }
  if(is.null(type)){
    if(length(intersect(which(Y!=1),which(Y!=0)))==0){
      type="D"
    }else{type="C"}
  }
  N = length(Y)
  P = ncol(G)
  X = cbind(rep(1,N),X)
  if(type=="D"){
    alpha_h = glm(Y~X-1,family="binomial")$coefficients
    Xalpha=X %*% alpha_h
    Mu=as.numeric(exp(Xalpha)/(1+exp(Xalpha)))
    W = Mu*(1-Mu)
  }else{
    lm.out = lm(Y~X-1)
    sigma2 = sum(lm.out$residual^2)/lm.out$df.residual
    Mu = lm.out$fitted.values
    W = rep(sigma2,N)
  }  
  D=t(G) - t(G)%*%X %*%solve(t(X*W) %*% X) %*% t(X*W)
  denoms=sqrt(diag(t(t(D)*W)%*%t(D)))
  D = D/denoms
  tG = t(G)/denoms
  Sigmamat=(t(t(D)*W)%*%t(D))
  S = as.numeric(tG%*%(Y-Mu))
  Sigma=Sigmamat[lower.tri(Sigmamat)]
  return(list(S,Sigma,Sigmamat))
}
