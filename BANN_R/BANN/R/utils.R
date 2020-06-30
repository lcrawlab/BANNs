#' BANN helper functions

softplus <- function(x){
  return(log(1+ exp(x)))
}

leakyrelu <- function(x)
  ifelse(x >= 0, x, 0.01*x)
relu <- function(x)
  ifelse(x >= 0, x, 0)
logpexp <- function (x) {
  y    <- x
  i    <- which(x < 16)
  y[i] <- log(1 + exp(x[i]))
  return(y)
}
sigmoid <- function (x)
  1/(1 + exp(-x))

logsigmoid <- function (x)
  -logpexp(-x)


var1 <- function (x) {
  n <- length(x)
  return(var(x)*(n-1)/n)
}

var1.cols <- function (X)
  return(apply(X,2,var1))

rep.col <- function (x, n)
  matrix(x,length(x),n,byrow = FALSE)

rep.row <- function (x, n)
  matrix(x,n,length(x),byrow = TRUE)

rand <- function (m, n)
  matrix(runif(m*n),m,n)

randn <- function (m, n)
  matrix(rnorm(m*n),m,n)

dot <- function (x,y)
  sum(x*y)

norm2 <- function (x)
  sqrt(dot(x,x))

betavar <- function (p, mu, s)
  p*(s + (1 - p)*mu^2)


diagsq <- function (X, a = NULL) {
  m<-nrow(X)
  n<-ncol(X)
  if (is.null(a))
    a<-rep(1,m)
  else
    a<-as.double(a)
  y<-rep(0,n)
  for (j in 1:n){
    for(i in 1:m){
      t=X[i,j]
      y[j]=y[j]+(t*t*a[i])
    }
  }
  return(y)
}

varLoss<-function(Xr, d, y, sigma, alpha, mu, s, sa, logodds){
  n <- length(y)
  linearLoss<--n/2*log(2*pi*sigma) - norm2(y - Xr)^2/(2*sigma)
  - dot(d,betavar(alpha,mu,s))/(2*sigma)
  kleffect<-(sum(alpha) + dot(alpha,log(s/sa)) - dot(alpha,s + mu^2)/sa)/2 -
    dot(alpha,log(alpha+5e-52)) - dot(1 - alpha,log(1 - alpha+5e-52))
  return(linearLoss+kleffect+sum((alpha-1)*logodds + logsigmoid(logodds)))
}

normalizelogweights <- function (logw) {

  c <- max(logw)
  w <- exp(logw - c)

  return(w/sum(w))
}

linear.predictors <- function (X, Z, b, alpha, mu) {
  numModels <- ncol(alpha)
  Y  <- Z %*% b + X %*% (alpha*mu)
  return(Y)
}

sigmoid <- function (x)
  1/(1 + exp(-x))

logsigmoid <- function (x)
  -logpexp(-x)
