library(Matrix)


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

softplus <- function(x){
  return(log(1+ exp(x)))
}

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

int.klbeta <- function (alpha, mu, s, sa)
  (sum(alpha) + dot(alpha,log(s/sa)) - dot(alpha,s + mu^2)/sa)/2 -
  dot(alpha,log(alpha+5e-52)) - dot(1 - alpha,log(1 - alpha+5e-52))

int.gamma <- function (logodds, alpha)
  sum((alpha-1)*logodds + logsigmoid(logodds))

int.linear <- function (Xr, d, y, sigma, alpha, mu, s) {
  n <- length(y)
  return(-n/2*log(2*pi*sigma) - norm2(y - Xr)^2/(2*sigma) 
         - dot(d,betavar(alpha,mu,s))/(2*sigma))
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
int.gamma <- function (logodds, alpha)
  sum((alpha-1)*logodds + logsigmoid(logodds))

int.klbeta <- function (alpha, mu, s, sa)
  (sum(alpha) + dot(alpha,log(s/sa)) - dot(alpha,s + mu^2)/sa)/2 -
  dot(alpha,log(alpha + 5e-22)) - dot(1 - alpha,log(1 - alpha + 5e-22))

int.linear <- function (Xr, d, y, sigma, alpha, mu, s) {
  n <- length(y)
  return(-n/2*log(2*pi*sigma) - norm2(y - Xr)^2/(2*sigma) 
         - dot(d,betavar(alpha,mu,s))/(2*sigma))
}

sigmoid <- function (x)
  1/(1 + exp(-x))

logsigmoid <- function (x)
  -logpexp(-x)

