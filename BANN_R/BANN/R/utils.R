#' This program is free software: you can redistribute it under the
#' terms of the GNU General Public License; either version 3 of the
#' License, or (at your option) any later version.
#'
#' This program is distributed in the hope that it will be useful, but
#' WITHOUT ANY WARRANY; without even the implied warranty of
#' MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE. See the GNU
#' General Public License for more details.
#'
#' BANN helper functions

# Softplus activation function
softplus <- function(x){
  return(log(1+ exp(x)))
}

# LeakyReLu activation function
leakyrelu <- function(x)
  ifelse(x >= 0, x, 0.01*x)

# ReLu activation function
relu <- function(x)
  ifelse(x >= 0, x, 0)

#sigmoid activation function
sigmoid <- function (x)
  1/(1 + exp(-x))

# function for log(sigmoid(x))
logsigmoid <- function (x)
  -logpexp(-x)

# returns log(1 + exp(x)). For large entries of x, log(1 + exp(x)) is effectively the same as x.
logpexp <- function (x) {
  y    <- x
  i    <- which(x < 16)
  y[i] <- log(1 + exp(x[i]))
  return(y)
}

# return the second moment of x about its mean.
var1 <- function (x) {
  n <- length(x)
  return(var(x)*(n-1)/n)
}

# return the second moment of each column of x about its mean.
var1.cols <- function (X)
  return(apply(X,2,var1))

# replicate vector x to create an m x n matrix, where m = length(x).
rep.col <- function (x, n)
  matrix(x,length(x),n,byrow = FALSE)

# replicate vector x to create an n x m matrix, where m = length(x).
rep.row <- function (x, n)
  matrix(x,n,length(x),byrow = TRUE)

# return matrix containing values randomly drawn from the standard uniform distribution.
rand <- function (m, n)
  matrix(runif(m*n),m,n)

# return matrix containing values randomly drawn from the standard normal distribution.
randn <- function (m, n)
  matrix(rnorm(m*n),m,n)

# return the dot product of vectors x and y.
dot <- function (x,y)
  sum(x*y)

# return the quadratic norm (2-norm) of vector x.
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

# Generates a vector of n points that are equally spaced on the
# logarithmic scale. Note that x and y should be positive numbers.
logspace <- function (x, y, n)
  2^seq(log2(x),log2(y),length = n)

selectmixsd <- function (x, k) {
  smin <- 1/10
  if (all(x^2 < 1))
    smax <- 1
  else
    smax <- 2*sqrt(max(x^2 - 1))
  return(c(0,logspace(smin,smax,k - 1)))
}

# ----------------------------------------------------------------------
# betavarmix(p,mu,s) returns variances of variables drawn from mixtures of
# normals. Each of the inputs is a n x k matrix, where n is the number of
# variables and k is the number of mixture components. Specifically,
# variable i is drawn from a mixture in which the jth mixture component is
# the univariate normal with mean mu[i,j] and variance s[i,j].
#
# Note that the following two lines should return the same result when k=2
# and the first component is the "spike" density with zero mean and
# variance.
#
#   y1 <- betavar(p,mu,s)
#   y2 <- betavarmix(c(1-p,p),cbind(0,mu),cbind(0,s))
#
betavarmix <- function (p, mu, s)
  rowSums(p*(s + mu^2)) - rowSums(p*mu)^2

# ----------------------------------------------------------------------
# Compute the lower bound to the marginal log-likelihood.
computevarlbmix <- function (Z, Xr, d, y, sigma, sa, w, alpha, mu, s) {

  # Get the number of samples (n), variables (p) and mixture
  # components (K).
  n <- length(y)
  p <- length(d)
  K <- length(w)

  # Compute the variational lower bound.
  out <- (-n/2*log(2*pi*sigma)
          - determinant(crossprod(Z),logarithm = TRUE)$modulus/2
          - (norm2(y - Xr)^2 + dot(d,betavarmix(alpha,mu,s)))/(2*sigma))
  for (i in 1:K)
    out <- (out + sum(alpha[,i]*log(w[i] + 5e-52))
            - sum(alpha[,i]*log(alpha[,i] + 5e-52)))
  for (i in 2:K)
    out <- (out + (sum(alpha[,i]) + sum(alpha[,i]*log(s[,i]/(sigma*sa[i]))))/2
            - sum(alpha[,i]*(s[,i] + mu[,i]^2))/(sigma*sa[i])/2)
  return(out)
}

# ----------------------------------------------------------------------
# Compute the local false sign rate (LFSR) for each variable. This
# assumes that the first mixture component is a "spike" (that is, a
# normal density with a variance approaching zero).
computelfsrmix <- function (alpha, mu, s) {

  # Get the number of variables (p) and the number of mixture
  # components (k).
  p <- nrow(alpha)
  k <- ncol(alpha)

  # For each variable, get the posterior probability that the
  # regression coefficient is exactly zero.
  p0 <- alpha[,1]

  # For each variable, get the posterior probability that the
  # regression coefficient is negative.
  if (k == 2)
    pn <- alpha[,2] * pnorm(0,mu[,2],sqrt(s[,2]))
  else
    pn <- rowSums(alpha[,-1] * pnorm(0,mu[,-1],sqrt(s[,-1])))

  # Compute the local false sign rate (LFSR) following the formula
  # given in the Biostatistics paper, "False discovery rates: a new
  # deal".
  lfsr     <- rep(0,p)
  b        <- pn > 0.5*(1 - p0)
  lfsr[b]  <- 1 - pn[b]
  lfsr[!b] <- p0[!b] + pn[!b]

  return(lfsr)
}

getColumn<-function(X, j, n) {
  return(X + n*j);
}

