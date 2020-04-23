\name{LinearModelScoreTest}
\alias{LinearModelScoreTest}
\title{
Marginal score test statistics from the GLM framework 
}
\description{
Given a genotype matrix, outcome vector, and a matrix of covariates to control for, a vector of marginal test statistics, one for each SNP, will be generated. The joint model with the full genotype and covariate matrix is fit as the systematic component. A linear link is used for continuous outcomes, with a logit link used for dichotomous outcomes.
}
\usage{
LinearModelScoreTest(Y, G, X = NULL, type = NULL)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{Y}{
Outcome vector (length n)
}
  \item{G}{
Genotype matrix (dimension n by p)
}
  \item{X}{
Covariate matrix (dimension n by q)
}
  \item{type}{
"C" for continous Y and "D" for dichotomous Y. By default, if Y is dichotomous then type="D", but type="C" otherwise.
}
}
\details{
If covariate matrix X is not specified, only an intercept term is assumed. Do not include the intercept term in the covariate matrix X!
}
\value{

\item{1 }{Vector of marginal test statistics}
\item{2 }{Correlation matrix of marginal test statistics vector, in vector form.}
\item{3 }{Correlation matrix of marginal test statistics vector, in matrix form. This form is likely more useful.}
}
\references{
}
\author{
Ian Barnett
}
\note{
Only logit and linear links are considered, not any GLM.
}


\seealso{
}
\examples{
n=1000
p=6
G=matrix(rbinom(n*p,1,.3),nrow=n,ncol=p)
X=matrix(rnorm(n*2),nrow=n,ncol=2)
Y=rnorm(n)
objout=LinearModelScoreTest(Y,G,X)
objout[[1]]
objout[[3]]
}
\keyword{ models }
