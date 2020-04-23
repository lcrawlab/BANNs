\name{HCindep}
\alias{HCindep}
\title{
P-values for the higher criticism under independence
}
\description{
Given a set of independent test statistics, the higher criticism p-value is computed where the supremum is not restricted. If the provided correlation structure is not independent (diag(p)), then the test statistics are first tranformed to become indepedent through the cholesky square root transformation.
}
\usage{
HCindep(S, Sigmamat, just_stat = F, maxcorr = 0.95, p_thresh = 100)
}
\arguments{
  \item{S}{
Marginal test statistics (length p)
}
  \item{Sigmamat}{
Correlation matrix of S (dimension p by p)
  }
  \item{just_stat}{
Boolean valued. If TRUE, then the test statistic is returned, not the p-value. Default to FALSE where the p-value is returned.
}
  \item{maxcorr}{
If any pairwise correlation between test statistics is greater than maxcorr, one of the test statistics will be treated as redundant and removed.
}
  \item{p_thresh}{
If the dimension p is greater than p_thresh, p-values are calculated through simulation as opposed to analytically. The default is p_thresh=100. The reason for this is because the computational complexity of the analytic p-value calculation is O(p^3).
}
}
\details{
When starting with genotype data, use the LinearModelScoreTest function to create marginal test statistics and the corresponding correlation matrices. These can be input into HCindep.
}
\value{
Numeric. A p-value.
}
\references{
Barnett, I. and Lin, X. "Analytic p-values for the higher criticism test in finite d problems," Biometrika. (2013)
}
\author{
Ian Barnett
}
\note{
While analytic p-values can reach values smaller than 10^-6, simulated p-values cannot be any smaller than that.
}


\seealso{
\code{\link{LinearModelScoreTest}}, \code{\link{GHC}}
}
\examples{
rho=.4
p=10
n=1000
cormat=autocorr.mat(p,rho)
H1 = GenCorBernoulli(n,mu=rep(.3,p),rhomat=cormat)
H2 = GenCorBernoulli(n,mu=rep(.3,p),rhomat=cormat)
G=H1+H2
Y=rnorm(n)
out=LinearModelScoreTest(Y,G,X=NULL)
Z = out[[1]]
Sigma= out[[3]]
HCindep(Z,Sigma) # analytic p-value
HCindep(Z,Sigma,p_thresh=1) # simulated p-value because p > p_thresh
}
\keyword{ htest }
