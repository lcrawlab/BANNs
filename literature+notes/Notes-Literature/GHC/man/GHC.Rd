\name{GHC}
\alias{GHC}

\title{
P-values for the higher criticism allowing for arbitrary correlation structure.
}
\description{
Given a set of correlated marginal test statistics, approximate analytic p-values are computed. For large dimension, simulation of the null distribution is used instead for obtaining p-values. 
}
\usage{
GHC(S, Sigmamat, just_stat = F, maxcorr = 0.95, t_0 = 0, p_thresh = 100, restricted = FALSE)
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
  \item{t_0}{
Minimum value when searching for a supremum. Default to 0 for full range search.
}
  \item{p_thresh}{
If the dimension p is greater than p_thresh, p-values are calculated through simulation as opposed to analytically. The default is p_thresh=100. The reason for this is because the computational complexity of the analytic p-value calculation is O(p^3).
}
  \item{restricted}{
Defaults to FALSE. If TRUE, then the supremum is restricted to a very small, high, range.
}
}
\details{
When starting with genotype data, use the LinearModelScoreTest function to create marginal test statistics and the corresponding correlation matrices. The t_0 and restricted parameters matter more in asymptotic analysis of the higher criticism. For smaller, finite p, it is recommended to search the full range and to leave these parameters as their default values.
}
\value{
Numeric. A p-value.
}
\references{
Barnett, I., and Lin, X. "The Generalized Higher Criticism for testing SNP-set effects in genetic association studies". (under review)
}
\author{
Ian Barnett
}
\note{
While analytic p-values can reach values smaller than 10^-6, simulated p-values cannot be any smaller than that.
}


\seealso{
\code{\link{GHC_GWAS_SSD}}, \code{\link{LinearModelScoreTest}}
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
GHC(Z,Sigma) # analytic p-value
GHC(Z,Sigma,p_thresh=1) # simulated p-value because p > p_thresh
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ htest}
