\name{GHC-package}
\alias{GHC-package}
\docType{package}
\title{
P-values for the generalized higher criticism
}
\description{
Given a set of correlated marginal test statistics, approximate analytic p-values are computed. For large dimension, simulation of the null distribution is used instead for obtaining p-values. 
}
\details{
\tabular{ll}{
Package: \tab GHC\cr
Type: \tab Package\cr
Version: \tab 1.0\cr
Date: \tab 2015-10-28\cr
License: \tab GPL-2\cr
}
Given a vector marginal test statistics Z which follow are each each standard normals marginally along with their correlation matrix Sigma, the GHC p-value is obtained using the function call: GHC(Z,Sigma).
}
\author{
Ian Barnett

Maintainer: Ian Barnett <ianjamesbarnett@gmail.com>
}
\references{
Barnett, I., and Lin, X. "The Generalized Higher Criticism for testing SNP-set effects in genetic association studies". (under review)

Barnett, I. and Lin, X. "Analytic p-values for the higher criticism test in finite d problems," Biometrika. (2013)
}
\keyword{ package }
\seealso{
\code{\link{GHC}}
}
\examples{
}
