\name{autocorr.mat}
\alias{autocorr.mat}
\title{
Creates an autocorrelated correlation matrix
}
\description{
Given a correlation parameter, rho, and a dimension, p, this function creates and outputs a correlation matrix with autocorrelation structure.
}
\usage{
autocorr.mat(p = 100, rho = 0.4)
}
\arguments{
  \item{p}{
Dimension of matrix.
}
  \item{rho}{
Autocorrelation parameter.
}
}
\details{
Creates a matrix with ijth entry equal to rho^|j-i|.
}
\value{
Correlation matrix of dimension p by p.
}
\references{
}
\author{
Ian Barnett
}
\note{
}


\seealso{
}
\examples{
cormat=autocorr.mat(p=5,rho=.5)
cormat
}

\keyword{datagen}
