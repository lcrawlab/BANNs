\name{GenCorBernoulli}
\alias{GenCorBernoulli}

\title{
Create a matrix of correlated bernoulli random variables.
}
\description{
Given a vector of expected values along with a correlation structure (with some restrictions based on the expected values of the Bernoullis), a matrix of correlated Bernoullis following the provided correlation structure is generated.
}
\usage{
GenCorBernoulli(n, mu, rhomat)
}

\arguments{
  \item{n}{
Numeric. Number of instances of the correlated Bernoullis to simulated. This corresponds to the number of rows in the output matrix.
 }
  \item{mu}{
A vector of length p specifying the mean of the each Bernoulli random variable.
}
  \item{rhomat}{
A p by p matrix specifying the desired correlation structure between the Bernoullis.
}
}
\details{
The provided correlation matrix must be positive definite. This is a necessary but not sufficient condition.
}
\value{
An n by p matrix with elements taking values of 0 or 1.
}
\references{
Using the algorithm of Emrich and Piedmonte (1991)
}
\author{
Ian Barnett
}
\note{
For more precise details regarding the restrictions on the correlation structure, so Emrich and Piedmonte (1991).
}


\seealso{
}
\examples{
rho=.4
p=5
n=1000
cormat=autocorr.mat(p,rho)
H1 = GenCorBernoulli(n,mu=rep(.3,p),rhomat=cormat)
H2 = GenCorBernoulli(n,mu=rep(.3,p),rhomat=cormat)
G=H1+H2
cor(G)
}

\keyword{ datagen }
