\name{ProgressBar}
\alias{ProgressBar}
\title{
A console progress bar
}
\description{
A progress bar output to the console that will keep track of large time-consuming loops.
}
\usage{
ProgressBar(maxn, ind)
}
\arguments{
  \item{maxn}{
The total number of iterations in the loop.
}
  \item{ind}{
Index of the current loop iteration.
}
}
\details{
This function should be inserted inside of for loops with a known number of iterations. It requires both the current index of iteration as well as the total number of iterations in the loop. Other outputs to the console during the looping will visually disrupt the ProgressBar.
}
\value{
NULL
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
cumsum = 0
for(i in 1:10000){
	cumsum=cumsum+i
	ProgressBar(10000,i)
}
}
\keyword{utilities}
