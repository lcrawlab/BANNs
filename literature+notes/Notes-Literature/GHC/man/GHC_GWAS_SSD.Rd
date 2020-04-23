\name{GHC_GWAS_SSD}
\alias{GHC_GWAS_SSD}

\title{
Find GHC p-values for all SNP-sets in a GWAS
}
\description{
Given an SSD file, iterate over all SNP-sets and output the list of p-values in increasing over.
}
\usage{
GHC_GWAS_SSD(SSD.Info, Y, X, list_restrict = NULL)
}

\arguments{
  \item{SSD.Info}{
The SSD file. This can be created from a binary PLINK file along with a setID file using the Generate_SSD_SetID function in the SKAT package.
}
  \item{Y}{
Outcome vector of length n.
}
  \item{X}{
Covariate matrix of dimension n by q. These will be controlled for as confounders for each SNP-set.
}
  \item{list_restrict}{
A list of SNP-set IDs to restrict the analysis to. By default each SNP-set will be tested.
}
}
\details{
This function borrows heavily from the SKAT package. In order to perform a GWAS, a bed, bim, and fam file in plink format along with a setID file. The setID file has two columns: column 1 contains SNP-set ID while column 2 contains the SNP ID that belongs to that SNP-set. With these four files, a call to the Generate_SSD_SetID function will create the necessary SSD files. See ?Generate_SSD_SetID for more information.
}
\value{
A matrix of SNP-set ID's, their dimension, and the corresponding p-values, sorted from smallest to largest.
}
\references{
}
\author{
Ian Barnett
}
\note{
}


\seealso{
\code{\link{Generate_SSD_SetID}}
}
\examples{
}
