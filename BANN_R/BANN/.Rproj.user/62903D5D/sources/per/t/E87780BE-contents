#' BANN main function
#' @X is the genotype matrix with n by p.
#' @y is the phenotype vector with length n.
#' @mask is the annotation file with p by g.
#' @centered is used to check whether the X is normalized.
#' @tol is the tolerance for checking convergence.
#' @maxiter is the maximum iteration for updating parameters.
#' @show_progress is the indicator for whether printing out the progress.
#' @examples
#' res = BANN(X, mask, y)
BANN <-function(X, mask, y, centered=FALSE, numModels=20, tol = 1e-4,maxiter = 1e4, show_progress = TRUE){
  ########  Input data quality control ###############

  ### Check input variables:
  if (!(is.matrix(X)) | !(is.numeric(X)) | sum(is.na(X))> 0){
    stop("The genotype data passed in, X, needs to be a numeric matrix")
  }
  if (!(is.numeric(y)) | sum(is.na(y))>0){
    stop("The phenotype data passed in, y, needs to be a numeric vector with the same number of samples (columns) as X, the genotype matrix")
  }

  if (!( (is.matrix(X)) & (is.numeric(X)) & (sum(is.na(X))==0)) ){
    stop("The genotype data passed in, X, needs to be an all numeric matrix")
  }
  if (!(is.finite(tol)& (tol>0))){
    stop("tol parameter needs to be a finite positive number")
  }

  N=nrow(X)
  p=ncol(X)
  g=ncol(mask)
  #first run SNP layer
  SNP_res=BANNvarEM(X, y, centered=centered, numModels=numModels, tol = tol,maxiter = maxiter, show_progress = show_progress)
  #derive posterior mean estimator for weights
  w<-rep.row(SNP_res$w, p)
  b<-rep.col(rowSums(w*SNP_res$mu*SNP_res$alpha),g)*mask
  G<-X%*%b
  G<-leakyrelu(G)
  #run gene layer
  SNPset_res=BANNvarEM(G, y, centered=FALSE, numModels=numModels, tol = tol,maxiter = maxiter, show_progress = show_progress)
  #summarize results
  results=list(SNP_level=SNP_res, SNPset_level=SNPset_res)
  return(results)
}
