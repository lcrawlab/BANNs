autocorr.mat <-
function(p = 100, rho = 0.4) {
  mat <- diag(p)
  return(rho^abs(row(mat)-col(mat)))
}
