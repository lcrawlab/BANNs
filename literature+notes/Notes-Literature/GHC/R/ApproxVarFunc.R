ApproxVarFunc <-
function(x,y){
  x2 = x^2;x3 = x^3;x4 = x^4
  params=lm(log(y)~log(x)+x+x2+x3+x4)$coef
  g = function(t_meshh){
    return(as.numeric(exp(params[1])*exp(params[2]*log(t_meshh)+params[3]*t_meshh+params[4]*t_meshh^2+params[5]*t_meshh^3+params[6]*t_meshh^4)))
  }
  if(params[6] < 0){return(g)}
  params=lm(log(y)~log(x)+x+x2)$coef
  g = function(t_meshh){
    return(as.numeric(exp(params[1])*exp(params[2]*log(t_meshh)+params[3]*t_meshh+params[4]*t_meshh^2)))
  }
  return(g)
}
