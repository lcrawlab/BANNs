ProgressBar <-
function(maxn,ind){
  if(ind==1){
    cat("|1%--------------------50%--------------------100%|\n")
    cat("|")
    return()
  }
  if(maxn==ind){cat("|\n");return()}
  if(floor(50*ind/maxn) != floor(50*(ind-1)/maxn)){cat("|")}
}
