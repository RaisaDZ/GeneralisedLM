#function for backpropagation for single neuron with softmax transfer function

#Inputs: y - binary represenation of target vector. For example if there are three classes,
#        y should be decoded as (0, 0, 1), (0, 1, 0) or (1, 0, 0)
#        X - matrix of features where rows are observations and columns are features,
#        X should be in matrix format. If data.frame then put X = as.matrix(X)
#        step_size - step size for backpropagation
#        reg - parameter of regularization
#        maxiter - maximum number of iteration

#Outputs: W, b - vector of parameters,
#         loss - loss on each iteration

neuron <- function(y, X, step_size, reg, maxiter){
  N = dim(X)[1]
  D = dim(X)[2]  #dimentionality
  K = dim(y)[2]   #number of classes
  
  # initialize parameters randomly
  W = matrix(runif(D*K, 0, 0.1), ncol = K, nrow = D) 
  b = matrix(0, ncol = K, nrow = 1)
  
  threshold = 10^(-6)
  loss = rep(0, maxiter)
  for (loop in 1:maxiter) {
    # evaluate class scores, [N x K]
    scores = X %*% W + matrix(rep(b, N), ncol = K, byrow = T)
    
    # compute the class probabilities
    exp_scores = exp(scores)
    probs = exp_scores / apply(exp_scores, 1, sum) # [N x K]
    
    # compute the loss: average cross-entropy loss and regularization
    correct_logprobs = -log(apply(probs * y, 1, sum))
    data_loss = sum(correct_logprobs)/N
    reg_loss = 0.5*reg*sum(W^2)
    loss[loop] = data_loss + reg_loss
    if (loop > 1) {
      if (abs(loss[loop] - loss[loop-1]) < threshold) {
        print(paste("Converged after", loop, "iterations", sep = " "))
        break
      }
    }
    # compute the gradient on scores
    dscores = probs
    dscores = (dscores - y) / N
    
    # backpropate the gradient to the parameters (W,b)
    dW = t(X) %*% dscores
    db = apply(dscores, 2, sum)
    
    dW = dW + reg*W # regularization gradient
    
    # perform a parameter update
    W = W - step_size * dW
    b = b - step_size * db
  }
  
  return(list(W = W, b = b, loss = loss[1:loop]))
}
