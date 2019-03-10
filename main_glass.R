#upload Glass data set
library(mlbench)
data("Glass")
N = dim(Glass)[1] #size of sample

#randomly re-order data 
set.seed(1)
order = sample(seq(1, N))
X = Glass[order, !names(Glass) %in% c("Type")]
y = as.numeric(Glass[order, names(Glass) %in% c("Type")])

#scaling to -1 to 1
X.min <- apply(X, 2, min)
X.max <- apply(X, 2, max)
X.min <- matrix(rep(X.min, N), nrow = N, byrow = T)
X.max <- matrix(rep(X.max, N), nrow = N, byrow = T)
a0 = -1
b0 = 1
X = (X - X.min) / (X.max - X.min) * (b0-a0) + a0
X = as.matrix(X)
Xones = cbind(rep(1, dim(X)[1]), X) #add bias

D = dim(X)[2]  #dimensionality
K =  max(y) #number of classes

outcomes = matrix(0, ncol = K, nrow = N)
for (i in 1:N) {
  outcomes[i, y[i]] = 1
}

#train nn on the whole data
nn = neuron(outcomes, X, step_size = 0.5, reg = 0.005, maxiter = 100000)
params.nn = rbind(nn$b, nn$W)
probs.nn = Xones %*% params.nn
probs.nn = exp(probs.nn)
probs.nn = probs.nn / apply(probs.nn, 1, sum)
loss_nn <- -log(apply(probs.nn * outcomes, 1, sum))
Loss_nn <- cumsum(loss_nn)
max(Loss_nn)
sum(params.nn^2)
y.nn = rep(0, N)
for (i in 1:N) {
  y.nn[i] = which.max(probs.nn[i, ])
}
correct.nn = sum(y.nn == y) / N

#params
M <- 3000
M0 <- 1000 #burn-in

#AA on test data set
AA1 = AA_multiclass(outcomes, X, M, M0, a = 0.01, sigma = 0.3)
loss_AA1 <- -log(apply(AA1$gamma * outcomes, 1, sum))
Loss_AA1 <- cumsum(loss_AA1)
max(Loss_AA1)

AA2 = AA_multiclass(outcomes, X, M, M0, a = 1, sigma = 0.3)
loss_AA2 <- -log(apply(AA2$gamma * outcomes, 1, sum))
Loss_AA2 <- cumsum(loss_AA2)
max(Loss_AA2)

time = seq(0, dim(X)[1])
a1 = 0.01
a2 = 1
bound1 = -(a1*sum(params.nn^2) + K*D/2*log(1+1/(8*a1)*time))
bound2 = -(a2*sum(params.nn^2) + K*D/2*log(1+1/(8*a2)*time))

pdf("glass_Loss_diff_a001.pdf", height = 8.5, width = 8.5, paper = "special")
plot(time, c(0, Loss_nn - Loss_AA1), ylim = c(min(bound1), 0), lwd=3,cex=2,cex.lab=2, cex.axis=2, cex.main=2, cex.sub=2, type = "l", xlab = "Time", ylab = "", main= "Loss difference")
lines(time, bound1, lwd=3,cex=2)
dev.off()

pdf("glass_Loss_diff_a1.pdf", height = 8.5, width = 8.5, paper = "special")
plot(time, c(0, Loss_nn - Loss_AA2), ylim = c(min(bound2), 0), lwd=3,cex=2,cex.lab=2, cex.axis=2, cex.main=2, cex.sub=2, type = "l", xlab = "Time", ylab = "", main= "Loss difference")
lines(time, bound2, lwd=3,cex=2)
dev.off()

