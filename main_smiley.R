#Generate Smiley data set
D = 2  #dimensionality
K = 4  #number of classes
N = 1000  #size of sample

library(mlbench)
smiley = mlbench.smiley(n=N, sd1 = 0.1, sd2 = 0.05)

dt = data.frame(cbind(smiley$x, smiley$classes))
names(dt) = c("x1", "x2", "y")
plot(dt$x1, dt$x2, col = dt$y)

medians = aggregate(dt[, !names(dt) %in% c("y")], list(dt$y), median)
dt = merge(dt, medians, by.x = c("y"), by.y = ("Group.1"))

dt$flag_train = ifelse((dt$y == 1)&(dt$x1.x <= dt$x1.y), 1, 0)
dt$flag_train = ifelse((dt$y == 2)&(dt$x1.x > dt$x1.y), 1, dt$flag_train)
dt$flag_train = ifelse((dt$y == 3)&(dt$x2.x < dt$x2.y), 1, dt$flag_train)
dt$flag_train = ifelse((dt$y == 4)&(dt$x2.x < dt$x2.y), 1, dt$flag_train)

dt.train = dt[dt$flag_train == 1, ]
plot(dt.train$x1.x, dt.train$x2.x, col = dt.train$y)

set.seed(1)
order = sample(seq(1,N))
dt = dt[order, ]

X = as.matrix(dt[, c("x1.x", "x2.x")])
y = dt[, c("y")]

outcomes = matrix(0, ncol = K, nrow = N)
for (i in 1:N) {
  outcomes[i, y[i]] = 1
}

#train and test
X.train = as.matrix(dt[dt$flag_train == 1, c("x1.x", "x2.x")])
y.train = as.matrix(dt[dt$flag_train == 1, c("y")])
N.train = length(y.train)
X.test = as.matrix(dt[dt$flag_train == 0, c("x1.x", "x2.x")])
y.test = as.matrix(dt[dt$flag_train == 0, c("y")])

outcomes.train = matrix(0, ncol = K, nrow = N.train)
for (i in 1:N.train) {
  outcomes.train[i, y.train[i]] = 1
}

outcomes.test = matrix(0, ncol = K, nrow = (N-N.train))
for (i in 1:(N-N.train)) {
  outcomes.test[i, y.test[i]] = 1
}

Xones.test = cbind(rep(1, dim(X.test)[1]), X.test)

#params
M <- 3000
M0 <- 1000 #burn-in

#AA on test data set
AA = AA_multiclass(outcomes.test, X.test, M, M0, a = 0.01, sigma = 0.9)
loss_AA <- -log(apply(AA$gamma * outcomes.test, 1, sum))
Loss_AA <- cumsum(loss_AA)
max(Loss_AA)

#nn on training data
nn = neuron(outcomes.train, X.train, step_size = 2, reg = 0, maxiter = 10000)
params.nn = rbind(nn$b, nn$W)
probs.nn = Xones.test %*% params.nn
probs.nn = exp(probs.nn)
probs.nn = probs.nn / apply(probs.nn, 1, sum)
loss_nn <- -log(apply(probs.nn * outcomes.test, 1, sum))
Loss_nn <- cumsum(loss_nn)
max(Loss_nn)
y.nn = rep(0, N - N.train)
for (i in 1:(N-N.train)) {
  y.nn[i] = which.max(probs.nn[i, ])
}
correct.nn = sum(y.nn == y.test) / (N - N.train)

#nn on test data
nn2 = neuron(outcomes.test, X.test, step_size = 2, reg = 0, maxiter = 100000)
params.nn2 = rbind(nn2$b, nn2$W)
probs.nn2 = Xones.test %*% params.nn2
probs.nn2 = exp(probs.nn2)
probs.nn2 = probs.nn2 / apply(probs.nn2, 1, sum)
loss_nn2 <- -log(apply(probs.nn2 * outcomes.test, 1, sum))
Loss_nn2 <- cumsum(loss_nn2)
max(Loss_nn2)
y.nn2 = rep(0, N - N.train)
for (i in 1:(N-N.train)) {
  y.nn2[i] = which.max(probs.nn2[i, ])
}
correct.nn2 = sum(y.nn2 == y.test) / (N - N.train)

probs.nn3 = matrix(0, ncol = K, nrow = dim(X.test)[1])
probs.nn3[1, ] = Xones.test[1, ] %*% params.nn
#nn online on test data 
for (i in 1:(dim(X.test)[1]-1)) {
  outcomes.temp = rbind(outcomes.train, outcomes.test[1:i, ])
  X.temp = rbind(X.train, X.test[1:i, ])
  nn3 = neuron(outcomes.temp, X.temp, step_size = 2, reg = 0, maxiter = 100000)
  params.nn3 = rbind(nn3$b, nn3$W)
  probs.nn3[i+1, ] = Xones.test[i+1, ] %*% params.nn3
}

probs.nn3 = exp(probs.nn3)
probs.nn3 = probs.nn3 / apply(probs.nn3, 1, sum)
loss_nn3 <- -log(apply(probs.nn3 * outcomes.test, 1, sum))
Loss_nn3 <- cumsum(loss_nn3)

time = seq(0, dim(X.test)[1])
a = 0.01
bound = -(a*sum(params.nn2^2) + K*D/2*log(1+D/(8*a)*time))

pdf("smiley_Loss_diff_test.pdf", height = 8.5, width = 8.5, paper = "special")
plot(time, c(0, Loss_nn2 - Loss_AA), ylim = c(min(bound), 0), lwd=3,cex=2,cex.lab=2, cex.axis=2, cex.main=2, cex.sub=2, type = "l", xlab = "Time", ylab = "", main= "Loss difference")
lines(time, bound, lwd=3,cex=2)
dev.off()

pdf("smiley_Loss_diff_train.pdf", height = 8.5, width = 8.5, paper = "special")
plot(Loss_nn - Loss_AA, lwd=3,cex=2,cex.lab=2, cex.axis=2, cex.main=2, cex.sub=2, type = "l", xlab = "Time", ylab = "", main= "Loss difference")
dev.off()

pdf("smiley_Loss_diff_online.pdf", height = 8.5, width = 8.5, paper = "special")
plot(Loss_nn3 - Loss_AA, lwd=3,cex=2,cex.lab=2, cex.axis=2, cex.main=2, cex.sub=2, type = "l", xlab = "Time", ylab = "", main= "Loss difference")
lines(rep(0, dim(X.test)[1]), lwd=3,cex=2)
dev.off()