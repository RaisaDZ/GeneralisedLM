#download data sets
#data is available at http://www.football-data.co.uk/englandm.php

library(openxlsx)
dt1 <- read.xlsx("E0_201415.xlsx")
dt2 <- read.xlsx("E0_201516.xlsx")
dt3 <- read.xlsx("E0_201617.xlsx")

dt1 <- dt1[, names(dt1) %in% names(dt3)]

dt <- rbind(dt1, dt2, dt3)
dt$order <- seq(1, dim(dt)[1])

#convert odds to probabilities
odds <- dt[, 24:65]

odds_home <-odds[, substr(names(odds), nchar(names(odds)), nchar(names(odds))) == "H"]
odds_away <-odds[, substr(names(odds), nchar(names(odds)), nchar(names(odds))) == "A"]
odds_draw <-odds[, substr(names(odds), nchar(names(odds)), nchar(names(odds))) == "D"]

odds_home <- odds_home[, !names(odds_home) %in% c("BbAH", "BbMxAHH", "BbAvAHH")]
odds_away <- odds_away[, !names(odds_away) %in% c("BbMxAHA", "BbAvAHA")]

expert_home <- 1 / odds_home 
expert_away <- 1 / odds_away 
expert_draw <- 1 / odds_draw 

expert_total = expert_home + expert_away + expert_draw
expert_home = expert_home / expert_total
expert_away = expert_away / expert_total
expert_draw = expert_draw / expert_total

dt.main = cbind(dt[, c("order", "FTR")], expert_home, expert_draw)

#unique teams
teams <- unique(dt$HomeTeam)
teams <- data.frame(teams)
teams$team_num <- seq(1, dim(teams)[1])

dt <- merge(dt, teams, by.x = c("HomeTeam"), by.y = c("teams"))
colnames(dt)[dim(dt)[2]] <- "HomeTeam_num"
dt <- merge(dt, teams, by.x = c("AwayTeam"), by.y = c("teams"))
colnames(dt)[dim(dt)[2]] <- "AwayTeam_num"
dt <- dt[order(dt$order), ]

dt_new <- cbind(dt[, -c(24:65)])

#calculate average historical characteristics for home and away teams
dt.home <- merge(dt_new[, c("order", "HomeTeam_num")], dt_new, by = c("HomeTeam_num"))
dt.home <- dt.home[dt.home$order.x > dt.home$order.y, ]  #take only previous matches
dt.home <- dt.home[order(dt.home$HomeTeam_num, dt.home$order.x, dt.home$order.y), ]

dt.home$home_bin <- ifelse(dt.home$FTR == "H", 1, 0)
dt.home$away_bin <- ifelse(dt.home$FTR == "A", 1, 0)
dt.home$draw_bin <- ifelse(dt.home$FTR == "D", 1, 0)

home_list <- c("order.x", "FTHG", "FTAG", "HTHG", "HTAG",
               "HS", "HST", "HF", "HC", "HY", "HR",
               "home_bin", "away_bin", "draw_bin")

dt.home1 <- data.frame(dt.home)[, home_list] 
dt.home_aggr <- aggregate(dt.home1, by=list(dt.home1$order.x), FUN=mean)
names(dt.home_aggr) <- paste(names(dt.home_aggr), "_home", sep = "")

dt.away <- merge(dt_new[, c("order", "AwayTeam_num")], dt_new, by = c("AwayTeam_num"))
dt.away <- dt.away[dt.away$order.x > dt.away$order.y, ]  #take only previous matches
dt.away <- dt.away[order(dt.away$AwayTeam_num, dt.away$order.x, dt.away$order.y), ]

dt.away$home_bin <- ifelse(dt.away$FTR == "H", 1, 0)
dt.away$away_bin <- ifelse(dt.away$FTR == "A", 1, 0)
dt.away$draw_bin <- ifelse(dt.away$FTR == "D", 1, 0)

away_list <- c("order.x", "FTHG", "FTAG", "HTHG", "HTAG",
               "AS", "AST", "AF", "AC", "AY", "AR",
               "home_bin", "away_bin", "draw_bin")

dt.away1 <- data.frame(dt.away)[, away_list] 
dt.away_aggr <- aggregate(dt.away1, by=list(dt.away1$order.x), FUN=mean)
names(dt.away_aggr) <- paste(names(dt.away_aggr), "_away", sep = "")

dt.features <- merge(dt.home_aggr, dt.away_aggr, by.x = c("order.x_home"), by.y = c("order.x_away"))
dt.features$order <- dt.features$order.x_home
dt.features <- dt.features[, !names(dt.features) %in% c("order.x_home", "order.x_away", "Group.1_home", "Group.1_away")]

list_scaled <- c("FTHG_home", "FTAG_home", "HTHG_home",
                 "HTAG_home", "HS_home", "HST_home",     
                 "HF_home", "HC_home", "HY_home", "HR_home",
                 "FTHG_away", "FTAG_away", "HTHG_away",
                 "HTAG_away", "AS_away", "AST_away", "AF_away",      
                 "AC_away", "AY_away", "AR_away")

#min and max on training only
id.train <- 760 #take two seasons for training
dt.scaled <- dt.features[, list_scaled]
dt.min <- apply(subset(dt.features, dt.features$order <= id.train)[, list_scaled], 2, min)
dt.max <- apply(subset(dt.features, dt.features$order <= id.train)[, list_scaled], 2, max)
dt.min <- matrix(rep(dt.min, nrow(dt.scaled)), nrow = nrow(dt.scaled), byrow = T)
dt.max <- matrix(rep(dt.max, nrow(dt.scaled)), nrow = nrow(dt.scaled), byrow = T)
a = -1
b = 1
dt.scaled <- (dt.scaled - dt.min) / (dt.max - dt.min) * (b-a) + a
dt.scaled[dt.scaled < a] <- a
dt.scaled[dt.scaled > b] <- b

dt.features_new <- cbind(dt.features[, !names(dt.features) %in% list_scaled], dt.scaled)

dt.main <- merge(dt.main, dt.features_new, by = c("order"))
dt.main = dt.main[complete.cases(dt.main), ] #remove NA
N = dim(dt.main)[1]  # number of examples
N.train = sum(dt.main$order < id.train)

X <- as.matrix(dt.main[!names(dt.main) %in% c("order", "FTR")])
D = dim(X)[2] #dimentionality
K = 3 # number of classes
y = rep(0, N)
for (i in 1:N) {
  y[i] = ifelse(dt.main$FTR[i] == "H", 1, y[i])
  y[i] = ifelse(dt.main$FTR[i] == "A", 2, y[i])
  y[i] = ifelse(dt.main$FTR[i] == "D", 3, y[i])
}

outcomes = matrix(0, ncol = K, nrow = N)
for (i in 1:N) {
  outcomes[i, y[i]] = 1
}

#train and test
X.train = X[1:N.train, ]
y.train = y[1:N.train]
outcomes.train = outcomes[1:N.train, ]
X.test = X[(N.train+1):N, ]
y.test = y[(N.train+1):N]
outcomes.test = outcomes[(N.train+1):N, ]
Xones.test = cbind(rep(1, dim(X.test)[1]), X.test)

nn = neuron(outcomes.train, X.train, step_size = 0.9, reg = 0.005, maxiter = 100000)
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

#params
M <- 2000
M0 <- 500 #burn-in

#AA on test data set
AA = AA_multiclass(outcomes.test, X.test, M, M0, a = 0.05, sigma = 0.2)
loss_AA <- -log(apply(AA$gamma * outcomes.test, 1, sum))
Loss_AA <- cumsum(loss_AA)
max(Loss_AA)

pdf("football_Loss_diff_batch.pdf", height = 8.5, width = 8.5, paper = "special")
plot(Loss_nn - Loss_AA, lwd=3,cex=2,cex.lab=2, cex.axis=2, cex.main=2, cex.sub=2, type = "l", xlab = "Time", ylab = "", main= "Loss difference")
dev.off()

probs.nn3 = matrix(0, ncol = K, nrow = dim(X.test)[1])
probs.nn3[1, ] = Xones.test[1, ] %*% params.nn
#nn online on test data 
for (i in 1:(dim(X.test)[1]-1)) {
  outcomes.temp = rbind(outcomes.train, outcomes.test[1:i, ])
  X.temp = rbind(X.train, X.test[1:i, ])
  nn3 = neuron(outcomes.temp, X.temp, step_size = 0.6, reg = 0.1, maxiter = 100000)
  params.nn3 = rbind(nn3$b, nn3$W)
  probs.nn3[i+1, ] = Xones.test[i+1, ] %*% params.nn3
}

probs.nn3 = exp(probs.nn3)
probs.nn3 = probs.nn3 / apply(probs.nn3, 1, sum)
loss_nn3 <- -log(apply(probs.nn3 * outcomes.test, 1, sum))
Loss_nn3 <- cumsum(loss_nn3)

pdf("football_Loss_diff_online.pdf", height = 8.5, width = 8.5, paper = "special")
plot(Loss_nn3 - Loss_AA, lwd=3,cex=2,cex.lab=2, cex.axis=2, cex.main=2, cex.sub=2, type = "l", xlab = "Time", ylab = "", main= "Loss difference")
dev.off()
