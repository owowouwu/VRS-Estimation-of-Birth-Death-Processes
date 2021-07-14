birth.estim = MLEdata[,1]
death.estim = MLEdata[,2]
alpha.estim = MLEdata[,3]


hist(birth.estim, freq = FALSE, xlab =  "Estimates of Birth Rate")
abline(v = 0.3, col = "red", lty = 2, lwd = 3)

hist(death.estim, freq = FALSE, xlab = "Estimates of Death Rate")
abline(v = 0.05, col = "red", lty = 2, lwd = 3)

hist(alpha.estim, freq = FALSE, xlab = "Estimates of Alpha")
abline(v = 0.5, col = "red", lty = 2, lwd = 3)
