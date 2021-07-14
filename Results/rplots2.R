library(ggplot2)
library(dplyr)
library(hrbrthemes)
library(grid)
library(gridExtra)
library(extrafont)

loadfonts()
lambda.plot = ggplot(MLEdata2, aes(x = birth)) + 
       geom_density(color = "black") +
       geom_vline(xintercept =0.3, linetype="dashed", color = "#094183", size = 0.8) +
       geom_vline(xintercept =mean(MLEdata2[,1]), color = "red", size = 1.0) +
       xlab(expression(hat(lambda))) +
        theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"),text = element_text(size=15))     

mu.plot = ggplot(MLEdata2, aes(x = death)) + 
  geom_density(color = "black") +
  geom_vline(xintercept =0.05, linetype="dashed", color = "#094183", size = 0.8) +
  geom_vline(xintercept =mean(MLEdata2[,2]), color = "red", size = 1.0) +
  xlab(expression(hat(mu))) + 
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"), text = element_text(size=15))
  
alpha.plot = ggplot(MLEdata2, aes(x = alpha)) + 
  geom_density(color = "black") +
  geom_vline(xintercept =0.5, linetype="dashed", color = "#094183", size = 0.8) +
  geom_vline(xintercept =mean(MLEdata2[,3]), color = "red", size = 1.0) +
  xlab(expression(hat(alpha)))+
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"),text = element_text(size=15))

lst_p = list()
lst_p[[1]] = lambda.plot
lst_p[[2]] = mu.plot
lst_p[[3]] = alpha.plot

lst_p <- lapply(lst_p, ggplotGrob)

gridExtra::grid.arrange(lst_p[[1]], lst_p[[2]], grid::nullGrob(), lst_p[[3]], grid::nullGrob(),
                        layout_matrix = matrix(c(1,1,2,2,3,4,4,5), byrow = TRUE, ncol = 4))