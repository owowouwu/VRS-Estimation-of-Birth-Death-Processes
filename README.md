# Background
   A birth death process (BDP) is a continuous time Markov chain which models the number of particles in a system. This number can either transition up or down one step at random according to a sequence of birth-rates and death-rates, depending on the population.
   The model of interest is a type of logistic growth model with birth-rates and death-rates given respectively by <br/><br/>
   <img src="https://latex.codecogs.com/gif.latex?\lambda_k&space;=k^2e^{-k\alpha}" title="\lambda_k =k^2e^{-k\alpha}" /><br/>
   <img src="https://latex.codecogs.com/gif.latex?\mu_k&space;=&space;k\mu" title="\mu_k = k\mu" /><br/><br/>
   The aim of this project was to implement a novel way of finding transition probabilities in the process as described in [Crawford et al.](https://www.researchgate.net/publication/51957281_Estimation_for_General_Birth-Death_Processes), then apply an EM (expectation maximisation) algorithm to estimate the parameters in the model above.
   We aimed to compare this method to directly optimising the likelihood as well as checking compute times of transition probabilities against a matrix exponential method. <br/>

## Acknowledgements
I wish to thank my supervisors Dr Sophie Hautphenne and Dr Brendan Patch for their time and work in supervising me, as well as the School of Mathematics and Statistics for this wonderful opportunity. This experience has been quite invaluable to me, as it has allowed me to both develop my ability and grow my appreciation for scientific research.
    
## References
1. A. Kraus A.C. Davidson S. Hautphenne.“Parameter estimation for discretely observed linear birth-and-death processes”. In:The Interna-tional Biometric Society(2020).doi:10.1111/biom.13282. \\
1.  Marc A. Suchard Forrest W. Crawford.“Transition probabilities for general birth–death processes with applications in ecology, genetics, andevolution”. In:J. Math. Biol.65 (2011), pp. 553–580. \\
1.  Marc A. Suchard Forrest W. Crawford Vladimir N. Minin.“Estimation for General Birth-Death Processes”. In:Journal of the AmericanStatistical Association109.506 (2014), pp. 730–747.doi:10.1080/01621459.2013.866565. \\
1.  F. Gerber.“A parallel version of ’scipy.optimize.minimize(method=’L-BFGS-B’”. In: (2020).doi:10.5281/zenodo.3888570.
