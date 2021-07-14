# Background

I undertook a research project under The University of Melbourne's Mathematics and Statistics Department in Dec 2020 - Feb 2021. This repository has what I partially implemented
in a paper by [Crawford et al.]

## Introduction
   A birth death process (BDP) is a continuous time Markov chain which models the number of particles in a system. This number can either transition up or down one step at random according to a sequence of birth-rates and death-rates, depending on the population.
   The model of interest is a type of logistic growth model with birth-rates and death-rates given respectively by 
   ```math
   \begin{align}
             \lambda_k &= k^2 \lambda e ^{-\alpha k} \\
             \mu_k &= k\mu,
   \end{align}
   ```
   The aim of this project was to implement a novel way of finding transition probabilities as described in [Crawford et al.](https://www.researchgate.net/publication/51957281_Estimation_for_General_Birth-Death_Processes), then apply an EM (expectation maximisation algorithm).
   In the end a direct method of likelihood optimisation was used instead, and a comparison to matrix exponential methods was made.

## Acknowledgements
I wish to thank my supervisors Dr Sophie Hautphenne and Dr Brendan Patch for their time and work in supervising me, as well as the School of Mathematics and Statistics for this wonderful opportunity. This experience has been quite invaluable to me, as it has allowed me to both develop my ability and grow my appreciation for scientific research.
    
## References
1. A. Kraus A.C. Davidson S. Hautphenne.“Parameter estimation for discretely observed linear birth-and-death processes”. In:The Interna-tional Biometric Society(2020).doi:10.1111/biom.13282. \\
1.  Marc A. Suchard Forrest W. Crawford.“Transition probabilities for general birth–death processes with applications in ecology, genetics, andevolution”. In:J. Math. Biol.65 (2011), pp. 553–580. \\
1.  Marc A. Suchard Forrest W. Crawford Vladimir N. Minin.“Estimation for General Birth-Death Processes”. In:Journal of the AmericanStatistical Association109.506 (2014), pp. 730–747.doi:10.1080/01621459.2013.866565. \\
1.  F. Gerber.“A parallel version of ’scipy.optimize.minimize(method=’L-BFGS-B’”. In: (2020).doi:10.5281/zenodo.3888570.
