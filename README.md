# bayesian_lasso
Implementation of Park &amp; Casella's Bayesian Lasso.

Julia implementation of Park & Casella (2008) is in `implementation.jl`

Outputs histograms of 50000 post-burn-in (Gibbs) samples based on Park & Casella (2008), using the diabetes data used in that paper, originally from Efron et al. (2004).

Initializes the lambda value using OLS and then updates it after every 500 samples.

To actually see graph, needs to be run in julia's interactive mode as follows:
```
julia -i implementation.jl
```
