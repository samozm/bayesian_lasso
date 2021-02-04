# bayesian_lasso
Implementation of Park &amp; Casella's Bayesian Lasso.

Julia implementation of Park & Casella (2008) is in `implementation.jl`.

Outputs histograms and credible intervals from 60000 post-burn-in (Gibbs) samples based on Park & Casella (2008), using the diabetes data used in that paper, originally from Efron et al. (2004) (https://web.stanford.edu/~hastie/StatLearnSparsity_files/DATA/diabetes.html).

Initializes the lambda value using OLS and then updates it after every 600 samples.

To actually see the graphs, `implementation.jl` needs to be run in julia's interactive mode as follows:
```
julia -i implementation.jl
```

Output histograms (with credible intervals) are in `histograms.png`

Graph of credible intervals (one graph with all intervals) is in `credible_intervals.png`
