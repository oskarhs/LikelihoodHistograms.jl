# LikelihoodHistograms

[![Build Status](https://github.com/oskarhs/LikelihoodHistograms.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/oskarhs/LikelihoodHistograms.jl/actions/workflows/CI.yml?query=branch%3Amaster)

Automatic regular and irregular histogram construction based on maximizing a goodness-of-fit criterion.
Supports a variety of methods including those based on leave-one-out cross-validiation, penalized maximum likelihood and fully Bayesian approaches.

## Introduction
The development of this package started with the writing of the Master's thesis "Random Histograms" (Simensen, 2025). Most notably, this package provides support for the regular and irregular random histogram models proposed in Simensen (2025), two fully Bayesian approaches to histogram construction. Since the algorithms used to fit this model can also be used to construct histograms based on other criteria, the package offers support for other goodness-of-fit criteria as well.

PUT THIS SOMEWHERE ELSE LATER.
Our implementation uses the dynamical programming algorithm of Kanazawa (1988) together with the greedy search heuristic of Rozenholc et al. (2010) to build a histogram in linearithmic time, making this package an excellent option for histogram construction for large data sets.

## Installation
Installing the package is most easily done via Julia's builtin package manager `Pkg`.
```julia
using Pkg
Pkg.add("LikelihoodHistograms")
```

## Example usage

This module exports the two functions `histogram_regular` and `histogram_irregular`, offering automatic histogram construction for a 1-dimensional data samples. A detailed exposition of all keyword arguments can be found by typing `?histogram_regular` and `?histogram_irregular` in the repl.

```julia
julia> using LikelihoodHistograms, Plots, Random
julia> x = randn(Xoshiro(5960), 10^7);
julia> H1, _ = histogram_regular(x);
julia> plot(H1)

julia> H2, _ = histogram_irregular(x);
julia> plot(H2)
```

## Supported criteria

The keyword argument `rule` determines the method used to construct the histogram for both of the histogram functions. The rule used to construct the histogram can be changed by setting `rule` equal to a string indicating the method to be used.

The default method is the fully Bayesian approach of Simensen (2025), corresponding to keyword `"bayes"`.

A detailed description of the supported methods will be added at a later point in time. A list of the supported methods, along with their corresponding keywords can be found below. 

- Regular Histograms:
    - Regular random histogram, "bayes"
    - L2 cross-validation, "l2cv"
    - Kullback-Leibler cross-validation: "klcv"
    - AIC, "aic"
    - BIC, "bic"
    - Birgé and Rozenholc's criterion, "br"
    - Normalized Maximum Likelihood, "nml"
    - Minimum Description Length, "mdl"
- Irregular Histograms:
    - Irregular random histogram, "bayes"
    - L2 cross-validation, "l2cv"
    - Kullback-Leibler cross-validation: "klcv"
    - Rozenholc et al. penalty R: "penR"
    - Rozenholc et al. penalty B: "penB"
    - Rozenholc et al. penalty B: "penB"
    - Normalized Maximum Likelihood: "nml"


## References
References

Simensen, O. H. (2025). Random Histograms. University of Oslo.

Rozenholc, Y., Mildenberger, T., & Gather, U. (2010). Combining regular and irregular histograms by penalized likelihood. Computational Statistics & Data Analysis, 54, 3313–3323. doi:10.1016/j.csda.2010.04.021

Kanazawa, Y. (1988). An optimal variable cell histogram. Communications in Statistics-Theory and Methods, 17, 1401–1422. doi:10.1080/03610928808829688

Birgé, L., & Rozenholc, Y. (2006). How many bins should be put in a regular histogram. ESAIM: Probability and Statistics, 10, 24–45. doi:10.1051/ps:2006001