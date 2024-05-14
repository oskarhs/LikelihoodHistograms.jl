using Distributions, Random, FHist, BenchmarkTools, SpecialFunctions, StatsBase

include("objective_functions.jl")

"""
    histogram_regular(h::Hist1D, rule::Str="br", maxbins=Nothing, logprior::Fu=k->-log(k))

Create a regular histogram based on optimization of a likelihood-based criterion. Returns a StatsBase.Histogram object

...
# Arguments
- `x::AbstractArray`: The data for which a histogram is to be constructed.
- `rule::Str="br"`: The criterion used to determine the optimal number of bins. Defaults to the method of 
Birgé and Rozenholc.
- `maxbins`: The maximal number of bins to be considered by the optimization criterion. Ignored if the specified
argument is not a positive integer. Defaults to `k_max = floor(Int, n/log(n))`
- `logprior=k->-log(k)`: Unnormalized logprior distribution. Only used in the case where the supplied rule
is "bayes". Defaults to Jeffreys improper prior on the positive integers, p(k) ∝ 1/k
...
"""
function histogram_regular(x::AbstractArray, rule::Str="br", maxbins=Nothing, logprior=k->-log(k))
    rule = lowercase(rule)
    if !(rule in ["aic", "bic", "br", "bayes", "mdl", "sc", "klcv", "nml"])
        rule = "br"
    end

    n = length(y)
    if isinteger(maxbins) && maxbins >= 1 
        k_max = maxbins
    else
        k_max = floor(Int, n / log(n)) # Maximal number of bins
    end
    criterion = zeros(k_max) # Criterion to be maximized/minimized depending on the penalty

    # Scale data to the interval [0,1]:
    xmin = minimum(x)
    xmax = maximum(x)
    z = (x - xmin) / (xmax - xmin)

    if rule == "aic"
        for k = 1:k_max
            N = Hist1D(z; binedges = LinRange(0,1,k+1), overflow=true).bincounts
            criterion[k] = compute_AIC(N, k, n)
        end
        k_opt = argmin(criterion) # Minimization as per usual
        H_opt = convert(Histogram, Hist1D(z; binedges = xmin+(xmax-xmin)*LinRange(0,1,k_opt+1), overflow=true))
        H_opt.weights = H_opt.weights / n # cell probabilities
        H_opt.isdensity = true
    elseif rule == "bic"
        for k = 1:k_max
            N = Hist1D(z; binedges = LinRange(0,1,k+1), overflow=true).bincounts
            criterion[k] = compute_BIC(N, k, n)
        end
        k_opt = argmin(criterion) # Minimization as per usual
        H_opt = convert(Histogram, Hist1D(z; binedges = xmin+(xmax-xmin)*LinRange(0,1,k_opt+1), overflow=true))
        H_opt.weights = H_opt.weights / n
        H_opt.isdensity = true
    elseif rule == "br"
        for k = 1:k_max
            N = Hist1D(z; binedges = LinRange(0,1,k+1), overflow=true).bincounts
            criterion[k] = compute_BR(N, k, n)
        end
        k_opt = argmax(criterion) # maximization as in Birge and Rozenholc (2006)
        H_opt = convert(Histogram, Hist1D(z; binedges = xmin+(xmax-xmin)*LinRange(0,1,k_opt+1), overflow=true))
        H_opt.weights = H_opt.weights / n
        H_opt.isdensity = true
    elseif rule == "bayes"
        for k = 1:k_max
            N = Hist1D(z; binedges = LinRange(0,1,k+1), overflow=true).bincounts
            criterion[k] = logposterior_k(N, k, a, p0, n, logprior)
        end
        k_opt = argmax(criterion)
        H_opt = convert(Histogram, Hist1D(z; binedges = xmin+(xmax-xmin)*LinRange(0,1,k_opt+1), overflow=true))
        H_opt.weights = (H_opt.bincounts .+ 0.5) / (0.5*k_opt + n) # Bayes estimate of cellprob
        H_opt.isdensity = true
    elseif rule == "mdl"
        for k = 1:k_max
            N = Hist1D(z; binedges = LinRange(0,1,k+1), overflow=true).bincounts
            criterion[k] = compute_MDL(N, k, a, p0, n, logprior)
        end
        k_opt = argmax(criterion)
        H_opt = convert(Histogram, Hist1D(z; binedges = xmin+(xmax-xmin)*LinRange(0,1,k_opt+1), overflow=true))
        H_opt.weights = H_opt.bincounts / n
        H_opt.isdensity = true
    elseif rule == "sc"
        for k = 1:k_max
            N = Hist1D(z; binedges = LinRange(0,1,k+1), overflow=true).bincounts
            criterion[k] = compute_SC(N, k, a, p0, n, logprior)
        end
        k_opt = argmax(criterion)
        H_opt = convert(Histogram, Hist1D(z; binedges = xmin+(xmax-xmin)*LinRange(0,1,k_opt+1), overflow=true))
        H_opt.weights = H_opt.bincounts / n
        H_opt.isdensity = true
    elseif rule == "klcv"
        for k = 1:k_max
            N = Hist1D(z; binedges = LinRange(0,1,k+1), overflow=true).bincounts
            criterion[k] = compute_KLCV(N, k, a, p0, n, logprior)
        end
        k_opt = argmax(criterion)
        H_opt = convert(Histogram, Hist1D(z; binedges = xmin+(xmax-xmin)*LinRange(0,1,k_opt+1), overflow=true))
        H_opt.weights = H_opt.bincounts / n
        H_opt.isdensity = true
    elseif rule == "klcv"
        for k = 1:k_max
            N = Hist1D(z; binedges = LinRange(0,1,k+1), overflow=true).bincounts
            criterion[k] = compute_NML(N, k, a, p0, n, logprior)
        end
        k_opt = argmax(criterion)
        H_opt = convert(Histogram, Hist1D(z; binedges = xmin+(xmax-xmin)*LinRange(0,1,k_opt+1), overflow=true))
        H_opt.weights = H_opt.bincounts / n
        H_opt.isdensity = true
    end
    return H_opt
end
