using Distributions, Random, FHist, SpecialFunctions, StatsBase, Plots

include("objective_functions.jl")

"""
    histogram_regular(h::Hist1D, rule::Str="br", maxbins=Nothing, logprior=k->-log(k))

Create a regular histogram based on optimization of a likelihood-based criterion. Returns a StatsBase.Histogram object

...
# Arguments
- `x::AbstractArray`: The data for which a histogram is to be constructed.
- `rule::String="br"`: The criterion used to determine the optimal number of bins. Defaults to the method of 
Birgé and Rozenholc.
- `maxbins`: The maximal number of bins to be considered by the optimization criterion. Ignored if the specified
argument is not a positive integer. Defaults to `k_max = floor(Int, n/log(n))`
- `logprior=k->-log(k)`: Unnormalized logprior distribution. Only used in the case where the supplied rule
is "bayes". Defaults to Jeffreys improper prior on the positive integers, p(k) ∝ 1/k
...
"""
function histogram_regular(x::AbstractArray; rule::String="br", maxbins::Integer=-1, logprior=k->-log(k))
    rule = lowercase(rule)
    if !(rule in ["aic", "bic", "br", "bayes", "mdl", "sc", "klcv", "nml"])
        rule = "br"
    end

    n = length(x)
    if maxbins >= 1 
        k_max = maxbins
    else
        k_max = max(floor(Int, n / log(n)), 10^3) # Default maximal number of bins
    end
    criterion = zeros(k_max) # Criterion to be maximized/minimized depending on the penalty

    # Scale data to the interval [0,1]:
    xmin = minimum(x)
    xmax = maximum(x)
    z = @. (x - xmin) / (xmax - xmin)

    if rule == "aic"
        for k = 1:k_max
            N = Hist1D(z; binedges = LinRange(0,1,k+1), overflow=true).bincounts
            criterion[k] = -compute_AIC(N, k, n) # Note: negative of AIC is computed
        end
    elseif rule == "bic"
        for k = 1:k_max
            N = Hist1D(z; binedges = LinRange(0,1,k+1), overflow=true).bincounts
            criterion[k] = -compute_BIC(N, k, n) # Note: negative of BIC is computed
        end
    elseif rule == "br"
        for k = 1:k_max
            N = Hist1D(z; binedges = LinRange(0,1,k+1), overflow=true).bincounts
            criterion[k] = compute_BR(N, k, n)
        end
    elseif rule == "bayes"
        for k = 1:k_max
            N = Hist1D(z; binedges = LinRange(0,1,k+1), overflow=true).bincounts
            criterion[k] = logposterior_k(N, k, 0.5*k, ones(k)/k, n, logprior)
        end
    elseif rule == "mdl"
        for k = 1:k_max
            N = Hist1D(z; binedges = LinRange(0,1,k+1), overflow=true).bincounts
            criterion[k] = compute_MDL(N, k, n)
        end
    elseif rule == "sc"
        for k = 1:k_max
            N = Hist1D(z; binedges = LinRange(0,1,k+1), overflow=true).bincounts
            criterion[k] = compute_SC(N, k, n)
        end
    elseif rule == "klcv"
        for k = 1:k_max
            N = Hist1D(z; binedges = LinRange(0,1,k+1), overflow=true).bincounts
            criterion[k] = compute_KLCV(N, k, n)
        end
    elseif rule == "klcv"
        for k = 1:k_max
            N = Hist1D(z; binedges = LinRange(0,1,k+1), overflow=true).bincounts
            criterion[k] = compute_NML(N, k, n)
        end
    end

    # Create a StatsBase.Histogram object with the chosen number of bins
    k_opt = argmax(criterion)
    H_opt = convert(Histogram, Hist1D(x; binedges = xmin .+ (xmax-xmin)*LinRange(0,1,k_opt+1), overflow=true))
    if rule == "bayes"
        H_opt.weights = k_opt * (H_opt.weights .+ 0.5) / (0.5*k_opt + n) # Estimated density
    else
        H_opt.weights = k_opt * H_opt.weights / n # Estimated density
    end
    H_opt.isdensity = true
    return H_opt
end


function test_reghist()
    x = rand(Normal(), 10^7)
    #H = Hist1D(x)
    #plot(H)
    H = histogram_regular(x; rule="bayes")
    println(length(H.weights))
    H2 = histogram_regular(x; rule="bic")
    println(length(H2.weights))
    plot(H, alpha=0.5)
end

test_reghist()