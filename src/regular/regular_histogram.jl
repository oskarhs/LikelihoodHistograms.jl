using StatsBase

include("objective_functions.jl")
include("utils.jl")

"""
    histogram_regular(x::AbstractVector{<:Real}; rule::Str="bayes", right::Bool=true, maxbins::Int=1000, logprior::Function=k->0.0, a::Union{Real,Function}=1.0)

Create a regular histogram based on optimization criterion from Bayesian probability, penalized likelihood or LOOCV.
Returns a tuple where the first argument is a StatsBase.Histogram object, the second the value of the maxinized criterion.

...
# Arguments
- `x`: 1D vector of data for which a histogram is to be constructed.
- `rule`: The criterion used to determine the optimal number of bins. Defaults to the method Bayesian method of Simensen et al. (2025)
- `right`: Boolean indicating whether the drawn intervals should be right-inclusive or not. Defaults to `true`.
- `maxbins`: The maximal number of bins to be considered by the optimization criterion. Ignored if the specified argument is not a positive integer. Defaults to `maxbins=1000`
- `logprior`: Unnormalized logprior distribution of the number k of bins. Only used in the case where the supplied rule is `"bayes"`. Defaults to a uniform prior.
- `a`: Specifies Dirichlet concentration parameter in the Bayesian histogram model. Can either be a fixed positive number or a function computing aₖ for different values of k. Defaults to `1.0` if not supplied. Uses default if suppled value is negative.

# Examples
```
julia> x = [0.037, 0.208, 0.189, 0.656, 0.45, 0.846, 0.986, 0.751, 0.249, 0.447]
julia> H1, criterion1 = histogram_regular(x)
julia> H2, criterion2 = histogram_regular(x; logprior=k->-log(k), a=k->0.5*k)
```
...
"""
function histogram_regular(x::AbstractVector{<:Real}; rule::String="bayes", right::Bool=true, maxbins::Int=1000, logprior::Function=k->0.0, a::Union{Real,Function}=1.0)
    rule = lowercase(rule)
    if !(rule in ["aic", "bic", "br", "bayes", "mdl", "sc", "klcv", "nml", "l2cv"])
        rule = "bayes"
    end
    if rule == "bayes"
        if !isa(a, Function) # create constant function if typeof(a) <: Real
            if a ≤ 0.0
                a_func = k -> 1.0
            else 
                a_func = k -> a
            end
        else 
            a_func = k -> ifelse(a(k) > 0.0, a(k), 1.0)
        end
    end

    n = length(x)
    if maxbins < 1
        maxbins = 10^3 # Default maximal number of bins
    end
    k_max = min(ceil(Int, 4.0*n / log(n)^2), maxbins)

    criterion = zeros(k_max) # Criterion to be maximized depending on the specified rule

    # Scale data to the interval [0,1]:
    xmin = minimum(x)
    xmax = maximum(x)
    z = @. (x - xmin) / (xmax - xmin)

    if rule == "aic"
        for k = 1:k_max
            N = bin_regular(z, 0.0, 1.0, k, right)
            criterion[k] = compute_AIC(N, k, n) # Note: negative of AIC is computed
        end
    elseif rule == "bic"
        for k = 1:k_max
            N = bin_regular(z, 0.0, 1.0, k, right)
            criterion[k] = compute_BIC(N, k, n) # Note: negative of BIC is computed
        end
    elseif rule == "br"
        for k = 1:k_max
            N = bin_regular(z, 0.0, 1.0, k, right)
            criterion[k] = compute_BR(N, k, n)
        end
    elseif rule == "bayes"
        for k = 1:k_max
            aₖ = a_func(k)
            N = bin_regular(z, 0.0, 1.0, k, right)
            criterion[k] = logposterior_k(N, k, ones(k)/k, aₖ, n, logprior)
        end
    elseif rule == "mdl"
        for k = 1:k_max
            N = bin_regular(z, 0.0, 1.0, k, right)
            criterion[k] = compute_MDL(N, k, n)
        end
    elseif rule == "sc"
        for k = 1:k_max
            N = bin_regular(z, 0.0, 1.0, k, right)
            criterion[k] = compute_SC(N, k, n)
        end
    elseif rule == "klcv"
        for k = 1:k_max
            N = bin_regular(z, 0.0, 1.0, k, right)
            criterion[k] = compute_KLCV(N, k, n)
        end
    elseif rule == "l2cv"
        for k = 1:k_max
            N = bin_regular(z, 0.0, 1.0, k, right)
            criterion[k] = compute_L2CV(N, k, n)
        end
    elseif rule == "nml"
        for k = 1:k_max
            N = bin_regular(z, 0.0, 1.0, k, right)
            criterion[k] = compute_NML(N, k, n)
        end
    end

    # Create a StatsBase.Histogram object with the chosen number of bins
    k_opt = argmax(criterion)
    N = bin_regular(z, 0.0, 1.0, k_opt, right)
    if right
        H_opt = Histogram(LinRange(xmin, xmax, k_opt+1), N, :right, true)
    else
        H_opt = Histogram(LinRange(xmin, xmax, k_opt+1), N, :left, true)
    end
    if rule == "bayes"
        aₖ = a_func(k_opt)
        H_opt.weights = k_opt/(xmax-xmin) * (N.+ aₖ/k_opt) / (aₖ + n) # Estimated density
    else
        H_opt.weights = k_opt/(xmax-xmin) * N.weights / n # Estimated density
    end
    return H_opt, criterion[k_opt]
end

# Create regular grid with right- or left-inclusive intervals.
#= function fill_edges!(edges, k, right)
    edges[1] = -eps()
    edges[k+1] = 1.0+eps()
    if right
        edges[2:k] = LinRange(1.0/k+eps(), 1.0-1.0/k+eps(), k-1)
    else
        edges[2:k] = LinRange(1.0/k-eps(), 1.0-1.0/k-eps(), k-1)
    end
end =#