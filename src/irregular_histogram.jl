using FHist, StatsBase, Plots, Distributions
using SpecialFunctions

function dynamic_algorithm(phi::Function, n::Int, k_max::Int)
    cum_weight = Matrix{Float64}(undef, n, k_max)
    ancestor = zeros(Int64, n, k_max)
    weight = Matrix{Float64}(undef, n+1, n+1)

    function optimal_path!(ancestor, cum_weight, k)
        ancestor0 = Array{Int64}(undef, n-k+1)
        cum_weight0 = Array{Float64}(undef, n-k+1)

        @inbounds for i = k:n
            obj = cum_weight[(k-1):(i-1), k-1] .+ weight[k:i, i+1]
            ancestor0[i-k+1] = argmax(obj)
            cum_weight0[i-k+1] = obj[ancestor0[i-k+1]]
        end
        #= @inbounds for i = k:n
            ancestor[i, k] = ancestor0[i-k+1] + (k-2)
            cum_weight[i, k] = cum_weight0[i-k+1]
        end =#
        ancestor[k:n, k] = ancestor0 .+ (k-2)
        cum_weight[k:n, k] = cum_weight0
    end

    # Compute weights for each possible interval
    for i in 1:n
        for j in (i+1):(n+1)
            @inbounds weight[i, j] = phi(i, j)
        end
    end

    # Compute cumulative weights
    cum_weight[:,1] = weight[1,2:n+1]
    for k in 2:k_max
        optimal_path!(ancestor, cum_weight, k)
    end
    optimal = cum_weight[n,:] # Get weight function for each partition

    return optimal, ancestor
end

function compute_bounds(ancestor, grid, k)
    L = [size(ancestor)[1]]
    for i = k:-1:1
        pushfirst!(L, ancestor[L[1],i])
    end
    bounds = grid[L .+ 1]
    return bounds
end


# Function used to build a coarser grid using the greedy algorithm of Rozenholc et al.
function greedy_grid(N_cum, finestgrid, n, gr_maxbins)

    # Update increments between the values i and j
    function compute_loglik_increments!(incr, i, j)
        if finestgrid[i] < finestgrid[j]
            # Log-likelihood contribution 
            loglik_old = (N_cum[j] - N_cum[i]) * log((N_cum[j]-N_cum[i])/(n*(finestgrid[j]-finestgrid[i])))
            @inbounds for l = (i+1):(j-1)
                if isapprox(N_cum[l], N_cum[i]) || isapprox(N_cum[j], N_cum[l])
                    incr[l] = 0.0
                else
                    loglik_new = (N_cum[l] - N_cum[i]) * log((N_cum[l]-N_cum[i])/(n*(finestgrid[l]-finestgrid[i]))) +
                            (N_cum[j] - N_cum[l]) * log((N_cum[j]-N_cum[l])/(n*(finestgrid[j]-finestgrid[l])))
                    incr[l] = loglik_new - loglik_old
                end
            end
        end
    end
    grid_ind = fill(false, n+1) # Array of booleans storing which indices to use
    grid_ind[1] = true
    grid_ind[n+1] = true
    incr = zeros(Float64, n+1) # Array of increments from splitting at index d at each step
    incr[1] = -Inf # Would create a bin of lebesgue measure 0
    incr[n+1] = -Inf

    # First iteration
    i = 1
    j = n+1
    compute_loglik_increments!(incr, i, j)
    num_bins = 1

    # Terminate when num_bins has reached the limit or 
    # when increases to the log-likelihood are no longer possible
    while num_bins < gr_maxbins && maximum(incr) > 0
        # Update increments for indices in (i, d) and (d, j)
        d = argmax(incr)
        grid_ind[d] = true # Include finestgrid[d] in the grid
        incr[d] = -Inf # Included in grid
        num_bins = num_bins + 1

        # Set i to maximal index < than d s.t. grid_ind[i] == true
        i = findlast(grid_ind[1:d-1])
        # Set j to minimal index > than d s.t. grid_ind[j] == true
        j = findfirst(grid_ind[d+1:end]) + d

        compute_loglik_increments!(incr, i, d)
        compute_loglik_increments!(incr, d, j)
    end
    # Compute the grid we will use for the dynamic programming part
    grid = finestgrid[grid_ind]
    return grid
end


function phi_penB(i, j, N_cum, grid)
    @inbounds N_bin = N_cum[j] - N_cum[i]
    @inbounds len_bin = grid[j] - grid[i]
    contrib = N_bin * log(N_bin / len_bin) # Contribution of the given bin to log-likelihood
    return contrib
end

function phi_bayes(i, j, N_cum, grid, a)
    @inbounds N_bin = N_cum[j] - N_cum[i]
    @inbounds len_bin = grid[j] - grid[i] # Note: p0 = len_bin on the interval 0-1
    contrib = loggamma(a*len_bin + N_bin) - loggamma(a*len_bin) - N_bin * log(len_bin)
    return contrib
end

function phi_penR(i, j, N_cum, grid, n)
    @inbounds N_bin = N_cum[j] - N_cum[i]
    @inbounds len_bin = grid[j] - grid[i]
    contrib = N_bin * log(N_bin / len_bin) - 0.5 * N_bin / (n*len_bin)
    return contrib
end

function histogram_irregular(x::AbstractArray; rule::String="penB", right::Bool=true,
                            greedy=true,
                            maxbins::Int=-1, logprior=k->-log(k), a=1.0)
    rule = lowercase(rule)
    if !(rule in ["pena", "penb", "penr", "bayes"])
        rule = "penb" # Set penalty to default
    end 

    xmin = minimum(x)
    xmax = maximum(x)
    z = @. (x - xmin) / (xmax - xmin)
    y = sort(z)
    n = length(y)

    # Calculate gridpoints (left-open grid, breaks at data points)
    finestgrid = zeros(n+1)
    if right
        finestgrid[1] = y[1] - eps()
        finestgrid[2] = 0.5*(y[1]+y[2])
        finestgrid[3:n+1] = y[2:n] .+ eps()
    else
        finestgrid[1:n-1] = y[1:n-1] .- eps()
        finestgrid[n] = 0.5 * (y[n] - y[n-1])
        finestgrid[n+1] = y[n] + eps()
    end


    N = Hist1D(y; binedges=finestgrid).bincounts
    pushfirst!(N, 0)
    N_cum = cumsum(N)
    if greedy
        gr_maxbins = max(floor(Int, n^(1.0/3.0)), 100)
        grid = greedy_grid(N_cum, finestgrid, n, gr_maxbins)
        k_max = length(grid) - 1
        N = Hist1D(y; binedges=grid).bincounts
        pushfirst!(N, 0)
        N_cum = cumsum(N)
        #println(N_cum)
    else
        k_max = n
        grid = finestgrid
    end

    if rule in ["pena", "penb"]
        phi = (i,j) -> phi_penB(i, j, N_cum, grid)
    elseif rule == "bayes"
        phi = (i,j) -> phi_bayes(i, j, N_cum, grid, a)
    elseif rule == "penr"
        phi = (i,j) -> phi_penR(i, j, N_cum, grid, n)
    end

    optimal, ancestor = dynamic_algorithm(phi, k_max, k_max)
    psi = zeros(k_max)
    if rule == "penb"    
        @inbounds for k = 1:k_max
            psi[k] = -logabsbinomial(n-1, k-1)[1] - k - log(k)^(2.5)
        end
    elseif rule == "bayes"
        @inbounds for k = 1:k_max
            psi[k] = logprior(k) - logabsbinomial(n-1, k-1)[1] + loggamma(a) - loggamma(a + n)
        end
    elseif rule == "penr"
        @inbounds for k = 1:k_max
            psi[k] = -logabsbinomial(n-1, k-1)[1] - k - log(k)^(2.5)
        end
    elseif rule == "pena"
        @inbounds for k = 1:k_max
            psi[k] = -logabsbinomial(n-1, k-1)[1] - k - 2.0*log(k) -
                    2.0 * sqrt(1.0*0.5*(k-1)*(logabsbinomial(n-1, k-1)[1] + 1.0*log(k)))
        end
    end
    k_opt = argmax(optimal + psi)
    criterion_opt = optimal[k_opt] + psi[k_opt]
    println("k_opt = $k_opt")
    println("criterion_opt = $criterion_opt")

    bin_edges_norm = compute_bounds(ancestor, grid, k_opt)
    bin_edges =  xmin .+ (xmax - xmin) * bin_edges_norm
    #println(bin_edges)
    H = convert(Histogram, Hist1D(x; binedges=bin_edges)) # replace this
    p0 = bin_edges_norm[2:end] - bin_edges_norm[1:end-1]
    if rule == "bayes"
        H.weights = (H.weights .+ a*p0) ./ ((n + a)*(bin_edges[2:end] - bin_edges[1:end-1]))
    else
        H.weights = H.weights ./ (n * (bin_edges[2:end] - bin_edges[1:end-1]) )
    end 
    H.isdensity = true
    return H, criterion_opt
end


function test()
    x = rand(Laplace(), 10^4)
    H, criterion_opt = histogram_irregular(x; rule="pena", greedy=true)
    #println(H)
    p = plot(H, alpha=0.5)
    xlims!(-4.0, 4.0)
    t = LinRange(-4.0, 4.0, 1000)
    plot!(p, t, pdf.(Laplace(), t))
    #histogram!(x, normalize=:pdf, alpha=0.5)
    display(p)
end

test()