import SpecialFunctions.loggamma

# Function used to compute the 
function greedy_grid(N_cum, finestgrid, maxbins, gr_maxbins)
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
    n = N_cum[end]

    grid_ind = fill(false, maxbins+1) # Array of booleans storing which indices to use
    grid_ind[1] = true
    grid_ind[end] = true
    incr = zeros(Float64, maxbins+1) # Array of increments from splitting at index d at each step
    incr[1] = -Inf # Would create a bin of lebesgue measure 0
    incr[end] = -Inf

    # First iteration
    i = 1
    j = maxbins+1
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
    #grid = finestgrid[grid_ind]
    #return grid
    return grid_ind
end


function greedy_grid_bayes(N_cum, finestgrid, n, gr_maxbins, a)
    function phi(i, j)
        @inbounds N_bin = N_cum[j] - N_cum[i]
        @inbounds len_bin = finestgrid[j] - finestgrid[i] # Note: p0 = len_bin on the interval 0-1
        contrib = loggamma(a*len_bin + N_bin) - loggamma(a*len_bin) - N_bin * log(len_bin)
        return contrib
    end

    function compute_bayes_increments!(incr, i, j)
        if finestgrid[i] < finestgrid[j]
            # Log-likelihood contribution 
            bayes_old = phi(i,j)
            @inbounds for l = (i+1):(j-1)
                if isapprox(N_cum[l], N_cum[i]) || isapprox(N_cum[j], N_cum[l])
                    incr[l] = 0.0
                else
                    bayes_new = phi(i,l) + phi(l,j)
                    incr[l] = bayes_new - bayes_old
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
    compute_bayes_increments!(incr, i, j)
    num_bins = 1

    # Terminate when num_bins has reached the limit or 
    # when increases to the log-likelihood are no longer possible
    while num_bins < gr_maxbins && maximum(incr) > 0
        # Update increments for indices in (i, d) and (d, j)
        d = argmax(incr)
        grid_ind[d] = true # Include finestgrid[d] in the grid
        incr[d] = -Inf # Included in grid, no longer considered for splits
        num_bins = num_bins + 1

        # Set i to maximal index < d s.t. grid_ind[i] == true
        i = findlast(grid_ind[1:d-1])
        # Set j to minimal index > d s.t. grid_ind[j] == true
        j = findfirst(grid_ind[d+1:end]) + d
        
        # Update the increments that changed with the inclusion of d in the grid
        compute_bayes_increments!(incr, i, d)
        compute_bayes_increments!(incr, d, j)
    end
    # Compute the grid we will use for the dynamic programming part
    grid = finestgrid[grid_ind]
    return grid
end


#= function greedy_grid_l2cv(N_cum, finestgrid, n, gr_maxbins)
    # Update increments between the values i and j
    function compute_loglik_increments!(incr, i, j)
        if finestgrid[i] < finestgrid[j]
            # l2 contribution:
            l2_old = 
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
    #grid = finestgrid[grid_ind]
    #return grid
    return grid_ind
end =#