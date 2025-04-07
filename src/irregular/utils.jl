# Specialized function to perform regular binning quickly for known min, max.
# Supports both left- and right-inclusive binning
function bin_regular(x::AbstractVector{<:Real}, xmin::Real, xmax::Real, k::Int, right::Bool)
    R = xmax - xmin
    bincounts = zeros(Float64, k)
    edges_inc = k/R
    if right
        for val in x
            idval = min(k-1, floor(Int, (val-xmin)*edges_inc+eps())) + 1
            @inbounds bincounts[idval] += 1.0
        end
    else
        for val in x
            idval = max(0, floor(Int, (val-xmin)*edges_inc-eps())) + 1
            @inbounds bincounts[idval] += 1.0
        end
    end
    return bincounts
end

function bin_irregular(x::AbstractVector{<:Real}, edges::AbstractVector{<:Real}, right::Bool)
    bincounts = zeros(Float64, length(edges)-1)
    if right
        for val in x
            idval = max(1, searchsortedfirst(edges, val) - 1)
            @inbounds bincounts[idval] += 1.0
        end
    else
        for val in x
            idval = min(k, searchsortedlast(edges, val))
            @inbounds bincounts[idval] += 1.0
        end
    end
    return bincounts
end