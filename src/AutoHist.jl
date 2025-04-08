module AutoHist

export histogram_regular, histogram_irregular

include(joinpath("regular", "objective_functions.jl"))

include(joinpath("irregular" ,"greedy_grid.jl"))
include(joinpath("irregular", "dynamic_algorithm.jl"))

include("utils.jl")

include(joinpath("regular", "regular_histogram.jl"))
include(joinpath("irregular", "irregular_histogram.jl"))

end
