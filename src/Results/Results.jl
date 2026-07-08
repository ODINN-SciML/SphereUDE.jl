export Results, EnsambleResult, CVResult, AbstractResult

abstract type AbstractResult end

"""
Final results
"""
@kwdef struct Results{F<:AbstractFloat} <: AbstractResult
    params::SphereParameters
    θ::ComponentVector
    u0::Union{Vector{F},SVector{3,F}}
    regressor::AbstractRegressor
    fit_times::Vector{F}
    fit_directions::Matrix{F}
    fit_rotations::Matrix{F}
    losses::Vector{F}
    losses_dict::Dict
end

"""
Collection of `Results` obtained from multiple resamples of the same dataset,
used for uncertainty quantification. `datasets[i]` is the resampled dataset
that `results[i]` was trained on.
"""
@kwdef struct EnsambleResult{R<:AbstractResult,D<:AbstractData} <: AbstractResult
    results::Vector{R}
    datasets::Vector{D}
end

"""
Result of a cross-validated regularization search (see [`train_cv`](@ref)).
`λ_grid` is the candidate `λ` values that were tried, and `scores[k]` holds
the per-fold validation losses for `λ_grid[k]` (so `mean(scores[k])` is the
average validation loss used to pick `best_λ`). `best_results` is the
`Results` of the final fit on the full dataset using `best_λ`, filled in
only when that final refit is run (`nothing` otherwise).
"""
@kwdef struct CVResult{F<:AbstractFloat} <: AbstractResult
    best_results::Union{Results,Nothing}
    λ_grid::Vector{F}
    scores::Vector{Vector{F}}
    best_λ::F
end
