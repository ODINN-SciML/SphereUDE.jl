export Results, EnsambleResult, AbstractResult

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
