export Results, AbstractResult

abstract type AbstractResult end

"""
Final results
"""
@kwdef struct Results{F<:AbstractFloat} <: AbstractResult
    params::SphereParameters
    θ::ComponentVector
    u0::Union{Vector{F},SVector{3,F}}
    U::Lux.AbstractLuxLayer
    st::NamedTuple
    fit_times::Vector{F}
    fit_directions::Matrix{F}
    fit_rotations::Matrix{F}
    losses::Vector{F}
    losses_dict::Dict
end
