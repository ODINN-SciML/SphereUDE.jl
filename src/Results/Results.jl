export Results, AbstractResult

abstract type AbstractResult end

"""
Final results
"""
@kwdef struct Results{F<:AbstractFloat} <: AbstractResult
    θ::ComponentVector
    u0::Vector{F}
    U::Lux.AbstractLuxLayer
    st::NamedTuple
    fit_times::Vector{F}
    fit_directions::Matrix{F}
    fit_rotations::Matrix{F}
    losses::Vector{F}
end