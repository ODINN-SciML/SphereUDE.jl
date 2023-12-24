export SphereParameters, AbstractParameters
export SphereData, AbstractData

abstract type AbstractParameters end

@kwdef struct SphereParameters{F <: AbstractFloat} <: AbstractParameters
    tmin::F
    tmax::F
    u0::Union{Vector{F}, Nothing}
    ωmax::F
    reltol::F
    abstol::F
end

abstract type AbstractData end

@kwdef struct SphereData{F <: AbstractFloat} <: AbstractData
    times::Vector{F}
    directions::Matrix{F}
    kappas::Union{Vector{F}, Nothing}
end