export SphereParameters, AbstractParameters
export SphereData, AbstractData
export Results, AbstractResult

abstract type AbstractParameters end

@kwdef struct SphereParameters{F <: AbstractFloat, I <: Int} <: AbstractParameters
    tmin::F
    tmax::F
    u0::Union{Vector{F}, Nothing}
    ωmax::F
    niter_ADAM::I
    niter_LBFGS::I
    reltol::F
    abstol::F
end

abstract type AbstractData end

@kwdef struct SphereData{F <: AbstractFloat} <: AbstractData
    times::Vector{F}
    directions::Matrix{F}
    kappas::Union{Vector{F}, Nothing}
    L::Union{Function, Nothing}
end

abstract type AbstractResult end

@kwdef struct Results{F <: AbstractFloat} <: AbstractResult
    θ_trained::ComponentVector
    U::Lux.Chain
    st::NamedTuple
    fit_times::Vector{F}
    fit_directions::Matrix{F}
end