export SphereParameters, AbstractParameters
export SphereData, AbstractData
export Results, AbstractResult

abstract type AbstractParameters end

@kwdef struct SphereParameters{F <: AbstractFloat, I <: Int} <: AbstractParameters
    tmin::F
    tmax::F
    u0::Union{Vector{F}, Nothing}
    ωmax::F
    reg::Union{Nothing,Array{Tuple{Int64, Float64, Float64}}} # order, power, λ
    niter_ADAM::I
    niter_LBFGS::I
    reg_differentiation::Union{Nothing, String}
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