export SphereParameters, AbstractParameters
export SphereData, AbstractData
export Results, AbstractResult
export Regularization, AbstractRegularization

abstract type AbstractParameters end
abstract type AbstractData end
abstract type AbstractRegularization end
abstract type AbstractResult end

"""
Training parameters
"""
@kwdef struct SphereParameters{F <: AbstractFloat, I <: Int, R <: AbstractRegularization} <: AbstractParameters
    tmin::F
    tmax::F
    u0::Union{Vector{F}, Nothing}
    ωmax::F
    reg::Union{Nothing, Array{R}}
    niter_ADAM::I
    niter_LBFGS::I
    reltol::F
    abstol::F
end

"""
Spherical data information. 
"""
@kwdef struct SphereData{F <: AbstractFloat} <: AbstractData
    times::Vector{F}
    directions::Matrix{F}
    kappas::Union{Vector{F}, Nothing}
    L::Union{Function, Nothing}
end

"""
Final results
"""
@kwdef struct Results{F <: AbstractFloat} <: AbstractResult
    θ_trained::ComponentVector
    U::Lux.Chain
    st::NamedTuple
    fit_times::Vector{F}
    fit_directions::Matrix{F}
end

"""
Regularization information
"""
@kwdef struct Regularization{F <: AbstractFloat, I <: Int} <: AbstractRegularization
    order::I        # Order of derivative
    power::F        # Power of the Euclidean norm 
    λ::F            # Regularization hyperparameter
    diff_mode::Union{Nothing, String}       # AD differentiation mode used in regulatization
end