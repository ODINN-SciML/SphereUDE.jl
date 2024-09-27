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
@kwdef struct SphereParameters{F <: AbstractFloat, I <: Int} <: AbstractParameters
    tmin::F
    tmax::F
    u0::Union{Vector{F}, Nothing}
    ωmax::F
    reg::Union{Nothing, Array}
    train_initial_condition::Bool
    multiple_shooting::Bool
    niter_ADAM::I
    niter_LBFGS::I
    reltol::F
    abstol::F
    solver::OrdinaryDiffEqCore.OrdinaryDiffEqAlgorithm = Tsit5()
    sensealg::SciMLBase.AbstractAdjointSensitivityAlgorithm = QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))
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
    θ::ComponentVector
    u0::Vector{F}
    U::Lux.AbstractLuxLayer
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
    # AD differentiation mode used in regulatization
    diff_mode::Union{Nothing, String} = nothing

    # Include this in the constructor
    # @assert (order == 0) || (!isnothing(diff_mode)) "Diffentiation methods needs to be provided for regularization with order larger than zero." 
end