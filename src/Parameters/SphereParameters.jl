export SphereParameters, AbstractParameters

abstract type AbstractParameters end

"""
Training parameters
"""
@kwdef struct SphereParameters{F<:AbstractFloat,I<:Int,ADJ<:AbstractAdjointMethod} <: AbstractParameters
    tmin::F
    tmax::F
    u0::Union{Vector{F},Nothing}
    ωmax::F
    reg::Union{Nothing,Array}
    train_initial_condition::Bool
    multiple_shooting::Bool
    weighted::Bool
    niter_ADAM::I
    ADAM_learning_rate::F
    niter_LBFGS::I
    reltol::F
    abstol::F
    n_quadrature::I
    solver::OrdinaryDiffEqCore.OrdinaryDiffEqAlgorithm
    adtype::Optimization.AbstractADType
    sensealg::Union{SciMLBase.AbstractAdjointSensitivityAlgorithm, ADJ}
    pretrain::Bool
    hyperparameter_balance::Bool
    verbose::Bool
    verbose_step::I
end

function SphereParameters(;
    tmin::F,
    tmax::F,
    u0::Union{Vector{F},Nothing},
    ωmax::F,
    reg::Union{Nothing,Array} = nothing,
    train_initial_condition::Bool = false,
    multiple_shooting::Bool = false,
    weighted::Bool = false,
    niter_ADAM::I = 2000,
    ADAM_learning_rate::F = 0.001,
    niter_LBFGS::I = 2000,
    reltol::F = 1e-6,
    abstol::F = 1e-6,
    n_quadrature::I = 100,
    solver::OrdinaryDiffEqCore.OrdinaryDiffEqAlgorithm = Tsit5(),
    adtype::Optimization.AbstractADType = AutoZygote(),
    sensealg::Union{SciMLBase.AbstractAdjointSensitivityAlgorithm, ADJ} =
        QuadratureAdjoint(autojacvec = ReverseDiffVJP(true)),
    pretrain::Bool = false,
    hyperparameter_balance::Bool = false,
    verbose::Bool = true,
    verbose_step::I = 100,
    ) where {F<:AbstractFloat,I<:Int,ADJ<:AbstractAdjointMethod}

    # @assert

    ft = typeof(tmin)
    it = typeof(niter_ADAM)
    gt = isa(sensealg, SciMLBase.AbstractAdjointSensitivityAlgorithm) ? SphereBackSolveAdjoint : typeof(sensealg)

    params = SphereParameters{ft,it,gt}(
        tmin, tmax, u0, ωmax, reg, train_initial_condition, multiple_shooting, weighted,
        niter_ADAM, ADAM_learning_rate, niter_LBFGS,
        reltol, abstol, n_quadrature, solver, adtype, sensealg,
        pretrain, hyperparameter_balance,
        verbose, verbose_step
    )

    return params
end
