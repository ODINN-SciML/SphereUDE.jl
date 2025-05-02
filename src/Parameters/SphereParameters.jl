export SphereParameters, AbstractParameters

abstract type AbstractParameters end

"""
Training parameters
"""
@kwdef struct SphereParameters{F<:AbstractFloat,I<:Int} <: AbstractParameters
    tmin::F
    tmax::F
    u0::Union{Vector{F},Nothing}
    ωmax::F
    reg::Union{Nothing,Array} = nothing
    train_initial_condition::Bool = false
    multiple_shooting::Bool = false
    weighted::Bool = false
    niter_ADAM::I = 2000
    ADAM_learning_rate::F = 0.001
    niter_LBFGS::I = 2000
    reltol::F = 1e-6
    abstol::F = 1e-6
    n_quadrature::I = 100
    solver::OrdinaryDiffEqCore.OrdinaryDiffEqAlgorithm = Tsit5()
    adtype::Optimization.AbstractADType = AutoZygote()
    sensealg::SciMLBase.AbstractAdjointSensitivityAlgorithm =
        QuadratureAdjoint(autojacvec = ReverseDiffVJP(true))
    customgrad::Bool = false
    pretrain::Bool = false
    hyperparameter_balance::Bool = false
    verbose::Bool = true
    verbose_step::I = 100
end
