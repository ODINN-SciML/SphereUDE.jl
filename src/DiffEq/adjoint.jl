export AbstractAdjointMethod
export SphereBackSolveAdjoint

abstract type AbstractAdjointMethod end

@kwdef struct SphereBackSolveAdjoint{
    F <: AbstractFloat,
    I <: Integer
} <: AbstractAdjointMethod
    solver::Any = Tsit5()
    reltol::F = 1e-6
    abstol::F = 1e-6
    n_quadrature::I = 100
end

@kwdef struct DummyAdjoint <: AbstractAdjointMethod
end
