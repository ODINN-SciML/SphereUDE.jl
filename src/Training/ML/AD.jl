export FiniteDifferences, ComplexStepDifferentiation, LuxNestedAD
export AbstractDifferentiation

abstract type AbstractDifferentiation end

"""
Differentiation methods
"""
@kwdef struct FiniteDifferences{F <: AbstractFloat} <: AbstractDifferentiation
    ϵ::F = 1e-10
end

@kwdef struct ComplexStepDifferentiation{F <: AbstractFloat} <: AbstractDifferentiation
    ϵ::F = 1e-10
end

@kwdef struct LuxNestedAD <: AbstractDifferentiation
    method::Union{Nothing, String} = "ForwardDiff"
end