export AbstractKernel, GaussianKernel

abstract type AbstractKernel end

struct Kernel <: AbstractKernel
    mean::Function
    cov::Function
end

struct GaussianKernel{F<:AbstractFloat} <: AbstractKernel
    σ::F
    τ::F
end

struct MaternKernel{F<:AbstractFloat} <: AbstractKernel
    ν::F
end

"""
Kernel is a bi-variate function k(x, y) such that 
 - Is non-negative
 - Is semi-definite positive
 - Etc
"""
function GaussianKernel(
    σ::F,
    τ::F,
) where {F<:AbstractFloat}

    mean = x -> 0.0
    cov = (x, y) -> σ^2 * exp(-(x - y)^2 / (2τ^2))

    return Kernel(mean, cov)
end