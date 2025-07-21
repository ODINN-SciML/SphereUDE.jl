export numerical_integral, central_fdm, complex_step_differentiation

function numerical_integral(
    f::Function,
    t₀::F,
    t₁::F,
    quadrature::Q
    ) where {F<:AbstractFloat, Q<:AbstractQuadrature}

    nodes, weights = extract_nodes_weights(t₀, t₁, quadrature)
    return dot(weights, f.(nodes))
end

"""
    central_fdm(f::Function, x::Float64; ϵ=0.01)

Simple central differences implementation.

FiniteDifferences.jl does not work with AD so I implemented this manually.
Still remains to test this with FiniteDiff.jl
"""
function central_fdm(f::Function, x::Float64, ϵ::Float64)
    return (f(x + ϵ) - f(x - ϵ)) / (2ϵ)
end

"""
    complex_step_differentiation(f::Function, x::Float64; ϵ=1e-10)

Manual implementation of complex-step differentiation
"""
function complex_step_differentiation(f::Function, x::Float64, ϵ::Float64)
    return imag(f(x + ϵ * im)) / ϵ
end
