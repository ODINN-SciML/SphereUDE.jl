export quadrature, central_fdm, complex_step_differentiation

"""
    quadrature_integrate

Numerical integral using Gaussian quadrature
"""
function quadrature(f::Function, t₀::F, t₁::F, n_quadrature::Int) where {F <: AbstractFloat}
    nodes, weigths = quadrature(t₀, t₁, n_quadrature)
    return dot(weigths, f.(nodes))
end

function quadrature(t₀::F, t₁::F, n_quadrature::Int) where {F <: AbstractFloat}
    # Ignore AD here since FastGaussQuadrature is using mutating arrays
    @ignore_derivatives nodes, weigths = gausslegendre(n_quadrature)
    # ignore() do
    #     nodes, weigths = gausslegendre(n_quadrature)
    #     # nodes .+ rand(sampler(Uniform(-0.1,0.1)), n_quadrature)
    # end
    nodes = (t₀ + t₁) / 2 .+ nodes * (t₁ - t₀) / 2
    weigths = (t₁ - t₀) / 2 * weigths
    return nodes, weigths
end

function quadrature(
    f::Function,
    t₀::F,
    t₁::F,
    nodes::Vector{F},
    method::Symbol = :Linear
    ) where {F <: AbstractFloat}
    weights = quadrature(t₀, t₁, nodes; method = method)
    return dot(weights, f.(nodes))
end

function quadrature(
    t₀::F,
    t₁::F,
    nodes::Vector{F};
    method::Symbol = :Linear
    ) where {F <: AbstractFloat}

    @assert t₀ <= minimum(nodes) <= maximum(nodes) <= t₁
    n = length(nodes)

    if method == :Vandermonde
        """
        This works quite well for numerical integration but it has the problem that returns
        (potentially) negative weights, which are not suitable for a loss function.
        """
        # Build Vandermonde matrix
        V = [nodes[i]^j for j in 0:n-1, i in 1:n]
        # Right-hand side: exact integrals of monomials
        b = [(t₁^(k + 1) - t₀^(k + 1)) / (k + 1) for k in 0:n-1]
        # solve for the weigths
        weights = V \ b
    elseif method == :Linear
        midpoints = (nodes[1:end-1] .+ nodes[2:end] ) ./ 2.0
        edges = [t₀; midpoints; t₁]
        weights = (edges[2:end] .- edges[1:end-1])
        @assert sum(weights) ≈ t₁ - t₀
    else
        @error "Method $(method) not implemented."
    end
    return weights
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
