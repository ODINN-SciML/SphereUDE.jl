export quadrature, central_fdm, complex_step_differentiation

"""
    quadrature_integrate

Numerical integral using Gaussian quadrature
"""
function quadrature(f::Function, t₀, t₁, n_quadrature::Int)
    nodes, weigths = quadrature(t₀, t₁, n_quadrature)
    return dot(weigths, f.(nodes))
end

function quadrature(t₀, t₁, n_quadrature::Int)
    # Ignore AD here since FastGaussQuadrature is using mutating arrays
    @ignore_derivatives nodes, weigths = gausslegendre(n_quadrature)
    # ignore() do
    #     nodes, weigths = gausslegendre(n_quadrature)
    #     # nodes .+ rand(sampler(Uniform(-0.1,0.1)), n_quadrature)
    # end
    nodes = (t₀+t₁)/2 .+ nodes * (t₁-t₀)/2
    weigths = (t₁-t₀) / 2 * weigths
    return nodes, weigths
end

"""
    central_fdm(f::Function, x::Float64; ϵ=0.01)

Simple central differences implementation. 

FiniteDifferences.jl does not work with AD so I implemented this manually. 
Still remains to test this with FiniteDiff.jl
"""
function central_fdm(f::Function, x::Float64, ϵ::Float64)
    return (f(x+ϵ)-f(x-ϵ)) / (2ϵ) 
end

"""
    complex_step_differentiation(f::Function, x::Float64; ϵ=1e-10)

Manual implementation of complex-step differentiation
"""
function complex_step_differentiation(f::Function, x::Float64, ϵ::Float64)
    return imag(f(x + ϵ * im)) / ϵ
end