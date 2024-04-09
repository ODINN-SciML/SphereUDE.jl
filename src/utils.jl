export sigmoid_cap, relu_cap, step_cap
export cart2sph
export AbstractNoise, FisherNoise
export quadrature

# Normalization of the NN. Ideally we want to do this with L2 norm .

"""
    sigmoid_cap(x; ω₀=1.0)
"""
function sigmoid_cap(x; ω₀=1.0)
    min_value = - ω₀
    max_value = + ω₀
    return min_value + (max_value - min_value) / ( 1.0 + exp(-x) )
end

"""
    relu_cap(x; ω₀=1.0)
"""
function relu_cap(x; ω₀=1.0)
    min_value = - ω₀
    max_value = + ω₀
    return min_value + (max_value - min_value) * max(0.0, min(x, 1.0))
end

"""
    cart2sph(X::AbstractArray{<:Number}; radians::Bool=true)

Convert cartesian coordinates to spherical
"""
function cart2sph(X::AbstractArray{<:Number}; radians::Bool=true)
    @assert size(X)[1] == 3 "Input array must have three rows."
    Y = mapslices(x -> [angle(x[1] + x[2]*im), asin(x[3])] , X, dims=1)
    if !radians
        Y *= 180. / π
    end
    return Y
end

"""
Add Fisher noise to matrix of three dimensional unit vectors

This is carried by the definition of type FisherNoise <: AbstractNoise and 
extending the base definition +(,) to allow the simple syntax 

X_noise = X_noiseless + FisherNoise(kappa=200.) 
"""
abstract type AbstractNoise end

@kwdef struct FisherNoise{F <: AbstractFloat} <: AbstractNoise
    kappa::Union{F, Vector{F}}
end

function Base.:(+)(X::Array{F, 2}, ϵ::N) where {F <: AbstractFloat, N <: AbstractNoise}
    if typeof(ϵ.kappa) <: F
        return mapslices(x -> rand(sampler(VonMisesFisher(x/norm(x), ϵ.kappa)), 1), X, dims=1)
    else
        @assert length(ϵ.kappa) == size(X)[2] "Signal and noise must have same dimensions."
        Y = similar(X)
        for i in 1:size(X)[2]
            x = X[:,i]
            Y[:,i] = rand(sampler(VonMisesFisher(x/norm(x), ϵ.kappa[i])), 1)
        end
        return Y
    end
end

"""
    quadrature_integrate

Numerical integral using Gaussian quadrature
"""
function quadrature(f::Function, t₀, t₁, n_nodes::Int)
    ignore() do
        # Ignore AD here since FastGaussQuadrature is using mutating arrays
        nodes, weigths = gausslegendre(n_nodes)
    end
    nodes = (t₀+t₁)/2  .+ nodes * (t₁-t₀)/2
    weigths = (t₁-t₀) / 2 * weigths
    return dot(weigths, f.(nodes))
end