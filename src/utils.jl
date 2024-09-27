export sigmoid, sigmoid_cap
export relu, relu_cap
export cart2sph, sph2cart
export AbstractNoise, FisherNoise
export quadrature, central_fdm, complex_step_differentiation
export raise_warnings
export isL1reg

"""
    sigmoid_cap(x; ω₀=1.0)

Normalization of the neural network last layer
"""
function sigmoid_cap(x; ω₀=1.0)
    min_value = - ω₀
    max_value = + ω₀
    return min_value + (max_value - min_value) * sigmoid(x)
end

function sigmoid(x)
    return 1.0 / (1.0 + exp(-x))
#     if x > 0
#         return 1 / ( 1.0 + exp(-x) )
#     else
#         return exp(x) / (1.0 + exp(x))
#     end
end

function sigmoid(z::Complex)
    return 1.0 / ( 1.0 + exp(-z) )
    # if real(z) > 0
    #     return 1 / ( 1.0 + exp(-z) )
    # else
    #     return exp(z) / (1.0 + exp(z))
    # end
end

"""
    relu_cap(x; ω₀=1.0)
"""
function relu_cap(x; ω₀=1.0)
    min_value = - ω₀
    max_value = + ω₀
    return relu_cap(x, min_value, max_value)
end

function relu_cap(x, min_value::Float64, max_value::Float64)
    return min_value + (max_value - min_value) * max(0.0, min(x, 1.0))
end

"""
    relu(x::Complex)

Extension of ReLU function to complex numbers based on the complex cardioid introduced in 
Virtue et al. (2017), "Better than Real: Complex-valued Neural Nets for MRI Fingerprinting".
This function is equivalent to relu when x is real (and hence angle(x)=0 or angle(x)=π).
"""
function relu(z::Complex)
    return 0.5 * (1 + cos(angle(z))) * z
end

function relu_cap(z::Complex; ω₀=1.0)
    min_value = - ω₀
    max_value = + ω₀
    return relu_cap(z, min_value, max_value)
    # return min_value + (max_value - min_value) * relu(z - relu(z-1))
end

function relu_cap(z::Complex, min_value::Float64, max_value::Float64)
    return min_value + (max_value - min_value) * relu(z - relu(z-1))
end

"""
    cart2sph(X::AbstractArray{<:Number}; radians::Bool=true)

Convert cartesian coordinates to spherical
"""
function cart2sph(X::AbstractArray{<:Number}; radians::Bool=true)
    @assert size(X)[1] == 3 "Input array must have three rows."
    Y = mapslices(x -> [asin(x[3]), angle(x[1] + x[2]*im)] , X, dims=1)
    if !radians
        Y *= 180. / π
    end
    return Y
end


"""
    sph2cart(X::AbstractArray{<:Number}; radians::Bool=true)

Convert spherical coordinates to cartesian
"""
function sph2cart(X::AbstractArray{<:Number}; radians::Bool=true)
    @assert size(X)[1] == 2 "Input array must have two rows corresponding to Latitude and Longitude."
    if !radians
        X *= π / 180.
    end
    Y = mapslices(x -> [cos(x[1])*cos(x[2]), 
                        cos(x[1])*sin(x[2]),
                        sin(x[1])], X, dims=1)
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
    nodes = (t₀+t₁)/2 .+ nodes * (t₁-t₀)/2
    weigths = (t₁-t₀) / 2 * weigths
    return dot(weigths, f.(nodes))
end

"""
    central_fdm(f::Function, x::Float64; ϵ=0.01)

Simple central differences implementation. 

FiniteDifferences.jl does not work with AD so I implemented this manually. 
Still remains to test this with FiniteDiff.jl
"""
function central_fdm(f::Function, x::Float64; ϵ=0.01)
    return (f(x+ϵ)-f(x-ϵ)) / (2ϵ) 
end

"""
    complex_step_differentiation(f::Function, x::Float64; ϵ=1e-10)

Manual implementation of complex-step differentiation
"""
function complex_step_differentiation(f::Function, x::Float64; ϵ=1e-10)
    return imag(f(x + ϵ * im)) / ϵ
end

"""
    raise_warnings(data::AD, params::AP)

Raise warnings.
"""
function raise_warnings(data::SphereData, params::SphereParameters)
    if length(unique(data.times)) < length(data.times)
        @warn "[SphereUDE] Timeseries includes duplicated times. \n This can produce unexpected errors." 
    end
    # if !isnothing(params.reg)
    #     for reg in params.reg  
    #         if reg.diff_mode=="CS"
    #             @warn "[SphereUDE] Complex-step differentiation inside the loss function \n This just work for cases where the activation function of the neural network admits complex numbers \n Change predefined activation functions to be complex numbers."
    #         end
    #     end
    # end
    nothing
end

"""

Function to check for the presence of L1 regularization in the loss function. 
"""
function isL1reg(regs::Vector{R}) where {R <: AbstractRegularization}
    for reg in regs 
        if reg.power == 1
            return true
        end
    end
    return false
end