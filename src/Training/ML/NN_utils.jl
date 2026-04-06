export get_default_NN
export sigmoid, sigmoid_cap
export relu, relu_cap
export gelu, rbf
export scale_input, scale_norm
export fourier_feature

# Import activation function for complex extension
import Lux: relu, gelu
using Lux: Chain
# import Lux: sigmoid, relu, gelu

"""

Return a default neural network for those cases where NN has not being provided by user.
Input is scaled to [-1, 1] via `scale_input`, expanded with Fourier features, passed through
tanh hidden layers, and the output norm is capped at `ωmax` via `scale_norm`.
"""
function get_default_NN(params::AP, rng, θ_trained) where {AP<:AbstractParameters}
    n_fourier = 4
    U = Lux.Chain(
        Lux.WrappedFunction(x -> scale_input(x; xmin = params.tmin, xmax = params.tmax)),
        Lux.WrappedFunction(x -> fourier_feature(x; n = n_fourier)),
        Lux.Dense(2 * n_fourier, 10, tanh),
        Lux.Dense(10, 10, tanh),
        Lux.Dense(10, 10, tanh),
        Lux.Dense(10, 3, tanh),
        Lux.WrappedFunction(x -> scale_norm(params.ωmax .* x; scale = params.ωmax)),
    )
    return U
end

### Custom Activation Funtions

"""
    sigmoid_cap(x; ω₀=1.0)

Normalization of the neural network last layer
"""
rbf(x) = exp.(-(x .^ 2))

sigmoid_cap(x; ω₀ = 1.0) = sigmoid_cap(x, ω₀)

function sigmoid_cap(x, ω₀)
    min_value = -ω₀
    max_value = +ω₀
    return min_value + (max_value - min_value) * sigmoid(x)
end

# For some reason, when I import the Lux.sigmoid function this train badly, 
# increasing the value of the loss function over iterations...
function sigmoid(x)
    return 1.0 / (1.0 + exp(-x))
end

"""
    relu_cap(x; ω₀=1.0)
"""
relu_cap(x; ω₀ = 1.0) = relu_cap(x, ω₀)

function relu_cap(x, ω₀)
    min_value = -ω₀
    max_value = +ω₀
    return relu_cap(x, min_value, max_value)
end

function relu_cap(x, min_value::Float64, max_value::Float64)
    return min_value + (max_value - min_value) * max(0.0, min(x, 1.0))
end

# scale input 

function scale_input(x; xmin, xmax)
    return 2.0 .* (x .- xmin) ./ (xmax .- xmin) .- 1.0
end

# Scale norm

function scale_norm(x; scale = 1.0)
    @assert length(x) == 3
    return (scale * tanh(norm(x) / scale)) .* (x ./ norm(x))
end

# Extension to work with batched jacobian — vectorized over columns
function scale_norm(X::Matrix; scale = 1.0)
    @assert size(X, 1) == 3
    norms = vec(sqrt.(sum(abs2, X; dims = 1)))
    return (scale .* tanh.(norms' ./ scale)) .* (X ./ norms')
end

# Fourier features

function fourier_feature(v; n = 10)
    a₁ = ones(n)
    b₁ = ones(n)
    W = 1.0:1.0:n |> collect
    return [a₁ .* sin.(π .* W .* v); b₁ .* cos.(π .* W .* v)]
end

# Extension to work with batched_jacobian — fully vectorized over columns
function fourier_feature(X::Matrix; n = 10)
    W = collect(1.0:Float64(n))
    return [sin.(π .* W .* X); cos.(π .* W .* X)]
end

### Complex Expansion Activation Functions

"""
    relu(x::Complex)

Extension of ReLU function to complex numbers based on the complex cardioid introduced in 
Virtue et al. (2017), "Better than Real: Complex-valued Neural Nets for MRI Fingerprinting".
This function is equivalent to relu when x is real (and hence angle(x)=0 or angle(x)=π).
"""
function relu(z::Complex)
    return 0.5 * (1 + cos(angle(z))) * z
end

function relu_cap(z::Complex; ω₀ = 1.0)
    min_value = -ω₀
    max_value = +ω₀
    return relu_cap(z, min_value, max_value)
    # return min_value + (max_value - min_value) * relu(z - relu(z-1))
end

"""
    relu_cap(z::Complex, min_value::Float64, max_value::Float64)
"""
function relu_cap(z::Complex, min_value::Float64, max_value::Float64)
    return min_value + (max_value - min_value) * relu(z - relu(z - 1))
end

"""
    sigmoid(z::Complex)
"""
function sigmoid(z::Complex)
    return 1.0 / (1.0 + exp(-z))
end

"""
    gelu(x::Complex)

Extension of the GELU activation function for complex variables.
We use the approximation using tanh() to avoid dealing with the complex error function
"""
function gelu(z::Complex)
    # We use the Gelu approximation to avoid complex holomorphic error function
    return 0.5 * z * (1 + tanh((sqrt(2 / π)) * (z + 0.044715 * z^3)))
end
