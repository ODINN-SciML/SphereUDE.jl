export AbstractRegressor, NNRegressor
export jacobian_params, init_params

abstract type AbstractRegressor end

"""
    jacobian_params(r, t, θ) → Matrix (3 × |θ|)

Jacobian of the regressor output w.r.t. trainable parameters θ at time t.
Used by the custom SphereBackSolveAdjoint to accumulate parameter gradients.

The default implementation uses Zygote AD and works for any regressor.
Override this method to provide an analytical Jacobian (e.g. for splines).
"""
function jacobian_params(r::AbstractRegressor, t::Real, θ)
    ∇θ, = Zygote.jacobian(_θ -> r(t, _θ), θ)
    return ∇θ
end

# -------------------------------------------------------------------------
# NNRegressor — wraps a Lux neural network
# -------------------------------------------------------------------------

"""
    NNRegressor{L}

Wraps a Lux model and its (non-trainable) state into a single callable object.

    regressor(t, θ) → Vector{3}   # predict angular velocity L at time t
"""
struct NNRegressor{L<:Lux.AbstractLuxLayer} <: AbstractRegressor
    model::L
    st::NamedTuple   # non-trainable Lux state, frozen after setup
end

"""
    NNRegressor(model, rng) → (regressor, θ₀)

Initialise a NNRegressor from a Lux model and an RNG.
Returns the regressor and the initial ComponentVector of trainable parameters.
"""
function NNRegressor(model::L, rng) where {L<:Lux.AbstractLuxLayer}
    θ, st = Lux.setup(rng, model)
    return NNRegressor{L}(model, st), ComponentVector{Float64}(θ)
end

"""
    init_params(r, rng) → ComponentVector

Return a fresh set of initial trainable parameters for the regressor.
Every concrete regressor subtype must implement this.
"""
function init_params(r::NNRegressor, rng)
    θ, _ = Lux.setup(rng, r.model)
    return ComponentVector{Float64}(θ)
end

"""
Forward evaluation: predict L(t) given parameters θ.
"""
(r::NNRegressor)(t::Real, θ) = StatefulLuxLayer{true}(r.model, θ, r.st)([t])
