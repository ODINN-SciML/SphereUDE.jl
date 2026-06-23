export NNRegressor

"""
    NNRegressor{L}

Wraps a Lux neural network and its (non-trainable) state into a single
callable regressor that maps time t → angular velocity L ∈ ℝ³.

    regressor(t, θ) → Vector{3}
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
    init_params(r::NNRegressor, rng) → ComponentVector

Return a fresh set of initial trainable parameters by re-running Lux setup.
"""
function init_params(r::NNRegressor, rng)
    θ, _ = Lux.setup(rng, r.model)
    return ComponentVector{Float64}(θ)
end

"""
Forward evaluation: predict L(t) given parameters θ.
"""
(r::NNRegressor)(t::Real, θ) = StatefulLuxLayer{true}(r.model, θ, r.st)([t])

"""
    adam_optimizer(r::NNRegressor, learning_rate) → Optimisers.Adam
"""
function adam_optimizer(r::NNRegressor, learning_rate::Real)
    return Optimisers.Adam(learning_rate, (0.9, 0.999))
end

"""
    lbfgs_optimizer(r::NNRegressor) → Optim.LBFGS

Uses a small static initial step (alpha=0.01), empirically tuned for this
regressor's parameter/gradient scale.
"""
function lbfgs_optimizer(r::NNRegressor)
    return Optim.LBFGS(;
        alphaguess = LineSearches.InitialStatic(alpha = 0.01),
        linesearch = LineSearches.HagerZhang(),
    )
end
