export AbstractRegressor
export jacobian_params, init_params

"""
    AbstractRegressor

Abstract type for all regressors that map time t to an angular velocity L ∈ ℝ³.

## Interface

Every concrete subtype must implement:

- `(r::MyRegressor)(t::Real, θ)  → Vector{3}`  — forward evaluation
- `init_params(r::MyRegressor, rng) → ComponentVector` — initial parameters

Optionally override for an analytical (faster) Jacobian:

- `jacobian_params(r, t, θ) → Matrix (3 × |θ|)` — defaults to Zygote AD
"""
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
