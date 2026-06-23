export AbstractRegressor
export jacobian_params, init_params, adam_optimizer, lbfgs_optimizer

"""
    AbstractRegressor

Abstract type for all regressors that map time t to an angular velocity L ∈ ℝ³.

## Interface

Every concrete subtype must implement:

- `(r::MyRegressor)(t::Real, θ)  → Vector{3}`  — forward evaluation
- `init_params(r::MyRegressor, rng) → ComponentVector` — initial parameters
- `adam_optimizer(r::MyRegressor, learning_rate) → Optimisers.Adam` — ADAM phase
- `lbfgs_optimizer(r::MyRegressor) → Optim.LBFGS` — LBFGS phase

`adam_optimizer`/`lbfgs_optimizer` have no generic fallback: the right LBFGS
initial step scale and ADAM momentum depend on how the regressor's
parameterization shapes the loss landscape, so each regressor must state its
own choice explicitly (see `NNRegressor.jl` / `SplineRegressor.jl`).

Optionally override for an analytical (faster) Jacobian:

- `jacobian_params(r, t, θ) → Matrix (3 × |θ|)` — defaults to Zygote AD
"""
abstract type AbstractRegressor end

"""
    adam_optimizer(r::AbstractRegressor, learning_rate) → Optimisers.Adam

ADAM optimizer used for the main training phase. No generic implementation —
must be defined for each concrete regressor type.
"""
function adam_optimizer end

"""
    lbfgs_optimizer(r::AbstractRegressor) → Optim.LBFGS

LBFGS optimizer (initial step guess + line search) used for the LBFGS training
phase. No generic implementation — must be defined for each concrete
regressor type.
"""
function lbfgs_optimizer end

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
