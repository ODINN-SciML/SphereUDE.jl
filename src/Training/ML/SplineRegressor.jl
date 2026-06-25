export SplineRegressor, get_default_splines

"""
    SplineRegressor

Clamped B-spline regressor that maps time t ∈ [tmin, tmax] to an angular
velocity vector L(t,θ) ∈ ℝ³.

## Parameterisation

The trainable parameters are the B-spline control points, stored as a
ComponentVector with three named fields:

    θ = ComponentVector(x = [c₁ˣ,…,cₙˣ],  y = [c₁ʸ,…,cₙʸ],  z = [c₁ᶻ,…,cₙᶻ])

Each triple (cᵢˣ, cᵢʸ, cᵢᶻ) is the i-th 3D control point.

## Forward map

    x(t) = Σᵢ₌₁ⁿ Bᵢₚ(t) cᵢ  ∈ ℝ³        ("raw" spline output)

    L(t,θ) = scale_norm( x(t) ; scale = ωmax )

where Bᵢₚ(t) are the clamped B-spline basis functions of degree p, and
`scale_norm` is a smooth norm-limiting map (see `_jacobian_scale_norm`).

## Fields

- `knots`   : full clamped knot vector of length n_basis + degree + 1 (fixed)
- `degree`  : polynomial degree p (default 3 = cubic)
- `n_basis` : number of basis functions / control points n
- `ωmax`    : output norm cap — scale parameter for `scale_norm`

## Constraint

n_basis ≥ degree + 1  (equivalently, at least one basis function per degree).
"""
struct SplineRegressor <: AbstractRegressor
    knots  ::Vector{Float64}   # full clamped knot vector (fixed, not trained)
    degree ::Int               # spline degree p (3 = cubic)
    n_basis::Int               # number of control points / basis functions n
    ωmax   ::Float64           # output norm cap (matches SphereParameters.ωmax)
end

"""
    get_default_splines(params, rng) → SplineRegressor

Return a default SplineRegressor for those cases where no regressor has been
provided by the user.  Uses cubic B-splines (degree=3) with 10 control points.
"""
function get_default_splines(params::AP, rng) where {AP<:AbstractParameters}
    return SplineRegressor(
        tmin    = Float64(params.tmin),
        tmax    = Float64(params.tmax),
        n_basis = 10,
        degree  = 3,
        ωmax    = Float64(params.ωmax),
    )
end

"""
    SplineRegressor(; tmin, tmax, n_basis, degree=3, ωmax)

Build a clamped B-spline regressor over [tmin, tmax] with `n_basis` control
points and uniform interior knots.  `n_basis ≥ degree + 1` is required.

The knot vector is constructed by `_make_clamped_knots`.
"""
function SplineRegressor(;
    tmin   ::Float64,
    tmax   ::Float64,
    n_basis::Int,
    degree ::Int     = 3,
    ωmax   ::Float64,
)
    knots = _make_clamped_knots(tmin, tmax, n_basis, degree)
    return SplineRegressor(knots, degree, n_basis, ωmax)
end

# -------------------------------------------------------------------------
# Interface implementation
# -------------------------------------------------------------------------

"""
    (r::SplineRegressor)(t, θ) → Vector{3}

Forward evaluation of the B-spline regressor at time t.

## Mathematics

Let b = [B₁ₚ(t), …, Bₙₚ(t)] ∈ ℝⁿ be the vector of B-spline basis values.
The raw (uncapped) output is the linear combination

    x(t,θ) = [ b⋅θ.x,  b⋅θ.y,  b⋅θ.z ]  ∈ ℝ³

The final output is passed through `scale_norm` to enforce the ωmax cap:

    L(t,θ) = scale_norm( x(t,θ) ; scale = ωmax )

This is identical in structure to the NNRegressor output layer.
"""
function (r::SplineRegressor)(t::Real, θ)
    b     = _eval_basis(r, t)
    L_raw = [dot(b, θ.x), dot(b, θ.y), dot(b, θ.z)]
    return scale_norm(L_raw; scale = r.ωmax)
end

"""
    init_params(r::SplineRegressor, rng) → ComponentVector

Initialise control points as small Gaussian random values:

    cᵢˢ ~ N(0, σ²),   σ = 0.1 · ωmax,   s ∈ {x, y, z}

The factor 0.1 ensures that ‖x(t,θ)‖ ≪ ωmax initially, placing the network
in the approximately-linear regime of `scale_norm` where training starts easily.
"""
function init_params(r::SplineRegressor, rng)
    σ = 0.3 * r.ωmax
    n = r.n_basis
    return ComponentVector{Float64}(
        x = σ .* randn(rng, n),
        y = σ .* randn(rng, n),
        z = σ .* randn(rng, n),
    )
end

"""
    adam_optimizer(r::SplineRegressor, learning_rate) → Optimisers.Adam
"""
function adam_optimizer(r::SplineRegressor, learning_rate::Real)
    return Optimisers.Adam(learning_rate, (0.9, 0.999))
end

"""
    lbfgs_optimizer(r::SplineRegressor) → Optim.LBFGS

Uses a unit static initial step (alpha=1.0), the textbook choice for LBFGS:
its search direction is already curvature-scaled by the inverse-Hessian
approximation, so a unit step is expected to be a good initial guess for the
line search. The control-point gradient scale made the NN's alpha=0.01 a poor
fit here — it forced the line search to backtrack/extrapolate on every
iteration, which is why LBFGS appeared to make no progress.
"""
function lbfgs_optimizer(r::SplineRegressor)
    return Optim.LBFGS(;
        alphaguess = LineSearches.InitialStatic(alpha = 1.0),
        linesearch = LineSearches.HagerZhang(),
    )
end

"""
    jacobian_params(r::SplineRegressor, t, θ) → Matrix{Float64}  (3 × 3n)

Analytical Jacobian of L(t,θ) with respect to the parameter vector θ.

## Mathematics

Write x = x(t,θ) = B_block(t) θ  where B_block(t) ∈ ℝ^{3×3n} is the
block-diagonal matrix

         ┌ b(t)ᵀ   0      0    ┐
B_block =│  0    b(t)ᵀ   0    │   ∈ ℝ^{3×3n}
         └  0      0    b(t)ᵀ ┘

with b(t) = [B₁ₚ(t), …, Bₙₚ(t)] ∈ ℝⁿ.

By the chain rule through scale_norm:

    ∂L/∂θ = J_sn(x) · B_block(t)   ∈ ℝ^{3×3n}

where J_sn(x) = ∂scale_norm(x)/∂x ∈ ℝ^{3×3} is computed by
`_jacobian_scale_norm`.

No automatic differentiation is needed: the B-spline is linear in θ.
"""
function jacobian_params(r::SplineRegressor, t::Real, θ)
    b     = _eval_basis(r, t)
    L_raw = [dot(b, θ.x), dot(b, θ.y), dot(b, θ.z)]
    J_sn  = _jacobian_scale_norm(L_raw; scale = r.ωmax)   # 3×3

    # ∂L_raw/∂θ = B_block(t)  (3×3n, block-diagonal)
    n = r.n_basis
    dLraw_dθ = zeros(3, 3n)
    dLraw_dθ[1,   1: n ] .= b
    dLraw_dθ[2, n+1:2n ] .= b
    dLraw_dθ[3, 2n+1:3n] .= b

    return J_sn * dLraw_dθ
end

# -------------------------------------------------------------------------
# Internal utilities
# -------------------------------------------------------------------------

"""
    _make_clamped_knots(tmin, tmax, n_basis, degree) → Vector{Float64}

Build a clamped (open) B-spline knot vector over [tmin, tmax].

## Knot structure

A B-spline of degree p with n basis functions requires exactly n + p + 1 knots.
For a clamped spline, the first and last knot are repeated (p+1) times so that
the spline interpolates the first and last control points:

    t = [ tmin, …, tmin,   t₁, t₂, …, t_m,   tmax, …, tmax ]
         ╰───(p+1)───╯    ╰──── interior ────╯ ╰───(p+1)───╯

The number of interior knots is  m = n - p - 1  (may be zero).
Interior knots are uniformly spaced in (tmin, tmax).
"""
function _make_clamped_knots(tmin, tmax, n_basis, degree)
    n_interior = n_basis - degree - 1
    n_interior < 0 && throw(ArgumentError(
        "SplineRegressor requires n_basis ≥ degree + 1 = $(degree + 1), got $n_basis"
    ))
    interior = n_interior > 0 ?
        collect(LinRange(tmin, tmax, n_interior + 2))[2:end-1] : Float64[]
    return [fill(tmin, degree + 1); interior; fill(tmax, degree + 1)]
end

"""
    _eval_basis(r::SplineRegressor, t) → Vector{Float64}  (length n_basis)

Evaluate all n basis functions at t:

    b(t) = [ B₁ₚ(t), B₂ₚ(t), …, Bₙₚ(t) ]

Calls `_bspline_basis` for each i = 1, …, n.
Accepts any Real for t (including AD-traced types such as ReverseDiff.TrackedReal).
"""
function _eval_basis(r::SplineRegressor, t::Real)
    return [_bspline_basis(r.knots, r.degree, i, t) for i in 1:r.n_basis]
end

"""
    _bspline_basis(knots, degree, i, t) → Real

Evaluate the i-th (1-indexed) B-spline basis function of degree p at t using
the Cox–de Boor recursion.

## Recursion

Base case (p = 0):

    B_{i,0}(t) = 1   if  t_i ≤ t < t_{i+1}  (or t == t_end for the last interval)
               = 0   otherwise

Recursive step (p ≥ 1):

    B_{i,p}(t) = (t - t_i)/(t_{i+p} - t_i) · B_{i,p-1}(t)
               + (t_{i+p+1} - t)/(t_{i+p+1} - t_{i+1}) · B_{i+1,p-1}(t)

with the convention  0/0 = 0  (implemented by checking equal knots before dividing).

## Notes

- Accepts any `Real` for t so that AD tracers (ReverseDiff, ForwardDiff) pass through.
- Uses 1-based indexing: knot vector is accessed as knots[i], knots[i+1], …, knots[i+p+1].
- For a knot vector of length n + p + 1, valid basis indices are i = 1, …, n.
"""
function _bspline_basis(knots::Vector{Float64}, degree::Int, i::Int, t::Real)
    if degree == 0
        at_end = (t == knots[end]) && (knots[i] ≤ t ≤ knots[i+1])
        return Float64((knots[i] ≤ t < knots[i+1]) || at_end)
    end
    c1 = knots[i+degree] == knots[i] ? 0.0 :
         (t - knots[i]) / (knots[i+degree] - knots[i]) *
         _bspline_basis(knots, degree-1, i, t)
    c2 = knots[i+degree+1] == knots[i+1] ? 0.0 :
         (knots[i+degree+1] - t) / (knots[i+degree+1] - knots[i+1]) *
         _bspline_basis(knots, degree-1, i+1, t)
    return c1 + c2
end

"""
    time_derivative(r::SplineRegressor, t, θ) → Vector{3}

Exact analytical time derivative dL/dt at time t.

## Mathematics

Differentiating L(t,θ) = scale_norm(x(t)) with x(t) = b(t)⋅θ:

    dL/dt = J_sn(x(t)) · ẋ(t)

where:
- J_sn(x) = ∂scale_norm(x)/∂x  ∈ ℝ^{3×3}  (see `_jacobian_scale_norm`)
- ẋ(t) = ṡ(t) where  ṡˢ(t) = b̊(t)⋅θ.s,   s ∈ {x,y,z}
- b̊(t) = [Ḃ₁ₚ(t), …, Ḃₙₚ(t)]  is the vector of basis function derivatives

No finite differences are used: both b̊(t) and J_sn(x) are computed analytically.
"""
function time_derivative(r::SplineRegressor, t::Real, θ)
    b      = _eval_basis(r, t)
    b_dot  = _eval_basis_derivative(r, t)
    L_raw  = [dot(b, θ.x), dot(b, θ.y), dot(b, θ.z)]
    dLraw  = [dot(b_dot, θ.x), dot(b_dot, θ.y), dot(b_dot, θ.z)]
    return _jacobian_scale_norm(L_raw; scale = r.ωmax) * dLraw
end

"""
    _eval_basis_derivative(r::SplineRegressor, t) → Vector{Float64}  (length n_basis)

Evaluate the time derivatives of all n basis functions at t:

    b̊(t) = [ Ḃ₁ₚ(t), Ḃ₂ₚ(t), …, Ḃₙₚ(t) ]

Calls `_bspline_basis_derivative` for each i = 1, …, n.
"""
function _eval_basis_derivative(r::SplineRegressor, t::Real)
    return [_bspline_basis_derivative(r.knots, r.degree, i, t) for i in 1:r.n_basis]
end

"""
    _bspline_basis_derivative(knots, degree, i, t) → Real

Analytical derivative of the i-th B-spline basis function of degree p using the
standard differentiation recurrence.

## Formula

    Ḃ_{i,p}(t) = p · B_{i,p-1}(t) / (t_{i+p} - t_i)
               - p · B_{i+1,p-1}(t) / (t_{i+p+1} - t_{i+1})

with the convention  0/0 = 0  (same equal-knot guard as in `_bspline_basis`).

## Derivation

This follows directly from differentiating the Cox–de Boor recursion.
Writing the two terms of B_{i,p} as u·v and applying the product rule, then
using the fact that the derivatives of the linear weight functions are
constant:  d/dt [(t - tᵢ)/(tᵢ₊ₚ - tᵢ)] = 1/(tᵢ₊ₚ - tᵢ), and noting that
the lower-order basis derivatives cancel telescopically, one obtains the formula
above.
"""
function _bspline_basis_derivative(knots::Vector{Float64}, degree::Int, i::Int, t::Real)
    c1 = knots[i+degree] == knots[i] ? zero(t) :
         degree * _bspline_basis(knots, degree-1, i, t) / (knots[i+degree] - knots[i])
    c2 = knots[i+degree+1] == knots[i+1] ? zero(t) :
         degree * _bspline_basis(knots, degree-1, i+1, t) / (knots[i+degree+1] - knots[i+1])
    return c1 - c2
end

"""
    _vjp_L(r, t, θ, η) → ComponentVector

Vector–Jacobian product  (∂L(t,θ)/∂θ)ᵀ η,  where L = scale_norm(x),  x = B_block θ.

## Mathematics

From `jacobian_params`:

    ∂L/∂θ = J_sn(x) · B_block(t)   ∈ ℝ^{3×3n}

The VJP with a cotangent η ∈ ℝ³ is:

    (∂L/∂θ)ᵀ η = B_block(t)ᵀ · J_sn(x)ᵀ η = B_block(t)ᵀ ξ

where ξ = J_sn(x)ᵀ η ∈ ℝ³.  Because J_sn is symmetric (see `_jacobian_scale_norm`),
ξ = J_sn(x) η.

The action of B_block(t)ᵀ on ξ decomposes by component:

    (∂L/∂θ.s)ᵀ η = b(t) · ξ[s],   s ∈ {x, y, z}

Result has the same ComponentVector structure as θ.
"""
function _vjp_L(r::SplineRegressor, t::Real, θ, η)
    b   = _eval_basis(r, t)
    x   = [dot(b, θ.x), dot(b, θ.y), dot(b, θ.z)]
    ξ   = _jacobian_scale_norm(x; scale = r.ωmax)' * η      # J_snᵀ η  (3-vector)
    return ComponentVector(x = b .* ξ[1], y = b .* ξ[2], z = b .* ξ[3])
end

"""
    _vjp_dLdt(r, t, θ, η) → ComponentVector

Vector–Jacobian product  (∂(dL/dt)(t,θ)/∂θ)ᵀ η.

## Setting up the derivative

Let:
    x  = x(t,θ)  = B_block  θ  ∈ ℝ³     (raw spline value)
    y  = ẋ(t,θ)  = Ḃ_block  θ  ∈ ℝ³     (raw time derivative)
    r  = ‖x‖

The time derivative is:

    v(t,θ) = dL/dt = J_sn(x) · y

where J_sn(x) = ∂scale_norm(x)/∂x depends on θ through x.

## Product-rule differentiation

By the product rule:

    ∂v/∂θ = (∂J_sn/∂θ) y  +  J_sn · Ḃ_block

The second term is easy:

    (J_sn · Ḃ_block)ᵀ η = Ḃ_blockᵀ (J_snᵀ η) = Ḃ_blockᵀ ξ,    ξ = J_snᵀ η

For the first term we need ∂J_sn/∂x · (B_block δθ).  Writing J_sn explicitly:

    J_sn = (f/r) I  +  β · x xᵀ,    β = (f′ - f/r) / r²

Differentiating J_sn in direction δx and contracting with y and η:

    ηᵀ (∂J_sn/∂x δx) y = ηᵀ [ ∂β/∂x · (y xᵀ + x yᵀ) δx
                               + β (δx yᵀ + δx ← wait, use d(f/r)) ]

Working this out fully (see derivation notes):

    ηᵀ (∂J_sn(x)/∂x δx) y
        = β (y⋅η)(xᵀ δx) + β (x⋅η)(yᵀ δx) + (β c)(ηᵀ δx)
          + γ c (xᵀ δx)(x⋅η)

where:
    c  = x⋅y
    β  = (f′ - f/r) / r²
    γ  = ∂β/∂r = (f″ r² - 3f′ r + 3f) / r⁵

Collecting the scalar weights on B_blockᵀ acting on x, y, η respectively:

    ηᵀ (∂J_sn/∂θ y) = B_blockᵀ [ (β(y⋅η) + cγ(x⋅η)) x
                                 + β(x⋅η) y
                                 + βc η ]

So the full VJP is:

    (∂v/∂θ)ᵀ η = Ḃ_blockᵀ ξ  +  B_blockᵀ η_comb

where:
    ξ       = J_snᵀ η
    p1      = β(y⋅η) + cγ(x⋅η)
    p2      = β(x⋅η)
    η_comb  = βc η + p1 x + p2 y

The action of B_blockᵀ decomposes identically to `_vjp_L`:

    (Ḃ_blockᵀ ξ).s = b̊(t) ξ[s],       s ∈ {x,y,z}
    (B_blockᵀ η_comb).s = b(t) η_comb[s]

## Near-zero branch

When r = ‖x‖ < 1e-10, J_sn ≈ I (identity), so β = γ = 0 and only the
Ḃ_block term survives:

    (∂v/∂θ)ᵀ η ≈ Ḃ_blockᵀ η

## Scalar quantities derived from scale_norm

    f(r)   = ωmax · tanh(r / ωmax)
    f′(r)  = sech²(r / ωmax)
    f″(r)  = -2 sech²(r/ωmax) tanh(r/ωmax) / ωmax  = -2f′(r) tanh(r/ωmax) / ωmax
"""
function _vjp_dLdt(r::SplineRegressor, t::Real, θ, η)
    b     = _eval_basis(r, t)
    b_dot = _eval_basis_derivative(r, t)
    x     = [dot(b,     θ.x), dot(b,     θ.y), dot(b,     θ.z)]   # L_raw
    y     = [dot(b_dot, θ.x), dot(b_dot, θ.y), dot(b_dot, θ.z)]   # dL_raw/dt

    J_sn  = _jacobian_scale_norm(x; scale = r.ωmax)
    ξ     = J_sn' * η       # 3-vector

    rv = norm(x)
    if rv < 1e-10
        # β = γ = 0: only the Ḃ_block term survives
        return ComponentVector(x = b_dot .* ξ[1], y = b_dot .* ξ[2], z = b_dot .* ξ[3])
    end

    sc  = r.ωmax
    f   = sc * tanh(rv / sc)
    fp  = sech(rv / sc)^2
    fpp = -2fp * tanh(rv / sc) / sc

    β  = (fp - f / rv) / rv^2
    γ  = (fpp * rv^2 - 3fp * rv + 3f) / rv^5

    c   = dot(x, y)
    p1  = β * dot(y, η) + c * γ * dot(x, η)   # weight on B_blockᵀ x
    p2  = β * dot(x, η)                        # weight on B_blockᵀ y
    βc  = β * c                                 # weight on B_blockᵀ η

    η_comb = βc .* η .+ p1 .* x .+ p2 .* y

    return ComponentVector(
        x = b_dot .* ξ[1] .+ b .* η_comb[1],
        y = b_dot .* ξ[2] .+ b .* η_comb[2],
        z = b_dot .* ξ[3] .+ b .* η_comb[3],
    )
end

"""
    _jacobian_scale_norm(x; scale) → Matrix{Float64}  (3×3)

Analytical Jacobian  J_sn = ∂scale_norm(x)/∂x  ∈ ℝ^{3×3}.

## The scale_norm map

    scale_norm(x; scale) = f(r) · x/r,    r = ‖x‖,    f(r) = scale · tanh(r/scale)

This is a smooth, norm-limiting map: ‖scale_norm(x)‖ = f(r) → scale as r → ∞,
and scale_norm(x) ≈ x for ‖x‖ ≪ scale.

## Jacobian derivation

Differentiating scale_norm(x) = f(r)/r · x  (with r = ‖x‖):

    ∂(scale_norm)/∂xₖ = f(r)/r · eₖ  +  x · ∂(f(r)/r)/∂xₖ

Using ∂r/∂xₖ = xₖ/r:

    ∂(f(r)/r)/∂xₖ = (f′(r)/r - f(r)/r²) · xₖ/r = β · xₖ

where  β = (f′ - f/r) / r².

Collecting:

    J_sn = (f/r) I  +  β · x xᵀ

## Symmetry

J_sn is symmetric because it is a scalar multiple of I plus a rank-1 symmetric
outer product.  Thus J_sn = J_snᵀ.

## Near-zero branch

When r < 1e-10,  scale_norm(x) ≈ x  (linear regime), so J_sn ≈ I.

## Scalar quantities

    f(r)  = scale · tanh(r/scale)
    f′(r) = sech²(r/scale)         (derivative of f w.r.t. r)
    β     = (f′ - f/r) / r²
"""
function _jacobian_scale_norm(x::AbstractVector; scale::Real = 1.0)
    r = norm(x)
    if r < 1e-10                                       # near-zero: scale_norm ≈ identity
        return Matrix{Float64}(I, 3, 3)
    end
    f  = scale * tanh(r / scale)                       # ‖scale_norm(x)‖
    fp = sech(r / scale)^2                             # df/dr
    return (f / r) .* Matrix{Float64}(I, 3, 3) .+ ((fp - f/r) / r^2) .* (x * x')
end
