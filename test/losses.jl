using LinearAlgebra

"""
    test_loss_empirical_kappa_weighting()

Regression test for a broadcasting bug in `loss_empirical`: `data.kappas` (a
length-N `Vector`) was being multiplied against the per-point residual while
it still had shape `(1,N)` (from `sum(..., dims = 1)`), so the broadcast
silently computed an N×N outer product (`sum(kappas) * sum(residuals)`)
instead of pairing each `kappa_i` with its own point's residual
(`Σᵢ kappaᵢ·residualᵢ`).

Uses a `SplineRegressor` with tiny (not exactly zero — `scale_norm` divides
by the raw output's norm, so an exact zero is its own separate edge case)
control points, so the angular velocity `L(t) ≈ 0` and the predicted
direction stays at `u0` (up to ~1e-8 drift) for every time — making the
expected per-point residuals (and hence the expected loss) computable by
hand.
"""
function test_loss_empirical_kappa_weighting()

    u0 = [0.0, 0.0, 1.0]
    # Columns: u0 itself, a point 90° away, and the antipode.
    directions = [0.0 1.0 0.0; 0.0 0.0 0.0; 1.0 0.0 -1.0]
    kappas = [10.0, 1.0, 5.0]
    times = [0.1, 0.5, 0.9]

    data = SphereData(times = times, directions = directions, kappas = kappas, L = nothing)

    tmin, tmax = 0.0, 1.0
    params = SphereParameters(tmin = tmin, tmax = tmax, u0 = u0, ωmax = 0.1, weighted = false)

    regressor = SplineRegressor(tmin = tmin, tmax = tmax, n_basis = 4, degree = 1, ωmax = 0.1)
    θ = SphereUDE.ComponentVector(x = fill(1e-8, 4), y = fill(1e-8, 4), z = fill(1e-8, 4))
    β = SphereUDE.ComponentVector(θ = θ)

    # Tiny control points ⟹ L(t) ≈ 0 ⟹ du/dt = L × u ≈ 0 ⟹ u(t) ≈ u0 (drift ~1e-8).
    residuals = [1 - dot(u0, directions[:, i]) for i = 1:3]
    @test residuals ≈ [0.0, 1.0, 2.0] atol = 1e-8

    weights = 1 / (length(times) * (tmax - tmin))
    expected = sum(weights .* kappas .* residuals)     # correctly paired
    buggy = weights * sum(kappas) * sum(residuals)     # what the outer-product bug computed

    # Sanity check that this example actually discriminates between the two.
    @test !isapprox(expected, buggy; rtol = 0.1)

    got = SphereUDE.loss_empirical(β, data, params, regressor)
    @test isapprox(got, expected; atol = 1e-6)
end
