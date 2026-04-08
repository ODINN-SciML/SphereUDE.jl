
function make_test_regressors(params, rng)
    # List of (regressor, θ₀) pairs to test against the AbstractRegressor interface.
    # Add new regressors here as they are implemented.
    nn = get_default_NN(params, rng, nothing)
    return [NNRegressor(nn, rng)]
end

function test_regressor_interface()

    rng = Random.default_rng()
    Random.seed!(rng, 42)

    params = SphereParameters(
        tmin = 0.0, tmax = 10.0,
        u0 = [0.0, 0.0, 1.0],
        ωmax = 0.1,
        niter_ADAM = 0, niter_LBFGS = 0,
    )

    t_test = 5.0

    for (regressor, θ₀) in make_test_regressors(params, rng)
        @info "Testing regressor interface | type=$(typeof(regressor))"

        # Type hierarchy
        @test regressor isa AbstractRegressor
        @test θ₀ isa AbstractVector

        # Forward call returns a 3-vector
        L = regressor(t_test, θ₀)
        @test length(L) == 3

        # jacobian_params returns a (3 × |θ|) matrix
        J = jacobian_params(regressor, t_test, θ₀)
        @test size(J, 1) == 3
        @test size(J, 2) == length(θ₀)

        # init_params returns a fresh ComponentVector of the same shape
        θ₁ = init_params(regressor, rng)
        @test size(θ₁) == size(θ₀)

    end
end

function test_reg_constructor()

    reg = Regularization(
        order = 1,
        power = 2.0,
        λ = 0.1,
        diff_mode = ComplexStepDifferentiation(),
    )

    @test reg.order == 1
    @test reg.power == 2.0
    @test reg.λ == 0.1
    @test typeof(reg.diff_mode) <: AbstractDifferentiation

end

function test_param_constructor()

    params = SphereParameters(
        tmin = 0.0,
        tmax = 100.0,
        u0 = [0.0, 0.0, 1.0],
        ωmax = 1.0,
        reg = nothing,
        train_initial_condition = false,
        multiple_shooting = true,
        niter_ADAM = 1000,
        niter_LBFGS = 300,
        reltol = 1e6,
        abstol = 1e-6,
    )

    reg1 = Regularization(order = 0, power = 2.0, λ = 0.1, diff_mode = FiniteDiff())
    reg2 = Regularization(order = 1, power = 1.0, λ = 0.1, diff_mode = LuxNestedAD())

    params2 = SphereParameters(
        tmin = 0.0,
        tmax = 100.0,
        u0 = [0.0, 0.0, 1.0],
        ωmax = 1.0,
        reg = [reg1, reg2],
        train_initial_condition = false,
        multiple_shooting = true,
        niter_ADAM = 1000,
        niter_LBFGS = 300,
        reltol = 1e6,
        abstol = 1e-6,
        quadrature = RandomQuadrature(n_nodes = 50)
    )

    @test params.niter_ADAM == 1000
    @test params.tmax == 100.0

    @test params2.reg[1].order == 0
    @test typeof(params2.reg[2].diff_mode) <: LuxNestedAD
    @test params2.quadrature.n_nodes == 50
end
