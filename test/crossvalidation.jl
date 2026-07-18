"""
    test_cv_splits()

Checks `SphereUDE._cv_splits`'s k-fold partition for several `k` and dataset
sizes (including sizes not divisible by `k`):
1) fold sizes are balanced — `ceil(N / k)` for the first `N % k` folds and
   `floor(N / k)` for the rest, so no two folds differ by more than one
   observation,
2) every observation appears in exactly one validation fold across the `k`
   splits (the validation folds partition the data), and
3) each fold's train and validation sets are disjoint and together cover
   all `N` observations.
"""
function test_cv_splits()
    for N in (13, 20, 21)
        data = SphereData(
            times = collect(1.0:N),
            directions = randn(3, N),
            kappas = fill(20.0, N),
            L = nothing,
        )

        for k = 2:5
            splits = SphereUDE._cv_splits(data, Random.MersenneTwister(k); k = k)
            @test length(splits) == k

            val_sizes = [length(data_val.times) for (_, data_val) in splits]
            base, remainder = divrem(N, k)
            expected_sizes = sort([i <= remainder ? base + 1 : base for i = 1:k]; rev = true)
            @test sort(val_sizes; rev = true) == expected_sizes
            @test maximum(val_sizes) - minimum(val_sizes) <= 1

            for (data_train, data_val) in splits
                @test length(data_train.times) + length(data_val.times) == N
                @test isempty(intersect(data_train.times, data_val.times))
            end

            all_val_times = sort(vcat([data_val.times for (_, data_val) in splits]...))
            @test all_val_times == sort(data.times)
        end
    end
end

"""
    _cv_test_setup(seed)

Small synthetic dataset plus `SphereParameters` with a single
`RegularizationCV` term, reused by the `train_cv` tests below.
"""
function _cv_test_setup(seed::Int; λ = [0.1, 1.0, 10.0])
    rng = Random.default_rng()
    Random.seed!(rng, seed)

    n = 16
    times = collect(range(0.0, 10.0, length = n))
    mu = [0.0, 0.0, 1.0]
    kappas = fill(50.0, n)
    directions = zeros(3, n)
    for i = 1:n
        directions[:, i] = rand(rng, SphereUDE.sampler(SphereUDE.VonMisesFisher(mu, kappas[i])))
    end
    data = SphereData(times = times, directions = directions, kappas = kappas, L = nothing)

    params = SphereParameters(
        tmin = 0.0,
        tmax = 10.0,
        u0 = [0.0, 0.0, 1.0],
        ωmax = 0.1,
        niter_ADAM = 8,
        niter_LBFGS = 0,
        verbose = false,
        reg = [RegularizationCV(order = 0, power = 2.0, λ = λ)],
    )
    regressor = SplineRegressor(tmin = 0.0, tmax = 10.0, n_basis = 6, degree = 3, ωmax = 0.1)

    return data, params, regressor
end

"""
    test_train_cv()

Checks the shape of `train_cv`'s output (one score per fold for each `λ`
candidate, `best_λ` drawn from the candidate grid, `best_results` filled in
only when `refit = true`), that it rejects `params.reg` with zero or more
than one `RegularizationCV` term, and that `parallel = true` gives
bit-identical scores to `parallel = false` for the same `rng`.
"""
function test_train_cv()
    data, params, regressor = _cv_test_setup(201)
    k_folds = 4
    n_candidates = length(params.reg[1].λ)

    cv = train_cv(data, params, Random.MersenneTwister(11), nothing, regressor; k_folds = k_folds, refit = false)

    @test cv isa CVResult
    @test cv.λ_grid == params.reg[1].λ
    @test length(cv.scores) == n_candidates
    @test all(s -> length(s) == k_folds, cv.scores)
    @test cv.best_λ in cv.λ_grid
    @test isnothing(cv.best_results)

    cv_refit = train_cv(data, params, Random.MersenneTwister(11), nothing, regressor; k_folds = k_folds, refit = true)
    @test !isnothing(cv_refit.best_results)
    @test cv_refit.best_results isa Results

    # No RegularizationCV term at all
    _, params_none, _ = _cv_test_setup(202)
    params_none = update_params(params_none; reg = [Regularization(order = 0, power = 2.0, λ = 1.0)])
    @test_throws AssertionError train_cv(data, params_none, Random.MersenneTwister(1), nothing, regressor; k_folds = k_folds)

    # Two RegularizationCV terms at once — unsupported for now
    params_two = update_params(params,
        reg = [RegularizationCV(order = 0, power = 2.0, λ = [1.0, 2.0]), RegularizationCV(order = 1, power = 2.0, λ = [1.0, 2.0])],
    )
    @test_throws AssertionError train_cv(data, params_two, Random.MersenneTwister(1), nothing, regressor; k_folds = k_folds)

    # Sequential and parallel must give identical scores and best_λ for the same rng
    cv_seq = train_cv(data, params, Random.MersenneTwister(9), nothing, regressor; k_folds = k_folds, parallel = false)
    cv_par = train_cv(data, params, Random.MersenneTwister(9), nothing, regressor; k_folds = k_folds, parallel = true)

    @test cv_seq.scores == cv_par.scores
    @test cv_seq.best_λ == cv_par.best_λ
end
