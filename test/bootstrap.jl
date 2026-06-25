"""
    _uq_test_setup(seed)

Small synthetic dataset (with `kappas`, required for resampling) plus a
quickly-trained baseline `Results`, reused by the uncertainty quantification
tests below.
"""
function _uq_test_setup(seed::Int)
    rng = Random.default_rng()
    Random.seed!(rng, seed)

    n = 15
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
        niter_ADAM = 10,
        niter_LBFGS = 0,
        verbose = false,
    )
    regressor = SplineRegressor(tmin = 0.0, tmax = 10.0, n_basis = 6, degree = 3, ωmax = 0.1)

    results = train(data, params, rng, nothing, regressor)
    return data, params, regressor, results
end

"""
    test_sample_uq()

Checks the shape of `sample_uq`'s output, that resampled datasets are
pairwise distinct, and that `parallel = true` gives bit-identical results to
`parallel = false` for the same `rng` — including when each sample itself
runs a multistart search (`n_runs > 1`).
"""
function test_sample_uq()
    data, params, regressor, results = _uq_test_setup(101)

    ens = sample_uq(
        data, params, Random.MersenneTwister(11), results.θ, regressor;
        n_samples = 5, n_runs = 1,
    )

    @test ens isa EnsambleResult
    @test length(ens.results) == 5
    @test length(ens.datasets) == 5
    @test all(d -> size(d.directions) == size(data.directions), ens.datasets)
    @test allunique(d.directions for d in ens.datasets)

    ens_seq = sample_uq(
        data, params, Random.MersenneTwister(7), results.θ, regressor;
        n_samples = 4, n_runs = 2, parallel = false,
    )
    ens_par = sample_uq(
        data, params, Random.MersenneTwister(7), results.θ, regressor;
        n_samples = 4, n_runs = 2, parallel = true,
    )

    for i = 1:4
        @test ens_seq.datasets[i].directions == ens_par.datasets[i].directions
        @test ens_seq.results[i].fit_directions == ens_par.results[i].fit_directions
    end
end

"""
    test_train_multistart_parallel()

Checks that `train`'s multistart search (`n_runs > 1`) gives bit-identical
results whether `parallel` is `false` or `true`, for the same `rng`.
"""
function test_train_multistart_parallel()
    data, params, regressor, _ = _uq_test_setup(102)

    r_seq = train(data, params, Random.MersenneTwister(5), nothing, regressor; n_runs = 4, parallel = false)
    r_par = train(data, params, Random.MersenneTwister(5), nothing, regressor; n_runs = 4, parallel = true)

    @test r_seq.fit_directions == r_par.fit_directions
    @test sum(values(r_seq.losses_dict)) == sum(values(r_par.losses_dict))
end

"""
    test_plot_sphere_ensemble()

Smoke test for the `plot_sphere(data, ::EnsambleResult, ...)` method: checks
it runs without erroring, with and without the resampled-points cloud.
"""
function test_plot_sphere_ensemble()
    data, params, regressor, results = _uq_test_setup(103)
    ens = sample_uq(data, params, Random.MersenneTwister(13), results.θ, regressor; n_samples = 3, n_runs = 1)

    p1 = plot_sphere(data, ens; main_result = results)
    @test p1 isa SphereUDE.Plots.Plot

    p2 = plot_sphere(data, ens; main_result = results, show_resampled_points = false)
    @test p2 isa SphereUDE.Plots.Plot
end
