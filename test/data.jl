"""
    test_resample_data()

Checks that `resample_data` draws from the noise model it claims to: for a
batch of directions each with their own concentration `kappa`, resample the
dataset `n_datasets` times and verify that, for each direction,
1) the average of the resampled directions points back at the original
   direction, and
2) the mean resultant length of the resampled directions matches the
   analytic value `A(κ) = coth(κ) - 1/κ` of the von Mises–Fisher
   distribution on the sphere (the κ ↔ dispersion relation classically used
   in paleomagnetism, e.g. Fisher 1953).
"""
function test_resample_data()

    rng = Random.default_rng()
    Random.seed!(rng, 42)

    mus = [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1 / √2, 1 / √2],
    ]
    test_kappas = [5.0, 50.0, 300.0]

    directions = reduce(hcat, [mu for mu in mus, _ in test_kappas])
    kappas = vec([k for _ in mus, k in test_kappas])
    times = collect(1.0:length(kappas))

    data = SphereData(times = times, directions = directions, kappas = kappas, L = nothing)

    n_datasets = 300
    mean_directions = zeros(size(directions))
    for _ = 1:n_datasets
        resampled = resample_data(data, rng)
        mean_directions .+= resampled.directions
    end
    mean_directions ./= n_datasets

    for i in axes(directions, 2)
        R̄ = norm(mean_directions[:, i])
        mean_dir = mean_directions[:, i] / R̄
        κ = kappas[i]
        A = coth(κ) - 1 / κ

        # 1) Resampled directions average back to the original direction
        @test isapprox(mean_dir, directions[:, i], atol = 0.1)
        # 2) Mean resultant length matches the von Mises-Fisher κ ↔ R̄ relation
        @test isapprox(R̄, A, atol = 0.05)
    end
end
