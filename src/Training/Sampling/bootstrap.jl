export sample_uq

"""
    sample_uq(data, params, rng, θ_trained, regressor; n_samples, n_runs, parallel)

Uncertainty quantification wrapper around [`train`](@ref). Generates
`n_samples` resampled realizations of `data` via [`resample_data`](@ref) and
calls `train` on each one with `params`, `θ_trained` and `regressor`. Returns
an [`EnsambleResult`](@ref) wrapping one `Results` and the corresponding
resampled dataset per sample.

`rng` seeds an independent random stream per sample, so results are the same
regardless of `parallel`. When `parallel = true`, samples are run across
`Threads.nthreads()` threads via `Threads.@threads`. The inner [`train`](@ref)
calls always run with its own `parallel = false`, regardless of `n_runs`, so
that multistart never adds a second, nested layer of threading on top of the
one used here.
"""
function sample_uq(
    data::AD,
    params::AP,
    rng,
    θ_trained,
    regressor::Union{AbstractRegressor,Nothing};
    n_samples::Int = 1,
    n_runs::Int = 1,
    parallel::Bool = false,
) where {AD<:AbstractData,AP<:AbstractParameters}

    results = Vector{Results}(undef, n_samples)
    datasets = Vector{AD}(undef, n_samples)
    seeds = rand(rng, UInt64, n_samples)

    _sample_one = i -> begin
        sample_rng = Random.Xoshiro(seeds[i])
        data_sample = resample_data(data, sample_rng)
        datasets[i] = data_sample
        # Multistart inside train() must stay sequential: sample_uq already
        # parallelizes across samples, and nesting Threads.@threads would
        # oversubscribe the available threads.
        results[i] =
            train(data_sample, params, sample_rng, θ_trained, regressor; n_runs = n_runs, parallel = false)
        params.verbose && @info "[sample_uq] UQ sample $(i)/$(n_samples) done."
    end

    if parallel
        if Threads.nthreads() == 1
            @warn "[sample_uq] parallel=true but Threads.nthreads() == 1 — running sequentially. Start Julia with `--threads=auto` (or set JULIA_NUM_THREADS) to actually parallelize."
        else
            @info "[sample_uq] Running $(n_samples) samples across $(Threads.nthreads()) threads."
        end
        Threads.@threads for i = 1:n_samples
            _sample_one(i)
        end
    else
        for i = 1:n_samples
            _sample_one(i)
        end
    end

    @assert allunique(d.directions for d in datasets) "[SphereUDE] Resampled datasets are not independent: identical realizations detected (check your rng)."

    return EnsambleResult(results = results, datasets = datasets)
end
