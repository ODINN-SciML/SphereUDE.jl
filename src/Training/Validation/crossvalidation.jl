export train_cv

"""
    train_cv(data, params, rng, θ_trained, regressor; n_runs, parallel, refit, k_folds)

Cross-validated wrapper around [`train`](@ref). `params.reg` must contain
exactly one [`RegularizationCV`](@ref) term (any other plain `Regularization`
or `CubicSplinesRegularization` terms in `params.reg` are left untouched).
For each candidate `λ` in that term's grid, trains on `k_folds` train/validation
splits of `data` (see [`_cv_splits`](@ref)) and scores each fold by the
empirical (data-fit only) loss on its held-out validation set. The `λ` with
the lowest average validation loss is selected.

Every `(λ, fold)` combination is an independent fit. When `parallel = true`,
they're distributed across `Threads.nthreads()` threads via `Threads.@threads`
— each task seeded from its own independent RNG stream, so results are the
same regardless of `parallel`. The inner `train` calls for these fits always
run with their own `parallel = false`, so this layer never nests with
`train`'s multistart threading.

When `refit = true` (the default), retrains once more on the full dataset
with the best `λ` (using `parallel` for that single call's own multistart, if
`n_runs > 1`) and stores that as `best_results` in the returned
[`CVResult`](@ref). Set `refit = false` to skip this and only get the
per-candidate scores (`best_results` is then `nothing`).
"""
function train_cv(
    data::AD,
    params::AP,
    rng,
    θ_trained,
    regressor::Union{AbstractRegressor,Nothing};
    n_runs::Int = 1,
    parallel::Bool = false,
    refit::Bool = false,
    refit_all::Bool = false,
    k_folds::Int = 5,
) where {AD<:AbstractData,AP<:AbstractParameters}

    @assert !isnothing(params.reg) "[SphereUDE] train_cv requires params.reg to contain a RegularizationCV term."
    cv_idxs = findall(r -> r isa RegularizationCV, params.reg)
    @assert length(cv_idxs) == 1 "[SphereUDE] train_cv currently supports exactly one RegularizationCV term in params.reg, found $(length(cv_idxs))."
    cv_idx = only(cv_idxs)
    cv_reg = params.reg[cv_idx]

    # CV split between training and validation set
    splits = _cv_splits(data, rng; k = k_folds)
    n_folds = length(splits)
    n_candidates = length(cv_reg.λ)

    all_scores = [Vector{Float64}(undef, n_folds) for _ = 1:n_candidates]

    # Flatten the (λ, fold) grid so every fit is an independent task with its
    # own seeded RNG stream, runnable in any order/on any thread.
    tasks = [(k, s) for k = 1:n_candidates for s = 1:n_folds]
    seeds = rand(rng, UInt64, length(tasks))

    # Single combined bar over every (λ, fold) fit. Safe to update from
    # multiple threads, and far more robust to render than one bar per λ
    # (multi-bar ANSI stacking is inconsistent across terminals).
    progress_bar = Progress(length(tasks); desc = "Cross-validation: ", enabled = params.verbose)

    _run_one = idx -> begin
        k, s = tasks[idx]
        λ = cv_reg.λ[k]
        data_train, data_val = splits[s]
        params_k = update_params(params; reg = _with_λ(params.reg, cv_idx, λ))
        # Silence each fold fit's own ADAM/LBFGS logging so it doesn't scroll
        # over the progress bars — same idea as train()'s multistart muting.
        params_k = update_params(params_k; verbose = false)
        task_rng = Random.Xoshiro(seeds[idx])

        # Multithread within each training loop is set to false
        results_ks = train(data_train, params_k, task_rng, θ_trained, regressor; n_runs = n_runs, parallel = false)
        β = params.train_initial_condition ?
            ComponentVector(θ = results_ks.θ, u0 = results_ks.u0) :
            ComponentVector(θ = results_ks.θ)
        all_scores[k][s] = loss_empirical(β, data_val, params_k, results_ks.regressor)
        next!(progress_bar)
    end

    if parallel
        if Threads.nthreads() == 1
            @warn "[train_cv] parallel=true but Threads.nthreads() == 1 — running sequentially. Start Julia with `--threads=auto` (or set JULIA_NUM_THREADS) to actually parallelize."
        else
            @info "[train_cv] Running $(length(tasks)) (λ, fold) fits across $(Threads.nthreads()) threads."
        end
        Threads.@threads for idx in eachindex(tasks)
            _run_one(idx)
        end
    else
        for idx in eachindex(tasks)
            _run_one(idx)
        end
    end

    finish!(progress_bar)

    if params.verbose
        for k = 1:n_candidates
            @info "[train_cv] λ = $(cv_reg.λ[k]) — average validation loss: $(mean(all_scores[k]))"
        end
    end

    best_idx = argmin(mean.(all_scores))
    best_λ = cv_reg.λ[best_idx]
    params.verbose && @info "[train_cv] Best λ = $(best_λ) (validation loss = $(mean(all_scores[best_idx])))"

    final_results = if refit
        final_params = update_params(params; reg = _with_λ(params.reg, cv_idx, best_λ))
        train(data, final_params, rng, θ_trained, regressor; n_runs = n_runs, parallel = parallel)
    else
        nothing
    end

    λ_results = Vector{Union{Results,Nothing}}(nothing, n_candidates)
    if refit_all
        for k = 1:n_candidates
            λ = cv_reg.λ[k]
            λ_params = update_params(params; reg = _with_λ(params.reg, cv_idx, λ))
            λ_results[k] = train(data, λ_params, rng, θ_trained, regressor; n_runs = n_runs, parallel = parallel)
        end
    end

    return CVResult(
        best_results = final_results,
        all_results  = λ_results,
        λ_grid       = cv_reg.λ,
        scores       = all_scores,
        best_λ       = best_λ,
    )
end

"""
    _with_λ(reg_vector, idx, λ)

Returns a copy of `reg_vector` with the `RegularizationCV` term at `idx`
replaced by a plain `Regularization` fixed at the scalar `λ`. Other entries
are left untouched.
"""
function _with_λ(reg_vector, idx::Int, λ::AbstractFloat)
    cv_reg = reg_vector[idx]
    reg_k = Regularization(order = cv_reg.order, power = cv_reg.power, λ = λ, diff_mode = cv_reg.diff_mode)
    return [i == idx ? reg_k : r for (i, r) in enumerate(reg_vector)]
end

"""
    _cv_splits(data, rng; k)

Builds the `k` train/validation pairs of `SphereData` for `k`-fold
cross-validation. Observations are randomly shuffled (using `rng`) and cut
into `k` folds; fold *sizes* are not randomized — they're fixed to
`ceil(N / k)` for the first `N % k` folds and `floor(N / k)` for the rest, so
folds differ in size by at most one observation. For each fold, the
validation set is that fold and the training set is the other `k - 1` folds
combined.
"""
function _cv_splits(data::SphereData, rng; k::Int)
    N = length(data.times)
    @assert k ≥ 2 "[SphereUDE] k-fold cross-validation requires k ≥ 2, got k = $(k)."
    @assert N ≥ k "[SphereUDE] Cannot make $(k) folds from $(N) observations."

    perm = randperm(rng, N)

    base, remainder = divrem(N, k)
    fold_sizes = [i ≤ remainder ? base + 1 : base for i = 1:k]

    fold_idxs = Vector{Vector{Int}}(undef, k)
    start = 1
    for i = 1:k
        stop = start + fold_sizes[i] - 1
        fold_idxs[i] = perm[start:stop]
        start = stop + 1
    end

    # `SphereData` requires sorted times whenever there are repeated values
    # (see `MakeVectorUnique`), so subsets must preserve chronological order
    # rather than the shuffled order used to assign observations to folds.
    function _subset(idx)
        idx_sorted = sort(idx)
        return SphereData(
            times = data.times[idx_sorted],
            directions = data.directions[:, idx_sorted],
            kappas = isnothing(data.kappas) ? nothing : data.kappas[idx_sorted],
            L = data.L,
        )
    end

    return [
        (_subset(reduce(vcat, fold_idxs[(1:k).!=i])), _subset(fold_idxs[i])) for i = 1:k
    ]
end
