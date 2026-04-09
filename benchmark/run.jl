import Pkg
Pkg.activate(@__DIR__)

using SphereUDE
using BenchmarkTools
using Logging
using PrettyTables
using ColorSchemes
using SciMLBase
using SimpleDiffEq
using Lux

Logging.disable_logging(Logging.Info)
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 30
BenchmarkTools.DEFAULT_PARAMETERS.samples = 5

using Distributions, Statistics, LinearAlgebra
using Printf
using SciMLSensitivity
using OrdinaryDiffEqTsit5, OrdinaryDiffEqVerner, OrdinaryDiffEqHighOrderRK, OrdinaryDiffEqRosenbrock

println("# Performance benchmark")

# Set to :light for a quick run of the top 5 methods,
# or :heavy for the full sweep of all sensealgs and solvers.
benchmark_mode = :light

using Random
rng = Random.default_rng()
Random.seed!(666)

### Create a simple simulation

# Total time simulation
tspan = [0, 160.0]
# Number of sample points
N_samples = 100
# Times where we sample points
# Include initial and final time for backadjoint
random_times = [tspan[1]; tspan[2]; rand(sampler(Uniform(tspan[1], tspan[2])), N_samples - 2)]
times_samples = sort(random_times)

# Expected maximum angular deviation in one unit of time (degrees)
Δω₀ = 1.0
# Angular velocity
ω₀ = Δω₀ * π / 180.0

# Create simple example
X = zeros(3, N_samples)
X[3, :] .= 1
X[1, :] = LinRange(0, 1, N_samples)
X = X ./ norm.(eachcol(X))'

data = SphereData(times = times_samples, directions = X, kappas = nothing, L = nothing)

# Custom neural network
n_fourier_features = 4
# Expected maximum angular deviation in one unit of time (degrees)
Δω₀ = 2.0
# Angular velocity
ωmax = Δω₀ * π / 180.0

U = Lux.Chain(
    # Scale function to bring input to [-1.0, 1.0]
    Lux.WrappedFunction(x -> scale_input(x; xmin = tspan[1], xmax = tspan[2])),
    # Fourier feautues
    Lux.WrappedFunction(x -> fourier_feature(x; n = n_fourier_features)),
    Lux.Dense(2 * n_fourier_features, 10, tanh),
    Lux.Dense(10, 10, tanh),
    Lux.Dense(10, 10, tanh),
    Lux.Dense(10, 3, tanh),
    # Output function to scale output to have norm less than ωmax
    Lux.WrappedFunction(x -> scale_norm(ωmax * x; scale = ωmax))
)

### Types of regularization to benchmark
regularization_types = [
    nothing,
    # [Regularization(order = 0, power = 1.0, λ = 0.1, diff_mode = nothing)],
    # [Regularization(order = 1, power = 1.0, λ = 0.1, diff_mode = FiniteDiff(1e-6))],
    # [Regularization(order = 1, power = 1.0, λ = 0.1, diff_mode = LuxNestedAD())],
]

tolerances = [1e-6]

# BenchmarkTools evaluates things at global scope
params_benchmark = []

if benchmark_mode == :light

    # -------------------------------------------------------
    # Light mode: best method per sensealg family (Tsit5 unless noted),
    # plus the two fastest overall (ReverseDiffAdjoint + SphereBackSolveAdjoint)
    # also tested with AutoTsit5 to check solver sensitivity.
    # -------------------------------------------------------
    light_combos = [
        # Top 2 overall — also compare solvers
        (ReverseDiffAdjoint(),                                          Tsit5()),
        (ReverseDiffAdjoint(),                                          AutoTsit5(Rosenbrock23())),
        (SphereBackSolveAdjoint(),                                      Tsit5()),
        (SphereBackSolveAdjoint(),                                      AutoTsit5(Rosenbrock23())),
        # Best per SciMLSensitivity family (all with Tsit5)
        (InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)),       Tsit5()),
        (BacksolveAdjoint(autojacvec = ReverseDiffVJP(true)),           Tsit5()),
        (GaussAdjoint(autojacvec = ReverseDiffVJP(true)),               Tsit5()),
        (QuadratureAdjoint(autojacvec = ReverseDiffVJP(true)),          Tsit5()),
        (GaussKronrodAdjoint(autojacvec = ReverseDiffVJP(true)),        Tsit5()),
    ]

    for tol in tolerances, regs in regularization_types, (sensealg, solver) in light_combos
        if typeof(sensealg) <: SphereBackSolveAdjoint
            sensealg = SphereBackSolveAdjoint(reltol = tol, abstol = tol)
        end
        params = SphereParameters(
            tmin = tspan[1],
            tmax = tspan[2],
            reg = regs,
            train_initial_condition = false,
            out_of_place = false,
            multiple_shooting = false,
            u0 = [0.0, 0.0, -1.0],
            ωmax = ω₀,
            solver = solver,
            reltol = tol,
            abstol = tol,
            niter_ADAM = 20,
            niter_LBFGS = 0,
            verbose = false,
            sensealg = sensealg,
        )
        push!(params_benchmark, params)
    end

else  # :heavy

    # -------------------------------------------------------
    # Heavy mode: full sweep — all sensealgs × all solvers
    # -------------------------------------------------------
    numerical_solver = [
        Tsit5(),
        Vern7(),
        # Vern9(),
        AutoTsit5(Rosenbrock23()),
        DP8(),
    ]

    sensealg_types = [
        InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)),
        InterpolatingAdjoint(autojacvec = ReverseDiffVJP(false)),
        InterpolatingAdjoint(autojacvec = ZygoteVJP()),
        InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true), checkpointing = true),
        InterpolatingAdjoint(autojacvec = ZygoteVJP(), checkpointing = true),
        BacksolveAdjoint(autojacvec = ReverseDiffVJP(true)),
        BacksolveAdjoint(autojacvec = ReverseDiffVJP(false)),
        BacksolveAdjoint(autojacvec = ZygoteVJP()),
        BacksolveAdjoint(autojacvec = TrackerVJP()),
        BacksolveAdjoint(autojacvec = ReverseDiffVJP(true), checkpointing = true),
        BacksolveAdjoint(autojacvec = ZygoteVJP(), checkpointing = true),
        QuadratureAdjoint(autojacvec = ReverseDiffVJP(true)),
        QuadratureAdjoint(autojacvec = ReverseDiffVJP(false)),
        QuadratureAdjoint(autojacvec = ZygoteVJP()),
        GaussAdjoint(autojacvec = ReverseDiffVJP(true)),
        GaussAdjoint(autojacvec = ReverseDiffVJP(false)),
        GaussAdjoint(autojacvec = ZygoteVJP()),
        GaussKronrodAdjoint(autojacvec = ReverseDiffVJP(true)),
        GaussKronrodAdjoint(autojacvec = ReverseDiffVJP(false)),
        ZygoteAdjoint(),
        ReverseDiffAdjoint(),
        TrackerAdjoint(),
        # QuadratureAdjoint(autojacvec = MooncakeVJP()),  # MooncakeVJP not available in current SciMLSensitivity version
        MooncakeAdjoint(),
        SphereBackSolveAdjoint(),
    ]

    # Main sweep: all sensealgs × all solvers, in-place
    for tol in tolerances, regs in regularization_types, solver in numerical_solver, sensealg in sensealg_types
        if typeof(sensealg) <: SphereBackSolveAdjoint
            sensealg = SphereBackSolveAdjoint(
                solver = sensealg.solver,
                reltol = tol,
                abstol = tol
            )
        end
        params = SphereParameters(
            tmin = tspan[1],
            tmax = tspan[2],
            reg = regs,
            train_initial_condition = false,
            out_of_place = false,
            multiple_shooting = false,
            u0 = [0.0, 0.0, -1.0],
            ωmax = ω₀,
            solver = solver,
            reltol = tol,
            abstol = tol,
            niter_ADAM = 20,
            niter_LBFGS = 0,
            verbose = false,
            sensealg = sensealg,
        )
        push!(params_benchmark, params)
    end

    # ZygoteVJP in-place vs out-of-place comparison (one solver per sensealg variant)
    zygote_sensealgs = [
        InterpolatingAdjoint(autojacvec = ZygoteVJP()),
        QuadratureAdjoint(autojacvec = ZygoteVJP()),
        GaussAdjoint(autojacvec = ZygoteVJP()),
        BacksolveAdjoint(autojacvec = ZygoteVJP()),
    ]
    for tol in tolerances, regs in regularization_types, sensealg in zygote_sensealgs, place in [true, false]
        params = SphereParameters(
            tmin = tspan[1],
            tmax = tspan[2],
            reg = regs,
            train_initial_condition = false,
            out_of_place = place,
            multiple_shooting = false,
            u0 = [0.0, 0.0, -1.0],
            ωmax = ω₀,
            solver = Tsit5(),
            reltol = tol,
            abstol = tol,
            niter_ADAM = 20,
            niter_LBFGS = 0,
            verbose = false,
            sensealg = sensealg,
        )
        push!(params_benchmark, params)
    end

end  # benchmark_mode

### Regressors to benchmark
# Match parameter counts for a fair comparison.
_nn_regressor, _nn_θ = NNRegressor(U, rng)
n_nn_params = length(_nn_θ)
# SplineRegressor has 3 × n_basis parameters (x, y, z control points).
n_basis_fair = 50
println("NN parameters: $n_nn_params  →  SplineRegressor n_basis = $n_basis_fair ($(3*n_basis_fair) params)")

# Each entry: (label, builder) where builder(params, rng) → AbstractRegressor
regressor_builders = [
    ("NNRegressor ($n_nn_params params)",         (params, rng) -> NNRegressor(U, rng)[1]),
    ("SplineRegressor ($(3*n_basis_fair) params)", (params, rng) -> SplineRegressor(
        tmin    = Float64(params.tmin),
        tmax    = Float64(params.tmax),
        n_basis = n_basis_fair,
        degree  = 3,
        ωmax    = Float64(params.ωmax),
    )),
]

println("Benchmarking $(length(params_benchmark)) sensealg combinations × $(length(regressor_builders)) regressors = $(length(params_benchmark) * length(regressor_builders)) total runs.")

# benchmark_results[(sensealg_key, reg_label)] = (time_s, time_per_epoch_s, memory_mb)
benchmark_results = Dict()

short_name(x) = split(string(typeof(x)), "{")[begin]
combo_key(params) = "$(short_name(params.sensealg)) | $(short_name(params.solver)) | oop=$(params.out_of_place)"

global combo_id = 0
for (i, params) in enumerate(params_benchmark)
    if (typeof(params.sensealg) <: SciMLBase.AbstractAdjointSensitivityAlgorithm) & params.out_of_place
        continue
    end
    global combo_id += 1
    key = combo_key(params)
    n_epochs = params.niter_ADAM + params.niter_LBFGS

    for (reg_label, reg_builder) in regressor_builders
        try
            println("[$combo_id] $(short_name(params.sensealg)) | $reg_label | tol=$(params.reltol) | epochs=$n_epochs")
            _fixed_regressor = reg_builder(params, Xoshiro(666))
            trial = @benchmark train(data, $params, _rng, nothing, $_fixed_regressor) setup=(_rng = Xoshiro(666))
            t_s   = mean(trial.times) / 1e9
            t_ep  = t_s / n_epochs
            mem   = trial.memory / 1e6
            benchmark_results[(key, reg_label)] = (t_s, t_ep, mem)
        catch _err
            @warn "Simulation with $reg_label + $(params) did not work." exception=_err
            benchmark_results[(key, reg_label)] = (NaN, NaN, NaN)
        end
    end
end

# Build side-by-side table: one row per sensealg combo, columns per regressor
trunc25(s) = length(s) > 25 ? s[1:25] * "…" : s

reg_labels = first.(regressor_builders)
all_keys   = unique([combo_key(p) for p in params_benchmark
                     if !(typeof(p.sensealg) <: SciMLBase.AbstractAdjointSensitivityAlgorithm && p.out_of_place)])

# Sort rows by mean time across regressors
all_keys = sort(all_keys, by = k -> mean(
    get(benchmark_results, (k, rl), (NaN,))[1]
    for rl in reg_labels
    if haskey(benchmark_results, (k, rl))
))

time_cols   = [[@sprintf("%.2f", get(benchmark_results, (k, rl), (NaN,NaN,NaN))[2] * 1000) for k in all_keys] for rl in reg_labels]
mem_cols    = [[@sprintf("%.1f", get(benchmark_results, (k, rl), (NaN,NaN,NaN))[3]) for k in all_keys] for rl in reg_labels]

table_data = hcat(
    trunc25.(all_keys),
    time_cols[1], mem_cols[1],
    time_cols[2], mem_cols[2],
)

nn_label     = reg_labels[1]
spline_label = reg_labels[2]

pretty_table(
    table_data;
    column_labels = ["Sensitivity | Solver | OOP",
                     "Time/1000ep [s] ($nn_label)", "Mem [MB]",
                     "Time/1000ep [s] ($spline_label)", "Mem [MB]"],
    show_column_labels = true,
    style = TextTableStyle(first_line_column_label = crayon"yellow bold"),
)
