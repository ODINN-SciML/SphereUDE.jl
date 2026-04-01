import Pkg
Pkg.activate(dirname(Base.current_project()))

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

numerical_solver = [
    Tsit5(),
    Vern7(),
    # Vern9(),
    # AutoTsit5(Rosenbrock23()),
    # DP8(),
]

sensealg_types = [
    InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)),
    # InterpolatingAdjoint(autojacvec = ReverseDiffVJP(false)),
    # InterpolatingAdjoint(autojacvec = ZygoteVJP()),
    BacksolveAdjoint(autojacvec = ReverseDiffVJP(true)),
    # BacksolveAdjoint(autojacvec = ReverseDiffVJP(false)),
    # BacksolveAdjoint(autojacvec = ZygoteVJP()),
    # BacksolveAdjoint(autojacvec = TrackerVJP()),
    # QuadratureAdjoint(autojacvec = ReverseDiffVJP(true)),
    QuadratureAdjoint(autojacvec = ZygoteVJP()),
    # GaussAdjoint(autojacvec = ReverseDiffVJP(true)),
    GaussAdjoint(autojacvec = ZygoteVJP()),
    GaussKronrodAdjoint(autojacvec = ReverseDiffVJP(true)),
    # ZygoteAdjoint(),
    # ReverseDiffAdjoint(),
    # TrackerAdjoint(),
    # QuadratureAdjoint(autojacvec = MooncakeVJP()),
    MooncakeAdjoint(),
    SphereBackSolveAdjoint(),
]

tolerances = [1e-6]

# BenchmarkTools evaluates things at global scope
params_benchmark = []

# Main sweep: all sensealgs, in-place
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
        niter_ADAM = 10,
        niter_LBFGS = 10,
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
        niter_ADAM = 10,
        niter_LBFGS = 10,
        verbose = false,
        sensealg = sensealg,
    )
    push!(params_benchmark, params)
end

println("Benchmarking in a total of $(length(params_benchmark)) combinations.")

benchmark_data = []
header = ["Sensitivity", "Reg", "Solver", "Tol", "out-of-place", "Time [s]", "Time/epoch [s]", "Alloc", "Memory [MB]"]

short_name(x) = split(string(typeof(x)), "{")[begin]

for (i, params) in enumerate(params_benchmark)
    try
        if (typeof(params.sensealg) <: SciMLBase.AbstractAdjointSensitivityAlgorithm) & params.out_of_place
            continue
        end
        println("[$i/$(length(params_benchmark))] $(short_name(params.sensealg)) | tol=$(params.reltol) | out-of-place=$(params.out_of_place) | epochs=$(params.niter_ADAM)+$(params.niter_LBFGS)")
        # TODO: using a fresh RNG per sample is a workaround to avoid NaNs from bad NN
        # initializations on subsequent @benchmark samples. A permanent fix would be to
        # make the NN initialization robust to the random seed (e.g. by normalizing weights).
        trial = @benchmark train(data, $params, _rng, nothing, $U) setup=(_rng = Xoshiro(666))
        n_epochs = params.niter_ADAM + params.niter_LBFGS
        push!(benchmark_data, [i, "$(params.sensealg)", "$(params.reg)", "$(params.solver)", "$(params.reltol)", "$(params.out_of_place)", mean(trial.times) / 1e9, mean(trial.times) / 1e9 / n_epochs * 1000, trial.allocs, trial.memory / 1e6])
    catch _err
        @warn "Simulation with $(params) did not work." exception=_err
    end
end

trunc20(s) = length(s) > 20 ? s[1:20] * "…" : s

# Sort by total time (column 7)
sorted_data = sort(benchmark_data, by = r -> r[7])

table_data = hcat(
    getindex.(sorted_data, 1),                                    # ID
    trunc20.(getindex.(sorted_data, 2)),                          # Sensitivity
    trunc20.(getindex.(sorted_data, 4)),                          # Solver
    string.(getindex.(sorted_data, 6)),                           # Out-of-place
    [@sprintf("%.2f", r[7]) for r in sorted_data],               # Time [s]
    [@sprintf("%.2f", r[8]) for r in sorted_data],               # Time/1000 epochs [s]
    [@sprintf("%.2f", r[10]) for r in sorted_data],              # Memory [MB]
)

pretty_table(
    table_data;
    column_labels = ["ID", "Sensitivity", "Solver", "OOP", "Time [s]", "Time/1000 epochs [s]", "Memory [MB]"],
    show_column_labels = true,
    style = TextTableStyle(first_line_column_label = crayon"yellow bold"),
)
