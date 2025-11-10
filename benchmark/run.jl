import Pkg
Pkg.activate(dirname(Base.current_project()))

using SphereUDE
using BenchmarkTools
using Logging
using PrettyTables
using ColorSchemes
using SciMLBase
using Lux
using ODE
using SimpleDiffEq
using DifferentialEquations
Logging.disable_logging(Logging.Info)
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 60

using DifferentiationInterface
import Mooncake
import SciMLSensitivity: MooncakeVJP

using Distributions, Statistics, LinearAlgebra
using SciMLSensitivity

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

### Types of regularization to benchmark
regularization_types = [
    nothing,
    # [Regularization(order = 0, power = 1.0, λ = 0.1, diff_mode = nothing)],
    # [Regularization(order = 1, power = 1.0, λ = 0.1, diff_mode = FiniteDiff(1e-6))],
    # [Regularization(order = 1, power = 2.0, λ = 0.1, diff_mode = LuxNestedAD())],
]

numerical_solver = [
    SphereUDE.Tsit5(),
    # SimpleATsit5(),
    # ODE.ode45(),
    # AutoTsit5(Rosenbrock23()),
    # Vern9(),
    # Rodas5P()
    ]

sensealg_types = [
    # SphereUDE.DummyAdjoint(),
    GaussAdjoint(autojacvec = ReverseDiffVJP(true)),
    InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)),
    QuadratureAdjoint(autojacvec = ReverseDiffVJP(true)),
    # BacksolveAdjoint(autojacvec = ReverseDiffVJP(false), checkpointing = false),
    GaussAdjoint(autojacvec = MooncakeVJP()),
    InterpolatingAdjoint(autojacvec = MooncakeVJP()),
    QuadratureAdjoint(autojacvec = MooncakeVJP()),
    SphereBackSolveAdjoint()
]

tolerances = [1e-6]
in_out_place = [false, true]
# tolerances = [1e-6]

# BenchmarkTools evaluates things at global scope
params_benchmark = []
for tol in tolerances, regs in regularization_types, solver in numerical_solver, place in in_out_place, sensealg in sensealg_types

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
        out_of_place = place,
        multiple_shooting = false,
        u0 = [0.0, 0.0, -1.0],
        ωmax = ω₀,
        solver = solver,
        reltol = tol,
        abstol = tol,
        niter_ADAM = 20,
        niter_LBFGS = 20,
        verbose = false,
        sensealg = sensealg,
    )
    push!(params_benchmark, params)
end

n_fourier_features = 4
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
    Lux.WrappedFunction(x -> scale_norm(ω₀ * x; scale = ω₀))
)

println("Benchmarking in a total of $(length(params_benchmark)) combinations. This will required a total maximum of ~$(length(params_benchmark) * BenchmarkTools.DEFAULT_PARAMETERS.seconds / 60) minutes.")

benchmark_data = []
header = (
    ["Sensitivity", "Reg", "Solver", "Tol", "out-of-place", "Time", "Alloc", "Memory"],
    ["", "", "", "", "", "[ns]", "", "bites"]
)

for params in params_benchmark
    @show params.sensealg
    @show params.out_of_place
    try
        if (typeof(params.sensealg) <: SphereBackSolveAdjoint) & !params.out_of_place
            continue
        end
        if (typeof(params.sensealg) <: SciMLBase.AbstractAdjointSensitivityAlgorithm) & params.out_of_place
            continue
        end
        println("## Benchmark of $(params.reg), $(params.sensealg), tolerance = $(params.reltol)")
        println("> Training for a total of $(params.niter_ADAM+params.niter_LBFGS) epochs")
        trial = @benchmark train(data, $params, $rng, nothing, U)
        display(trial)
        println("")
        push!(benchmark_data, ["$(params.sensealg)", "$(params.reg)", "$(params.solver)", "$(params.reltol)", "$(params.out_of_place)", mean(trial.times), trial.allocs, trial.memory])
    catch _err
        @warn "Simulation with $(params) did not work."
    end
end

time_col = 6

h1 = Highlighter(
    (data, i, j) -> j == time_col && data[i, j] >= mean(data[2:end, time_col]),
    bold       = true,
    foreground = :red )

h2 = Highlighter(
    (data,i,j)->j == time_col && data[i, j] <= 1.2 * minimum(data[2:end, time_col]),
    bold       = true,
    foreground = :green
)

formated_benchmark_data = permutedims(hcat(benchmark_data...))
formated_benchmark_data[:,1] .=  (x -> split(x, "{")[begin]).(formated_benchmark_data[:,1])
formated_benchmark_data[:,3] .=  (x -> split(x, "{")[begin]).(formated_benchmark_data[:,3])

pretty_table(
    formated_benchmark_data;
    sortkeys = true,
    header = header,
    linebreaks = true,
    header_crayon = crayon"yellow bold",
    highlighters = (h1, h2),
    formatters = ft_printf("%.2e")
    )
