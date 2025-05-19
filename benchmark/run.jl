import Pkg
Pkg.activate(dirname(Base.current_project()))

using SphereUDE
using BenchmarkTools
using Logging
using PrettyTables
using ColorSchemes
Logging.disable_logging(Logging.Info)
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 60

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
    # [Regularization(order = 1, power = 1.0, λ = 0.1, diff_mode = LuxNestedAD())],
]

numerical_solver = [SphereUDE.Tsit5()]

sensealg_types = [
    # SphereUDE.DummyAdjoint(),
    GaussAdjoint(autojacvec = ReverseDiffVJP(true)),
    InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)),
    # BacksolveAdjoint(),
    # BacksolveAdjoint(autojacvec = ReverseDiffVJP(false), checkpointing = false),
    SphereBackSolveAdjoint()
]

tolerances = [1e-6, 1e-12]
in_out_place = [false, true]
# tolerances = [1e-6]

# BenchmarkTools evaluates things at global scope
params_benchmark = []
for tol in tolerances, regs in regularization_types, place in in_out_place, sensealg in sensealg_types
    params = SphereParameters(
        tmin = tspan[1],
        tmax = tspan[2],
        reg = regs,
        train_initial_condition = false,
        out_of_place = place,
        multiple_shooting = false,
        u0 = [0.0, 0.0, -1.0],
        ωmax = ω₀,
        reltol = tol,
        abstol = tol,
        niter_ADAM = 11,
        niter_LBFGS = 0,
        verbose = false,
        sensealg = sensealg,
    )
    push!(params_benchmark, params)
end

benchmark_data = []
header = (
    ["Sensitivity", "Reg", "Tol", "out-of-place", "Time", "Alloc", "Memory"],
    ["", "", "", "","[ns]", "", "bites"]
)

for params in params_benchmark
    println("## Benchmark of $(params.reg), $(params.sensealg), tolerance = $(params.reltol)")
    println("> Training for a total of $(params.niter_ADAM+params.niter_LBFGS) epochs")
    trial = @benchmark train(data, $params, $rng, nothing, nothing)
    # display(trial)
    # println("")
    push!(benchmark_data, ["$(params.sensealg)", "$(params.reg)", "$(params.reltol)", "$(params.out_of_place)", mean(trial.times), trial.allocs, trial.memory])
end

h1 = Highlighter(
    (data, i, j) -> j == 4 && data[i, j] >= mean(data[2:end, 4]),
    bold       = true,
    foreground = :red )

h2 = Highlighter(
    (data,i,j)->j == 4 && data[i, j] <= 1.2 * minimum(data[2:end,4]),
    bold       = true,
    foreground = :green
)

formated_benchmark_data = permutedims(hcat(benchmark_data...))
formated_benchmark_data[:,1] .=  (x -> split(x, "{")[begin]).(formated_benchmark_data[:,1])

pretty_table(
    formated_benchmark_data;
    sortkeys = true,
    header = header,
    linebreaks = true,
    header_crayon = crayon"yellow bold",
    highlighters = (h1, h2),
    formatters = ft_printf("%.2e")
    )
