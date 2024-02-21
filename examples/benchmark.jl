using Pkg; Pkg.activate(".")
using Revise 

using LinearAlgebra, Statistics, Distributions 
using OrdinaryDiffEq
using SciMLSensitivity
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using BenchmarkTools, Infiltrator
# Random seed
using Random
rng = Random.default_rng()
Random.seed!(rng, 666)

using SphereUDE

##############################################################
###############  Simulation of Simple Example ################
##############################################################

function benchmark()

    # Total time simulation
    tspan = [0, 160.0]
    # Number of sample points
    N_samples = 50
    # Times where we sample points
    times_samples = sort(rand(sampler(Uniform(tspan[1], tspan[2])), N_samples))

    # Expected maximum angular deviation in one unit of time (degrees)
    Δω₀ = 1.0   
    # Angular velocity 
    ω₀ = Δω₀ * π / 180.0
    # Change point
    τ₀ = 65.0
    # Angular momentum
    L0 = ω₀    .* [1.0, 0.0, 0.0]
    L1 = 0.5ω₀ .* [0.0, 1/sqrt(2), 1/sqrt(2)]

    # Solver tolerances 
    reltol = 1e-7
    abstol = 1e-7

    # Solvers to be benchmarked
    #solvers = [Tsit5(), AutoTsit5(Rosenbrock23()), AutoVern7(Rodas4()), AutoVern9(Rodas4())]
    
    #solvers = [BS5(), OwrenZen5(), OwrenZen3(), BS3(), Tsit5()]
    solvers = [BS5(), Tsit5()]
    sensealgs = [QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))]
    #sensealgs = [QuadratureAdjoint(autojacvec=ReverseDiffVJP(true)), InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))]

    function L_true(t::Float64; τ₀=τ₀, p=[L0, L1])
        if t < τ₀
            return p[1]
        else
            return p[2]
        end
    end

    function true_rotation!(du, u, p, t)
        L = L_true(t; τ₀=τ₀, p=p)
        du .= cross(L, u)
    end

    i = 1
    for sensealg in sensealgs
        for solver in solvers

            println("\n Benchmarking ", sensealg, "and \n ", solver, "\n")

            prob = ODEProblem(true_rotation!, [0.0, 0.0, -1.0], tspan, [L0, L1])
            bm1 = @benchmark solve($prob, $solver, reltol=$reltol, abstol=$abstol, saveat=$times_samples)
            display(bm1)
            true_sol  = solve(prob, solver, reltol=reltol, abstol=abstol, saveat=times_samples)

            # Add Fisher noise to true solution 
            X_noiseless = Array(true_sol)
            X_true = X_noiseless + FisherNoise(kappa=200.) 

            ##############################################################
            #######################  Training  ###########################
            ##############################################################

            data   = SphereData(times=times_samples, directions=X_true, kappas=nothing, L=L_true)

            regs = [Regularization(order=1, power=1.0, λ=0.001, diff_mode="Finite Differences"), 
                    Regularization(order=0, power=2.0, λ=0.1, diff_mode="Finite Differences")]

            params = SphereParameters(tmin=tspan[1], tmax=tspan[2], 
                                    reg=regs, 
                                    u0=[0.0, 0.0, -1.0], ωmax=ω₀, reltol=reltol, abstol=abstol,
                                    niter_ADAM=1000, niter_LBFGS=600, solver=solver, sensealg= sensealg)
            
            bm2 = @benchmark train($data, $params, $rng, nothing)
            display(bm2)
            #results = train(data, params, rng, nothing)

            ##############################################################
            ######################  PyCall Plots #########################
            ##############################################################

            #plot_sphere(data, results, -20., 150., saveas="examples/double_rotation/plot_sphere_$i.pdf", title="$sensealg - \n $solver")
            #plot_L(data, results, saveas="examples/double_rotation/plot_L_$i.pdf", title="$sensealg - \n $solver")
            i += 1
        end
    end

    # Print the timings in the default way
    #show(to)

end

benchmark()