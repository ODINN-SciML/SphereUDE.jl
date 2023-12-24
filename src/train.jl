export train_sphere

function get_NN(params, rng, θ_trained)
    # Define neural network 
    U = Lux.Chain(
        Lux.Dense(1,  5,  tanh), 
        Lux.Dense(5,  10, tanh), 
        Lux.Dense(10, 5,  tanh), 
        Lux.Dense(5,  3,  x->sigmoid_cap(x; ω₀=params.ωmax))
    )
    # This is what we have in ODINN.jl, but not clear if neccesary
    #
    # UA = Flux.f64(UA)
    # # See if parameters need to be retrained or not
    # θ, UA_f = Flux.destructure(UA)
    # if !isempty(θ_trained)
    #     θ = θ_trained
    # end
    # return UA_f, θ

    θ, st = Lux.setup(rng, U)
    return U, θ, st
end

function train_sphere(data::AbstractData,
                      params::AbstractParameters,
                      rng, 
                      θ_trained=[])

    U, θ, st = get_NN(params, rng, θ_trained)

    function ude_rotation!(du, u, p, t)
        # Angular momentum given by network prediction
        L = U([t], p, st)[1]
        du .= cross(L, u)
        nothing
    end

    prob_nn = ODEProblem(ude_rotation!, params.u0, [params.tmin, params.tmax], θ)

    function predict(θ; u0=params.u0, T=data.times) 
        _prob = remake(prob_nn, u0=u0, 
                       tspan=(min(T[1], params.tmin), max(T[end], params.tmax)), 
                       p = θ)
        Array(solve(_prob, Tsit5(), saveat=T,
                    abstol=params.abstol, reltol=params.reltol,
                    sensealg=QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))))
    end

    function loss(θ)
        u_ = predict(θ)
        # Empirical error
        l_ = mean(abs2, u_ .- data.directions)
        return l_
        # TO DO: add regularization
        # ...
        # ...
    end

    losses = Float64[]
    callback = function (p, l)
        push!(losses, l)
        if length(losses) % 50 == 0
            println("Current loss after $(length(losses)) iterations: $(losses[end])")
        end
        return false
    end

    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction((x, θ) -> loss(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, ComponentVector{Float64}(θ))

    res1 = Optimization.solve(optprob, ADAM(0.001), callback=callback, maxiters=params.niter_ADAM)
    println("Training loss after $(length(losses)) iterations: $(losses[end])")

    optprob2 = Optimization.OptimizationProblem(optf, res1.u)
    res2 = Optimization.solve(optprob2, Optim.LBFGS(), callback=callback, maxiters=params.niter_LBFGS)
    println("Final training loss after $(length(losses)) iterations: $(losses[end])")

    # Optimized NN parameters
    θ_trained = res2.u

    # Final Fit 
    fit_times = collect(params.tmin:0.1:params.tmax)
    fit_directions = predict(θ_trained, T=fit_times)

    return Results(θ_trained=θ_trained, U=U, st=st,
                   fit_times=fit_times, fit_directions=fit_directions)
end



