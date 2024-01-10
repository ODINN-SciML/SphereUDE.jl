export train

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

function train(data::AbstractData,
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
        # l_emp = mean(abs2, u_ .- data.directions)
        if isnothing(data.kappas)
            # The 3 is needed since the mean is computen on a 3xN matrix
            l_emp = 1 - 3 * mean(u_ .* data.directions)
        else
            l_emp = norm(data.kappas)^2 - 3 * mean(data.kappas .* u_ .* data.directions)
        end
        # Regularization
        l_reg = 0.0
        if !isnothing(params.reg)
            # for (order, power, λ) in params.reg    
            for reg in params.reg    
                # l_reg += reg.λ * regularization(θ; order=reg.order, power=reg.power)      
                l_reg += regularization(θ, reg)      
            end 
        end
        return l_emp + l_reg
    end

    function regularization(θ, reg::AbstractRegularization; timesteps=100)
        # Create (uniform) spacing time
        Δt = (params.tmax - params.tmin) / timesteps
        times_reg = collect(params.tmin:Δt:params.tmax)
        # LinRange does not propagate thought the backprop step!
        # times_reg = collect(LinRange(params.tmin, params.tmax, timesteps))
        l_ = 0.0
        if reg.order==0
            for t in times_reg
                l_ += norm(U([t], θ, st)[1])^reg.power
            end
        elseif reg.order==1
            if reg.diff_mode=="AD"
                # Compute gradient using automatic differentiaion in the NN
                # This currently doesn't run... too slow.
                for t in times_reg
                    # Try ReverseDiff
                    grad = Zygote.jacobian(first ∘ U, [t], θ, st)[1]
                    l_ += norm(grad)^reg.power
                end
            elseif reg.diff_mode=="Finite Differences"
                # Compute finite differences
                # Do this with alocating memory!!! 
                for i in 2:timesteps
                    t₀, t₁ = times_reg[(i-1):i]
                    L₀ = (first ∘ U)([t₀], θ, st)
                    L₁ = (first ∘ U)([t₁], θ, st)
                    grad = (L₁ .- L₀) / (t₁-t₀)
                    l_ += norm(grad)^reg.power
                end
            else
                throw("Method no implemented.")
            end
        else
            throw("Method no implemented.")
        end
        return reg.λ * l_
    end

    losses = Float64[]
    callback = function (p, l)
        push!(losses, l)
        if length(losses) % 20 == 0
            println("Current loss after $(length(losses)) iterations: $(losses[end])")
        end
        return false
    end

    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction((x, θ) -> loss(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, ComponentVector{Float64}(θ))

    res1 = Optimization.solve(optprob, ADAM(0.002), callback=callback, maxiters=params.niter_ADAM)
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
