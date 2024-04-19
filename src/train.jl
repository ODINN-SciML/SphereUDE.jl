export train

# For L1 regularization relu_cap works better, but for L2 I think is better to include sigmoid
function get_NN(params, rng, θ_trained)
    # Define neural network 
    U = Lux.Chain(
        Lux.Dense(1,  5,  relu_cap), # explore discontinuity function for activation
        Lux.Dense(5,  10, relu_cap), 
        Lux.Dense(10, 5,  relu_cap),
        # Lux.Dense(1,  5,  sigmoid), 
        # Lux.Dense(5,  10, sigmoid), 
        # Lux.Dense(10, 5,  sigmoid), 
        Lux.Dense(5,  3,  x->sigmoid_cap(x; ω₀=params.ωmax))
    )
    θ, st = Lux.setup(rng, U)
    return U, θ, st
end

function train(data::AD,
               params::AP,
               rng, 
               θ_trained=[]) where{AD <: AbstractData, AP <: AbstractParameters}

    # Raise warnings 
    raise_warnings(data::AD, params::AP)

    U, θ, st = get_NN(params, rng, θ_trained)

    function ude_rotation!(du, u, p, t)
        # Angular momentum given by network prediction
        L = U([t], p, st)[1]
        du .= cross(L, u)
    end

    prob_nn = ODEProblem(ude_rotation!, params.u0, [params.tmin, params.tmax], θ)

    function predict(θ::ComponentVector; u0=params.u0, T=data.times) 
        _prob = remake(prob_nn, u0=u0, 
                       tspan=(min(T[1], params.tmin), max(T[end], params.tmax)), 
                       p = θ)
        Array(solve(_prob, params.solver, saveat=T,
                    abstol=params.abstol, reltol=params.reltol,
                    sensealg=params.sensealg))
    end

    function loss(θ::ComponentVector)
        u_ = predict(θ)
        # Empirical error
        # l_emp = mean(abs2, u_ .- data.directions)
        if isnothing(data.kappas)
            # The 3 is needed since the mean is computen on a 3xN matrix
            l_emp = 1 - 3.0 * mean(u_ .* data.directions)
        else
            l_emp = norm(data.kappas)^2 - 3.0 * mean(data.kappas .* u_ .* data.directions)
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

    function regularization(θ::ComponentVector, reg::AbstractRegularization; n_nodes=100)
        
        # Create (uniform) spacing time
        # Δt = (params.tmax - params.tmin) / n_nodes
        # times_reg = collect(params.tmin:Δt:params.tmax)
        # LinRange does not propagate thought the backprop step!
        # times_reg = collect(LinRange(params.tmin, params.tmax, n_nodes))
        l_ = 0.0
        if reg.order==0
            l_ += quadrature(t -> norm(U([t], θ, st)[1])^reg.power, params.tmin, params.tmax, n_nodes)
        elseif reg.order==1
            if reg.diff_mode=="AD"
                throw("Method not working well.")
                # Compute gradient using automatic differentiaion in the NN
                # This currently doesn't run... too slow.
                for t in times_reg
                    # Try ReverseDiff
                    grad = Zygote.jacobian(first ∘ U, [t], θ, st)[1]
                    l_ += norm(grad)^reg.power
                end
            elseif reg.diff_mode=="FD"
                # Finite differences 
                ϵ = 0.1 * (params.tmax - params.tmin) / n_nodes
                l_ += quadrature(t -> norm(central_fdm(τ -> (first ∘ U)([τ], θ, st), t, ϵ=ϵ))^reg.power, params.tmin, params.tmax, n_nodes)
            elseif reg.diff_mode=="CS"
                # Complex step differentiation
                l_ += quadrature(t -> norm(complex_step_differentiation(τ -> (first ∘ U)([τ], θ, st), t))^reg.power, params.tmin, params.tmax, n_nodes) 
            else
                throw("Method not implemented.")
            end
        else
            throw("Method not implemented.")
        end
        return reg.λ * l_
    end

    losses = Float64[]
    callback = function (p, l)
        push!(losses, l)
        if length(losses) % 100 == 0
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
    fit_times = collect(range(params.tmin,params.tmax, length=200))
    fit_directions = predict(θ_trained, T=fit_times)

    return Results(θ_trained=θ_trained, U=U, st=st,
                   fit_times=fit_times, fit_directions=fit_directions)
end
