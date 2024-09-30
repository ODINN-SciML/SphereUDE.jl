export train

function get_NN(params, rng, θ_trained)
    # Define neural network 
    
    # For L1 regularization relu_cap works better, but for L2 I think is better to include sigmoid
    if isL1reg(params.reg)
        @warn "[SphereUDE] Using ReLU activation functions for neural network due to L1 regularization."
        U = Lux.Chain(
            Lux.Dense(1,  5,  sigmoid), 
            Lux.Dense(5,  10, sigmoid), 
            Lux.Dense(10, 5,  sigmoid), 
            # Lux.Dense(1,  5,  relu_cap), 
            # Lux.Dense(5,  10, relu_cap), 
            # Lux.Dense(10, 5,  relu_cap), 
            Lux.Dense(5, 3, Base.Fix2(sigmoid_cap, params.ωmax))
            # Lux.Dense(5, 3, Base.Fix2(relu_cap, params.ωmax))
        )
    else        
        U = Lux.Chain(
            Lux.Dense(1,  5,  sigmoid), 
            Lux.Dense(5,  10, sigmoid), 
            Lux.Dense(10, 5,  sigmoid),
            Lux.Dense(5, 3, Base.Fix2(sigmoid_cap, params.ωmax))
        )
    end
    θ, st = Lux.setup(rng, U)
    return U, θ, st
end

"""
    predict_L(t, NN, θ, st)

Predict value of rotation given by L given by the neural network. 
"""
function predict_L(t, NN, θ, st)
    return NN([t], θ, st)[1]
end

"""
    train()

Training function. 
"""
function train(data::AD,
               params::AP,
               rng, 
               θ_trained=[]) where {AD <: AbstractData, AP <: AbstractParameters}

    # Raise warnings 
    raise_warnings(data::AD, params::AP)

    U, θ, st = get_NN(params, rng, θ_trained)

    # Set component vector for Optimization
    if params.train_initial_condition
        β = ComponentVector{Float64}(θ=θ, u0=params.u0)
    else
        β = ComponentVector{Float64}(θ=θ)
    end

    function ude_rotation!(du, u, p, t)
        # Angular momentum given by network prediction
        L = predict_L(t, U, p, st)
        du .= cross(L, u)
    end

    prob_nn = ODEProblem(ude_rotation!, params.u0, [params.tmin, params.tmax], β.θ)

    function predict(β::ComponentVector; T=data.times) 
        if params.train_initial_condition
            _prob = remake(prob_nn, u0=β.u0 / norm(β.u0), # We enforce the norm=1 condition again here
                           tspan=(min(T[1], params.tmin), max(T[end], params.tmax)), 
                           p = β.θ)
        else
            _prob = remake(prob_nn, u0=params.u0, 
                          tspan=(min(T[1], params.tmin), max(T[end], params.tmax)), 
                          p = β.θ)
        end
        sol = solve(_prob, params.solver, saveat=T,
                    abstol=params.abstol, reltol=params.reltol,
                    sensealg=params.sensealg, 
                    dtmin=1e-4 * (params.tmax - params.tmin), force_dtmin=true) # Force minimum step in case L(t) changes drastically due to bad behaviour of neural network
        return Array(sol), sol.retcode
    end

    function loss(β::ComponentVector)
        u_, retcode = predict(β)

        # If numerical integration fails or bad choice of parameter, return infinity
        if retcode != :Success
            @warn "[SphereUDE] Numerical solver not converging. This can be causes by numerical innestabilities around a bad choice of parameter."
            return Inf
        end

        # Record the value of each individual loss to the total loss function for hyperparameter selection.
        loss_dict = Dict()
        
        # Empirical error
        if isnothing(data.kappas)
            # The 3 is needed since the mean is computen on a 3xN matrix
            l_emp = 3.0 * mean(abs2.(u_ .- data.directions))
            # l_emp = 1 - 3.0 * mean(u_ .* data.directions)
        else
            l_emp = mean(data.kappas .* abs2.(u_ .- data.directions), dims=1)
            # l_emp = norm(data.kappas)^2 - 3.0 * mean(data.kappas .* u_ .* data.directions)
        end
        loss_dict["Empirical"] = l_emp
        
        # Regularization
        l_reg = 0.0
        if !isnothing(params.reg)
            # for (order, power, λ) in params.reg    
            for reg in params.reg    
                reg₀ = regularization(β.θ, reg)
                l_reg += reg₀      
                loss_dict["Regularization (order=$(reg.order), power=$(reg.power))"] = reg₀
            end 
        end
        return l_emp + l_reg, loss_dict
    end

    # Loss function to be called for multiple shooting
    function loss_function(data, pred)

        # Empirical error
        l_emp = 3.0 * mean(abs2.(pred .- data))
   
        # Regularization
        l_reg = 0.0
        if !isnothing(params.reg)
            for reg in params.reg    
                reg₀ = regularization(β.θ, reg)
                l_reg += reg₀      
            end 
        end

        return l_emp + l_reg
    end

    # Define parameters for Multiple Shooting
    group_size = 10
    continuity_term = 100

    ps = ComponentArray(θ) # are these necesary?
    pd, pax = getdata(ps), getaxes(ps) 

    function continuity_loss(u_pred, u_initial)
        if !isapprox(norm(u_initial), 1.0, atol=1e-6) || !isapprox(norm(u_pred), 1.0, atol=1e-6)
            @warn "Directions during multiple shooting are not in the sphere. Small deviations from unit norm observed:" 
            @show norm(u_initial), norm(u_pred)
        end
        return sum(abs2, u_pred - u_initial)
    end

    function loss_multiple_shooting(β::ComponentVector)

        ps = ComponentArray(β.θ, pax)

        if params.train_initial_condition
            _prob = remake(prob_nn, u0=β.u0 / norm(β.u0), # We enforce the norm=1 condition again here
                           tspan=(min(data.times[1], params.tmin), max(data.times[end], params.tmax)), 
                           p = β.θ)
        else
            _prob = remake(prob_nn, u0=params.u0, 
                          tspan=(min(data.times[1], params.tmin), max(data.times[end], params.tmax)), 
                          p = β.θ)
        end
  
        return multiple_shoot(β.θ, data.directions, data.times, _prob, loss_function, continuity_loss, params.solver,
                                group_size; continuity_term, sensealg=params.sensealg)
    end

    function regularization(θ::ComponentVector, reg::AG; n_nodes=100) where {AG <: AbstractRegularization}
        
        l_ = 0.0
        if reg.order==0
            
            l_ += quadrature(t -> norm(predict_L(t, U, θ, st))^reg.power, params.tmin, params.tmax, n_nodes)
        
        elseif reg.order==1
            
            if typeof(reg.diff_mode) <: LuxNestedAD
                throw("Method not working well.")
            
            elseif typeof(reg.diff_mode) <: FiniteDifferences
                # Finite differences 
                ϵ = reg.diff_mode.ϵ
                l_ += quadrature(t -> norm(central_fdm(τ -> predict_L(τ, U, θ, st), t, ϵ=ϵ))^reg.power, params.tmin, params.tmax, n_nodes)
            
            elseif typeof(reg.diff_mode) <: ComplexStepDifferentiation
                # Complex step differentiation
                ϵ = reg.diff_mode.ϵ
                l_ += quadrature(t -> norm(complex_step_differentiation(τ -> predict_L(τ, U, θ, st), t, ϵ))^reg.power, params.tmin, params.tmax, n_nodes) 
            
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
        if length(losses) % 50 == 0
            println("Current loss after $(length(losses)) iterations: $(losses[end])")
        end
        if params.train_initial_condition
            p.u0 ./= norm(p.u0)
        end 
        return false
    end

    # Dispatch the right loss function
    f_loss = params.multiple_shooting ? loss_multiple_shooting : loss

    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction((x, β) -> (first ∘ f_loss)(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, β)

    res1 = Optimization.solve(optprob, ADAM(), callback=callback, maxiters=params.niter_ADAM, verbose=false)
    println("Training loss after $(length(losses)) iterations: $(losses[end])")

    if params.niter_LBFGS > 0
        optprob2 = Optimization.OptimizationProblem(optf, res1.u)
        res2 = Optimization.solve(optprob2, Optim.LBFGS(), callback=callback, maxiters=params.niter_LBFGS) #, reltol=1e-6)
        println("Final training loss after $(length(losses)) iterations: $(losses[end])")
    else
        res2 = res1
    end

    # Optimized NN parameters
    β_trained = res2.u
    θ_trained = β_trained.θ

    # Optimized initial condition
    if params.train_initial_condition
        u0_trained = Array(β_trained.u0) # β.u0 is a view type
    else
        u0_trained = params.u0
    end 

    # Final Fit 
    fit_times = collect(range(params.tmin,params.tmax, length=length(data.times)))
    fit_directions, _ = predict(β_trained, T=fit_times)

    # Recover final balance between different terms involved in the loss function to assess hyperparameter selection.
    _, loss_dict = loss(β_trained)
    pretty_table(loss_dict, sortkeys=true, header=["Loss term", "Value"])

    return Results(θ=θ_trained, u0=u0_trained, U=U, st=st,
                   fit_times=fit_times, fit_directions=fit_directions)
end
