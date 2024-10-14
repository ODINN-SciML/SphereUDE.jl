export train

"""
    train()

Training function. 
"""
function train(data::AD,
               params::AP,
               rng, 
               θ_trained=[],
               model::Union{Chain, Nothing}=nothing) where {AD <: AbstractData, AP <: AbstractParameters}

    # Raise warnings 
    raise_warnings(data::AD, params::AP)
    
    # Set Neural Network
    if isnothing(model)
        U = get_default_NN(params, rng, θ_trained)
    else
        U = model
    end
    θ, st = Lux.setup(rng, U)

    # Set component vector for Optimization
    if params.train_initial_condition
        β = ComponentVector{Float64}(θ=θ, u0=params.u0)
    else
        β = ComponentVector{Float64}(θ=θ)
    end

    # Function to predict angular momentum. Used just for ODE part 
    # of the loss function
    function predict_L(t, NN, θ, st)
        smodel = StatefulLuxLayer{true}(NN, θ, st)
        return smodel([t]) 
    end

    # Sphere-constrained ODE
    function ude_rotation!(du, u, p, t)
        # Angular momentum given by network prediction
        L = predict_L(t, U, p, st)
        du .= cross(L, u)
    end

    global prob_nn = ODEProblem(ude_rotation!, params.u0, [params.tmin, params.tmax], β.θ)

    ### Callback
    losses = Float64[]
    callback = function (p, l)
        push!(losses, l)
        if length(losses) % 50 == 0
            @printf "Iteration: [%5d / %5d] \t Loss: %.9f \n" length(losses) (params.niter_ADAM+params.niter_LBFGS) losses[end]
        end
        if params.train_initial_condition
            p.u0 ./= norm(p.u0)
        end 
        return false
    end

    # Dispatch the right loss function
    if params.multiple_shooting
        throw("[SphereUDE] Method not implemented.")
        # f_loss(β) = loss_multiple_shooting
    else
        f_loss(β) = loss(β, data, params, U, st)
    end
    
    println("Loss after initalization: ", f_loss(β)[1])

    """
    Pretraining to find parameters without impossing regularization
    """
    if params.pretrain
        losses_pretrain = Float64[]
        callback_pretrain = function(p, l)
            push!(losses_pretrain, l)
            if length(losses_pretrain) % 100 == 0
                @printf "[Pretrain with no regularization] Iteration: [%5d / %5d] \t Loss: %.9f \n" length(losses_pretrain) (params.niter_ADAM+params.niter_LBFGS) losses_pretrain[end]
                # println("[Pretrain with no regularization] Current loss after $(length(losses_pretrain)) iterations: $(losses_pretrain[end])")
            end
            return false
        end
        optf₀ = Optimization.OptimizationFunction((x, β) -> loss_empirical(x), params.adtype)
        optprob₀ = Optimization.OptimizationProblem(optf₀, β)
        res₀ = Optimization.solve(optprob₀, ADAM(), callback=callback_pretrain, maxiters=params.niter_ADAM, verbose=false)
        optprob₁ = Optimization.OptimizationProblem(optf₀, res₀.u)
        res₁ = Optimization.solve(optprob₁, Optim.BFGS(; initial_stepnorm=0.01, linesearch=LineSearches.BackTracking()), callback=callback_pretrain, maxiters=params.niter_LBFGS)
        β = res₁.u
    end

    # To do: implement this with polyoptimizaion to put ADAM and BFGS in one step.
    # Maybe better to keep like this for the line search. 

    optf = Optimization.OptimizationFunction((x, β) -> (first ∘ f_loss)(x), params.adtype)
    optprob = Optimization.OptimizationProblem(optf, β)

    res1 = Optimization.solve(optprob, ADAM(), callback=callback, maxiters=params.niter_ADAM, verbose=true)
    println("Training loss after $(length(losses)) iterations: $(losses[end])")

    if params.niter_LBFGS > 0
        optprob2 = Optimization.OptimizationProblem(optf, res1.u)
        # res2 = Optimization.solve(optprob2, Optim.LBFGS(), callback=callback, maxiters=params.niter_LBFGS) #, reltol=1e-6)
        res2 = Optimization.solve(optprob2, Optim.BFGS(; initial_stepnorm=0.01, linesearch=LineSearches.BackTracking()), callback=callback, maxiters=params.niter_LBFGS) #, reltol=1e-6)
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
    fit_times = collect(range(params.tmin,params.tmax, length=1000))
    fit_directions, _ = predict(β_trained, params, fit_times)
    fit_rotations = reduce(hcat, (t -> predict_L(t, U, β_trained.θ, st)).(fit_times))

    # Recover final balance between different terms involved in the loss function to assess hyperparameter selection.
    _, loss_dict = f_loss(β_trained)
    pretty_table(loss_dict, sortkeys=true, header=["Loss term", "Value"])

    return Results(θ=θ_trained, u0=u0_trained, U=U, st=st,
                   fit_times=fit_times, fit_directions=fit_directions, 
                   fit_rotations=fit_rotations, losses=losses)
end