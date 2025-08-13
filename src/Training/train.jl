export train

"""
    train()

Training function.
"""
function train(
    data::AD,
    params::AP,
    rng,
    θ_trained = [],
    model::Union{Chain,Nothing} = nothing,
) where {AD<:AbstractData,AP<:AbstractParameters}

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
        β = ComponentVector{Float64}(θ = θ, u0 = params.u0)
    else
        β = ComponentVector{Float64}(θ = θ)
    end

    ### Callback
    losses = Float64[]
    callback_print_closure(p, l) = callback_print(p, l, params, losses, f_loss)
    callback_proj_closure(p, l) = callback_proj(p, l, params)
    callback_stop_condition_closure(p, l) = callback_stop_condition(p, l, losses)
    callback(p, l) = CallbackOptimizationSet(
        p,
        l;
        callbacks = (
            callback_print_closure,
            callback_proj_closure,
            callback_stop_condition_closure,
        ),
    )

    # Dispatch the right loss function
    if params.multiple_shooting
        throw("[SphereUDE] Method not implemented.")
        # f_loss(β) = loss_multiple_shooting
    else
        f_loss(β) = loss(β, data, params, U, st)
    end

    """
    Pretraining to find parameters without impossing regularization
    """
    if params.pretrain
        @info "Pretraining without regularization for initialization."
        losses_pretrain = Float64[]
        n_pretrain = 300
        callback_pretrain = function (p, l)
            push!(losses_pretrain, l)
            return false
        end
        # Define the loss function with just empirical component
        f_loss_empirical(β) = loss_empirical(β, data, params, U, st)
        optf₀ =
            Optimization.OptimizationFunction((x, β) -> f_loss_empirical(x), params.adtype)
        optprob₀ = Optimization.OptimizationProblem(optf₀, β)
        res₀ = Optimization.solve(
            optprob₀,
            ADAM(params.ADAM_learning_rate),
            callback = callback_pretrain,
            maxiters = n_pretrain,
            verbose = false,
        )
        println(
            "Improvement due to pretrain: $(losses_pretrain[begin]) --> $(losses_pretrain[end])",
        )
        β = res₀.u
    end

    @info "Start optimization with ADAM"
    # if params.customgrad
    if isa(params.sensealg, SciMLBase.AbstractAdjointSensitivityAlgorithm)
        optf =
            Optimization.OptimizationFunction((β, _p) -> (first ∘ f_loss)(β), params.adtype)
    elseif isa(params.sensealg, AbstractAdjointMethod)
        @info "Training with custom gradient method."

        # Closure functions to deliver data loader
        loss_function(_β, _p) = (first ∘ f_loss)(_β)
        loss_grad!(_dβ, _β, _p) =
            rotation_grad!(_dβ, _β, data, params, U, st, params.sensealg)

        optf = Optimization.OptimizationFunction(loss_function, grad = loss_grad!, NoAD())
    end

    # Create dummy data loader
    train_loader = DataLoader([666.0], batchsize = 1, shuffle = false)
    optprob = Optimization.OptimizationProblem(optf, β, train_loader)
    # optprob = Optimization.OptimizationProblem(optf, β)

    res1 = Optimization.solve(
        optprob,
        ADAM(params.ADAM_learning_rate),
        callback = callback,
        maxiters = params.niter_ADAM,
        verbose = true,
    )
    # @info "Training loss after $(length(losses)) iterations: $(losses[end])"

    if params.niter_LBFGS > 0
        @info "Start optimization with LBFGS"
        optprob2 = Optimization.OptimizationProblem(optf, res1.u)
        res2 = Optimization.solve(
            optprob2,
            Optim.BFGS(;
                initial_stepnorm = 0.01,
                # linesearch = LineSearches.BackTracking(iterations = 10)
                linesearch = LineSearches.HagerZhang(),
            ),
            callback = callback,
            maxiters = params.niter_LBFGS,
            allow_f_increases = true,
            successive_f_tol = 10,
            # g_tol = NaN # This disables stop criteria!
            g_abstol = 1e-12, # Toletance in the norm of the gradient
        )
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
    fit_times = collect(range(params.tmin, params.tmax, length = 1000))
    fit_directions = predict(β_trained, params, fit_times, U, st)
    fit_rotations = reduce(hcat, (t -> predict_L(t, U, β_trained.θ, st)).(fit_times))

    # Recover final balance between different terms involved in the loss function to assess hyperparameter selection.
    _, loss_dict = f_loss(β_trained)
    if params.verbose
        println("Final training loss after $(length(losses)) iterations: $(losses[end])")
        pretty_table(loss_dict, sortkeys = true, header = ["Loss term", "Value"])
    end

    return Results(
        params = params,
        θ = θ_trained,
        u0 = u0_trained,
        U = U,
        st = st,
        fit_times = fit_times,
        fit_directions = fit_directions,
        fit_rotations = fit_rotations,
        losses = losses,
        losses_dict = loss_dict,
    )
end
