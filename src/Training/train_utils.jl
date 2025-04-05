export CallbackOptimizationSet, callback_print


"""
    CallbackOptimizationSet(θ, l; callbacks)

Helper to combine callbacks for Optimization function. This executes the action of each callback.  
(equivalent to CallbackSet for DifferentialEquations.jl)
"""
function CallbackOptimizationSet(θ, l; callbacks)
    for cb in callbacks
        _ = cb(θ, l)
    end
    return false
end

"""
    callback_print(p, l, losses)
"""
function callback_print(p, l, params, losses, f_loss)
    push!(losses, l)
    if !params.verbose
        return nothing
    end
    step = params.verbose_step
    if length(losses) % step == 0 || length(losses) == 1
        _, l_dict = f_loss(p.u)
        if length(losses) < step + 1
            improvement = nothing
        else
            improvement = (losses[end]-losses[end-step]) / losses[end-step]
        end
        @assert losses[end] ≈ sum(values(l_dict))
        printProgressLoss(length(losses), (params.niter_ADAM+params.niter_LBFGS), losses[end], l_dict["Empirical"], (sum(values(l_dict))-l_dict["Empirical"]), improvement)
    end
end

"""
    printProgressLoss
"""
function printProgressLoss(iter, total_iters, loss, loss_emp, loss_reg, improvement)
    print("Iteration: [")
    print(@sprintf("%5i", iter))
    print(" / ")
    print(@sprintf("%5i", total_iters))
    print("]     ")
    print("Loss:  ")
    print(@sprintf("%9.4e", loss))
    print(" = ")
    print(@sprintf("%9.2e", loss_emp))
    print(" + ")
    print(@sprintf("%9.2e", loss_reg))
    if !isnothing(improvement)
        print("     ")
        print("Improvement: ")
        if improvement < 0
            printstyled(@sprintf("%.2f %%", 100*improvement); color=:green)
        else
            printstyled(@sprintf("%.2f %%", 100 * improvement); color = :red)
        end
    end
    println("")
end
