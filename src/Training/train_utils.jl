export CallbackOptimizationSet, callback_print

"""
    _replace_field(obj, field, value)

Returns a copy of `obj` with `field` set to `value`, leaving all other fields
unchanged. Relies on the default positional constructor of `typeof(obj)`.
"""
function _replace_field(obj::T, field::Symbol, value) where {T}
    return T((f === field ? value : getfield(obj, f) for f in fieldnames(T))...)
end

"""
    _print_loss_breakdown(loss_dict, n_iters)

Prints the final total loss and its breakdown by term.
"""
function _print_loss_breakdown(loss_dict, n_iters)
    total_loss = sum(values(loss_dict))
    println("Final training loss after $(n_iters) iterations: $(total_loss)")
    sorted_keys = sort(collect(keys(loss_dict)))
    pretty_table(
        hcat(sorted_keys, [loss_dict[k] for k in sorted_keys]);
        column_labels = ["Loss term", "Value"],
        style = TextTableStyle(first_line_column_label = crayon"yellow bold"),
    )
end

"""
    CallbackOptimizationSet(θ, l; callbacks)

Helper to combine callbacks for Optimization function. This executes the action of each callback.  
(equivalent to CallbackSet for DifferentialEquations.jl)
"""
function CallbackOptimizationSet(p, l; callbacks)
    cb_outputs = Bool[]
    for cb in callbacks
        _cb = cb(p, l)
        push!(cb_outputs, _cb)
    end
    return any(cb_outputs)
end

"""
    callback_print(p, l, losses)
"""
function callback_print(p, l, params, losses, f_loss)
    push!(losses, l)
    if !params.verbose
        return false
    end
    step = params.verbose_step
    if length(losses) % step == 0 || length(losses) == 1
        _l, l_dict = f_loss(p.u)
        if !isapprox(l, _l, rtol = 1e-2) & params.verbose & (length(losses) > 200)
            @warn """
            Loss function computed during epoch training (loss = $(l)) does not coincide
            with the one computed after training (loss = $(_l)). This can be cause by an
            error in the code or by randomness inside the loss function.
            """
        end
        if length(losses) < 2
            improvement = nothing
        elseif length(losses) < step + 1
            improvement = (losses[end] - losses[begin]) / losses[begin]
        else
            improvement = (losses[end] - losses[end-step]) / losses[end-step]
        end
        printProgressLoss(
            length(losses),
            (params.niter_ADAM + params.niter_LBFGS),
            losses[end],
            l_dict["Empirical"],
            (sum(values(l_dict)) - l_dict["Empirical"]),
            improvement,
        )
    end
    return false
end

"""
    callback_stop_condition(p, l, losses, tol)

Callback to determine stoping condition of optimization algorithm. `tol` is
the relative tolerance (`params.stop_tol`) below which the loss is considered
to have plateaued.
"""
function callback_stop_condition(p, l, losses, tol)
    n_window = 30
    if (length(losses) > n_window) & (length(losses) % 10 == 0)
        losses_last = losses[end-n_window+1:end]
        if (std(losses_last) / mean(losses_last) < tol) &
           (abs(losses[end] - losses[end-1]) < tol * losses[end-1])
            println("Optimization converged in $(length(losses)) epochs.")
            return true
        else
            return false
        end
    else
        return false
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
    print("  = ")
    print(@sprintf("%9.2e", loss_emp))
    print("  + ")
    print(@sprintf("%9.2e", loss_reg))
    if !isnothing(improvement)
        print("     ")
        print("Improvement: ")
        if improvement < 0
            printstyled(@sprintf("%.2f %%", 100 * improvement); color = :green)
        else
            printstyled(@sprintf("%.2f %%", 100 * improvement); color = :red)
        end
    end
    println("")
end
