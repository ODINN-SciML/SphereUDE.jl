export update_params

"""
    update_params(params::SphereParameters; kwargs...)

Return a copy of `params` with any subset of fields overridden by the provided
keyword arguments. All fields not listed in `kwargs` are taken from `params`.

Useful for creating a variant of an existing parameter set without repeating
all fields — e.g. loosening tolerances for a UQ run:

    params_uq = update_params(params_best; reltol = 1e-6, abstol = 1e-6)
"""
function update_params(
    params::SphereParameters;
    tmin = params.tmin,
    tmax = params.tmax,
    u0 = params.u0,
    ωmax = params.ωmax,
    reg = params.reg,
    train_initial_condition = params.train_initial_condition,
    multiple_shooting = params.multiple_shooting,
    weighted = params.weighted,
    niter_ADAM = params.niter_ADAM,
    ADAM_learning_rate = params.ADAM_learning_rate,
    niter_LBFGS = params.niter_LBFGS,
    stop_tol = params.stop_tol,
    reltol = params.reltol,
    abstol = params.abstol,
    quadrature = params.quadrature,
    solver = params.solver,
    adtype = params.adtype,
    sensealg = params.sensealg,
    out_of_place = params.out_of_place,
    pretrain = params.pretrain,
    hyperparameter_balance = params.hyperparameter_balance,
    verbose = params.verbose,
    verbose_step = params.verbose_step,
)
    return SphereParameters(
        tmin = tmin,
        tmax = tmax,
        u0 = u0,
        ωmax = ωmax,
        reg = reg,
        train_initial_condition = train_initial_condition,
        multiple_shooting = multiple_shooting,
        weighted = weighted,
        niter_ADAM = niter_ADAM,
        ADAM_learning_rate = ADAM_learning_rate,
        niter_LBFGS = niter_LBFGS,
        stop_tol = stop_tol,
        reltol = reltol,
        abstol = abstol,
        quadrature = quadrature,
        solver = solver,
        adtype = adtype,
        sensealg = sensealg,
        out_of_place = out_of_place,
        pretrain = pretrain,
        hyperparameter_balance = hyperparameter_balance,
        verbose = verbose,
        verbose_step = verbose_step,
    )
end
