export convert2dict, dict2results

function convert2dict(data::SphereData, results::Results)
    _dict = Dict()
    _dict["times"] = data.times
    _dict["directions"] = data.directions
    _dict["kappas"] = data.kappas
    _dict["u0"] = results.u0
    _dict["θ"] = results.θ
    _dict["regressor"] = results.regressor
    _dict["fit_times"] = results.fit_times
    _dict["fit_directions"] = results.fit_directions
    _dict["fit_rotations"] = results.fit_rotations
    _dict["losses"] = results.losses

    return _dict
end

"""
    dict2results(d, params)

Reconstruct a [`Results`](@ref) from a dict previously created by
[`convert2dict`](@ref). `params` must be supplied separately since it is not
stored in the dict.
"""
function dict2results(d::Dict, params::SphereParameters)
    return Results(
        params = params,
        θ = d["θ"],
        u0 = d["u0"],
        regressor = d["regressor"],
        fit_times = d["fit_times"],
        fit_directions = d["fit_directions"],
        fit_rotations = d["fit_rotations"],
        losses = d["losses"],
        losses_dict = Dict(),
    )
end

function convert2dict(data::SphereData, cv::CVResult)
    _dict = Dict()
    _dict["λ_grid"] = cv.λ_grid
    _dict["scores"] = cv.scores
    _dict["best_λ"] = cv.best_λ
    if !isnothing(cv.best_results)
        _dict["best_results"] = convert2dict(data, cv.best_results)
    end
    _dict["all_results"] = [isnothing(r) ? nothing : convert2dict(data, r) for r in cv.all_results]

    return _dict
end
