export convert2dict

function convert2dict(data::SphereData, results::Results)
    _dict = Dict()
    _dict["times"] = data.times
    _dict["directions"] = data.directions
    _dict["kappas"] = data.kappas
    _dict["u0"] = results.u0
    _dict["fit_times"] = results.fit_times
    _dict["fit_directions"] = results.fit_directions
    _dict["fit_rotations"] = results.fit_rotations
    _dict["losses"] = results.losses

    return _dict
end