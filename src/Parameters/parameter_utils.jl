export update_params

"""
    _replace_field(obj, field, value)

Returns a copy of `obj` with `field` set to `value`, leaving all other fields
unchanged. Relies on the default positional constructor of `typeof(obj)`.
"""
function _replace_field(obj::T, field::Symbol, value) where {T}
    return T((f === field ? value : getfield(obj, f) for f in fieldnames(T))...)
end

"""
    update_params(params::SphereParameters; kwargs...)

Return a copy of `params` with any subset of fields overridden by the provided
keyword arguments.

    params_uq = update_params(params_best; reltol = 1e-6, abstol = 1e-6)
"""
function update_params(params::SphereParameters; kwargs...)
    valid = fieldnames(SphereParameters)
    for field in keys(kwargs)
        @assert field in valid "update_params: unknown field `$(field)`. Valid fields are: $(valid)"
    end
    result = params
    for (field, value) in kwargs
        result = _replace_field(result, field, value)
    end
    return result
end
