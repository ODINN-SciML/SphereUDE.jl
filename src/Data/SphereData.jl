export SphereData, AbstractData

abstract type AbstractData end

"""
Spherical data information.
"""
mutable struct SphereData{F<:AbstractFloat,IN<:Union{Integer,Nothing}} <: AbstractData
    times::Vector{F}
    times_young::Union{Vector{F},Nothing}
    times_old::Union{Vector{F},Nothing}
    directions::Matrix{F}
    kappas::Union{Vector{F},Nothing}
    L::Union{Function,Nothing}
    repeat_times::Bool
    times_unique::Union{Vector{F},Nothing}
    times_unique_inverse::Union{Vector{IN},Nothing}
end

function SphereData(;
    times::Union{Vector{F},Nothing} = nothing,
    times_young::Union{Vector{F},Nothing} = nothing,
    times_old::Union{Vector{F},Nothing} = nothing,
    directions::Matrix{F},
    kappas::Union{Vector{F},Nothing},
    L::Union{Function,Nothing} = nothing,
) where {F<:AbstractFloat}

    if isnothing(times)
        @assert !isnothing(times_young) && !isnothing(times_old) "Provide either `times` or both `times_young` and `times_old`."
        times = (times_young .+ times_old) ./ 2
    end

    # Sort all variables in time
    idx = sortperm(times)
    times = times[idx]
    directions = directions[:, idx]
    kappas = isnothing(kappas) ? nothing : kappas[idx]
    times_young = isnothing(times_young) ? nothing : times_young[idx]
    times_old = isnothing(times_old) ? nothing : times_old[idx]

    # Determine if data times are unique or not
    if length(unique(times)) < length(times)
        repeat_times = true
        times_unique, times_unique_inverse = MakeVectorUnique(times)
        int_type = typeof(first(times_unique_inverse))
    else
        repeat_times = false
        times_unique = nothing
        times_unique_inverse = nothing
        int_type = Nothing
    end

    ft_type = typeof(first(times))
    return SphereData{ft_type,int_type}(
        times,
        times_young,
        times_old,
        directions,
        kappas,
        L,
        repeat_times,
        times_unique,
        times_unique_inverse,
    )
end
