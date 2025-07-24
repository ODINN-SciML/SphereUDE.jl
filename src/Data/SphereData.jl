export SphereData, AbstractData

abstract type AbstractData end

"""
Spherical data information.
"""
mutable struct SphereData{F<:AbstractFloat,IN<:Union{Integer,Nothing}} <: AbstractData
    times::Vector{F}
    directions::Matrix{F}
    kappas::Union{Vector{F},Nothing}
    L::Union{Function,Nothing}
    repeat_times::Bool
    times_unique::Union{Vector{F},Nothing}
    times_unique_inverse::Union{Vector{IN},Nothing}
end

function SphereData(;
    times::Vector{F},
    directions::Matrix{F},
    kappas::Union{Vector{F},Nothing},
    L::Union{Function,Nothing} = nothing,
) where {F<:AbstractFloat}

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
        directions,
        kappas,
        L,
        repeat_times,
        times_unique,
        times_unique_inverse,
    )
end
