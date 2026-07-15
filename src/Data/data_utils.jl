export cart2sph, sph2cart
export AbstractNoise, FisherNoise
export fisher_mean
export MakeVectorUnique
export raise_warnings
export resample_data


"""
    cart2sph(X::AbstractArray{<:Number}; radians::Bool=true)

Convert cartesian coordinates to spherical
"""
function cart2sph(X::AbstractArray{<:Number}; radians::Bool = true)
    @assert size(X)[1] == 3 "Input array must have three rows."
    Y = mapslices(x -> [asin(x[3]), angle(x[1] + x[2] * im)], X, dims = 1)
    if !radians
        Y *= 180.0 / π
    end
    return Y
end


"""
    sph2cart(X::AbstractArray{<:Number}; radians::Bool=true)

Convert spherical coordinates to cartesian
"""
function sph2cart(X::AbstractArray{<:Number}; radians::Bool = true)
    @assert size(X)[1] == 2 "Input array must have two rows corresponding to Latitude and Longitude."
    if !radians
        X *= π / 180.0
    end
    Y = mapslices(
        x -> [cos(x[1]) * cos(x[2]), cos(x[1]) * sin(x[2]), sin(x[1])],
        X,
        dims = 1,
    )
    return Y
end


"""
    fisher_mean(latitudes, longitudes; radians::Bool=true)

Return Fisher mean on the sphere
"""
function fisher_mean(latitudes, longitudes; radians::Bool = true)
    Y = sph2cart(hcat(latitudes, longitudes)', radians = radians)
    Ŷ = mean(Y, dims = 2)
    Ŷ ./= norm(Ŷ)
    return cart2sph(Ŷ, radians = radians)
end


"""
Add Fisher noise to matrix of three dimensional unit vectors

This is carried by the definition of type FisherNoise <: AbstractNoise and
extending the base definition +(,) to allow the simple syntax

X_noise = X_noiseless + FisherNoise(kappa=200.)
"""
abstract type AbstractNoise end

@kwdef struct FisherNoise{F<:AbstractFloat} <: AbstractNoise
    kappa::Union{F,Vector{F}}
end

function Base.:(+)(X::Array{F,2}, ϵ::N) where {F<:AbstractFloat,N<:AbstractNoise}
    if typeof(ϵ.kappa) <: F
        return mapslices(
            x -> rand(sampler(VonMisesFisher(x / norm(x), ϵ.kappa)), 1),
            X,
            dims = 1,
        )
    else
        @assert length(ϵ.kappa) == size(X)[2] "Signal and noise must have same dimensions."
        Y = similar(X)
        for i = 1:size(X)[2]
            x = X[:, i]
            Y[:, i] = rand(sampler(VonMisesFisher(x / norm(x), ϵ.kappa[i])), 1)
        end
        return Y
    end
end


"""
     MakeVectorUnique(v::Vector)

Tool to avoid repetition of times in the forward/reverse ODE calculation
"""
function MakeVectorUnique(v::Vector)
    @assert v == sort(v) "[SphereUDE] MakeVectorUnique requires `v` to be sorted."
    vector_unique = [v[1]]
    inverse_unique = Int64[1]
    counter = 1
    for i = 2:length(v)
        if v[i] != v[i-1]
            push!(vector_unique, v[i])
            counter += 1
        end
        push!(inverse_unique, counter)
    end
    @assert vector_unique[inverse_unique] == v
    return vector_unique, inverse_unique
end


"""
    resample_data(data::SphereData, rng)

Resample the directions of `data`, used for uncertainty quantification. Each
direction is redrawn from a Von Mises–Fisher distribution centered at the
original direction with the concentration parameter `kappa` given by
`data.kappas`, so the resampled dataset reflects the same noise model as the
original observations. Draws are made using `rng`, so seeding `rng` makes the
resampling reproducible.
"""
function resample_data(
    data::SphereData,
    rng;
    resample_times = true,
    resample_directions = true
    )

    @assert !isnothing(data.kappas) "[SphereUDE] Cannot resample data without concentration parameters (kappas)."
    @assert all(i -> norm(data.directions[:, i]) ≈ 1, axes(data.directions, 2)) "[SphereUDE] Directions must be unit vectors."

    # Resampling of times
    if resample_times
        @assert !isnothing(data.times_young) && !isnothing(data.times_old) "[SphereUDE] Cannot resample times without `times_young` and `times_old` in SphereData."
        times_resampled = data.times_young .+ rand(rng, length(data.times)) .* (data.times_old .- data.times_young)
    else
        times_resampled = data.times
    end

    # Resampling of directions
    directions_resampled = similar(data.directions)
    if resample_directions
        for i in axes(data.directions, 2)
            x = data.directions[:, i]
            directions_resampled[:, i] = rand(rng, sampler(VonMisesFisher(x, data.kappas[i])))
        end
    else
        directions_resampled = data.directions
    end

    return SphereData(
        times = times_resampled,
        times_young = data.times_young,
        times_old = data.times_old,
        directions = directions_resampled,
        kappas = data.kappas,
        L = data.L,
    )
end

"""
    raise_warnings(data::AD, params::AP)

Raise warnings.
"""
function raise_warnings(data::SphereData, params::SphereParameters)
    # if length(unique(data.times)) < length(data.times)
    #     @warn "[SphereUDE] Timeseries includes duplicated times. \n This can produce unexpected errors."
    # end
    # if !isnothing(params.reg)
    #     for reg in params.reg  
    #         if reg.diff_mode=="CS"
    #             @warn "[SphereUDE] Complex-step differentiation inside the loss function \n This just work for cases where the activation function of the neural network admits complex numbers \n Change predefined activation functions to be complex numbers."
    #         end
    #     end
    # end
    nothing
end
