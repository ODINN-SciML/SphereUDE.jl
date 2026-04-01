export plot_sphere, plot_L

"""
Create a lat/lon scatter plot with observations colored by time and the fitted path.
"""
function plot_sphere(
    data::AbstractData,
    results::AbstractResult;
    saveas::Union{String,Nothing} = nothing,
    title::String = "",
)
    X_true_points = cart2sph(data.directions, radians = false)
    X_fit_path = cart2sph(results.fit_directions, radians = false)

    lons_data = X_true_points[2, :]
    lats_data = X_true_points[1, :]
    lons_path = X_fit_path[2, :]
    lats_path = X_fit_path[1, :]

    p = scatter(
        lons_data,
        lats_data;
        marker_z = data.times,
        color = :viridis,
        markersize = 6,
        markerstrokewidth = 0,
        label = "Data",
        xlabel = "Longitude (°)",
        ylabel = "Latitude (°)",
        title = title,
        xlims = (-180, 180),
        ylims = (-90, 90),
        colorbar_title = "Time",
    )
    plot!(p, lons_path, lats_path; color = :black, linewidth = 2, label = "Fit")

    if !isnothing(saveas)
        savefig(p, saveas)
    end

    return p
end


"""
Plot fitted rotation (angular velocity over time).
"""
function plot_L(
    data::AbstractData,
    results::AbstractResult;
    saveas::Union{String,Nothing},
    title::String,
)
    times_smooth = collect(LinRange(results.fit_times[begin], results.fit_times[end], 1000))
    Ls = reduce(hcat, (t -> results.U([t], results.θ, results.st)[1]).(times_smooth))
    angular_velocity = mapslices(x -> norm(x), Ls, dims = 1)[1, :]

    p = plot(
        times_smooth,
        angular_velocity;
        label = "Estimated",
        xlabel = "Time",
        ylabel = "Angular Velocity",
        title = title,
    )

    if !isnothing(data.L)
        Ls_true = reduce(hcat, data.L.(times_smooth))
        angular_velocity_true = mapslices(x -> norm(x), Ls_true, dims = 1)[1, :]
        plot!(p, times_smooth, angular_velocity_true; label = "Reference")
    end

    if !isnothing(saveas)
        savefig(p, saveas)
    end

    return p
end
