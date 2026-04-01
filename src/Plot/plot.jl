export plot_sphere, plot_L

"""
Plot data points and the fitted path on a globe using an orthographic projection,
matching the visual style of Cartopy's globe view.
`central_latitude` and `central_longitude` control the viewing centre.
"""
function plot_sphere(
    data::AbstractData,
    results::AbstractResult,
    central_latitude::Union{Float64,Nothing} = nothing,
    central_longitude::Union{Float64,Nothing} = nothing;
    saveas::Union{String,Nothing} = nothing,
    title::String = "",
)
    lat_c = isnothing(central_latitude) ? 0.0 : central_latitude
    lon_c = isnothing(central_longitude) ? 0.0 : central_longitude
    φ_c = deg2rad(lat_c)
    λ_c = deg2rad(lon_c)

    # Orthographic projection: returns (x, y, cos_c) where cos_c > 0 means visible
    function ortho(lat_deg, lon_deg)
        φ = deg2rad(lat_deg)
        λ = deg2rad(lon_deg)
        cos_c = sin(φ_c) * sin(φ) + cos(φ_c) * cos(φ) * cos(λ - λ_c)
        x = cos(φ) * sin(λ - λ_c)
        y = cos(φ_c) * sin(φ) - sin(φ_c) * cos(φ) * cos(λ - λ_c)
        return x, y, cos_c
    end

    # Convert Cartesian directions to lat/lon
    X_sph = cart2sph(data.directions; radians = false)
    lats_d = X_sph[1, :]
    lons_d = X_sph[2, :]

    X_fit_sph = cart2sph(results.fit_directions; radians = false)
    lats_f = X_fit_sph[1, :]
    lons_f = X_fit_sph[2, :]

    # Project data points
    proj_d   = [ortho(lats_d[i], lons_d[i]) for i in eachindex(lats_d)]
    xd       = [q[1] for q in proj_d]
    yd       = [q[2] for q in proj_d]
    visible_d = [q[3] > 0 for q in proj_d]

    # Project fit path — set invisible segments to NaN to break the line
    proj_f = [ortho(lats_f[i], lons_f[i]) for i in eachindex(lats_f)]
    xf     = Float64[q[1] for q in proj_f]
    yf     = Float64[q[2] for q in proj_f]
    vf     = [q[3] > 0 for q in proj_f]
    xf[.!vf] .= NaN
    yf[.!vf] .= NaN

    # --- Build the plot ---
    θs = range(0, 2π; length = 361)

    # Globe background (filled circle)
    globe_shape = Shape(cos.(θs), sin.(θs))
    p = plot(
        globe_shape;
        fillcolor      = :aliceblue,
        linecolor      = :black,
        linewidth      = 1.5,
        label          = "",
        aspect_ratio   = 1,
        axis           = false,
        grid           = false,
        title          = title,
        xlims          = (-1.2, 1.2),
        ylims          = (-1.2, 1.2),
        size           = (600, 600),
    )

    # Graticule lines every 30°
    for lat in -60:30:60
        lons = range(-180, 180; length = 361)
        qs = [ortho(lat, lon) for lon in lons]
        x_ = Float64[q[1] for q in qs];  y_ = Float64[q[2] for q in qs]
        x_[[q[3] ≤ 0 for q in qs]] .= NaN
        y_[[q[3] ≤ 0 for q in qs]] .= NaN
        lw = lat == 0 ? 0.8 : 0.4
        plot!(p, x_, y_; color = :gray60, linewidth = lw, label = "")
    end
    for lon in -180:30:150
        lats = range(-89, 89; length = 180)
        qs = [ortho(lat, lon) for lat in lats]
        x_ = Float64[q[1] for q in qs];  y_ = Float64[q[2] for q in qs]
        x_[[q[3] ≤ 0 for q in qs]] .= NaN
        y_[[q[3] ≤ 0 for q in qs]] .= NaN
        plot!(p, x_, y_; color = :gray60, linewidth = 0.4, label = "")
    end

    # Data points (only visible hemisphere)
    vis_idx = findall(visible_d)
    if !isempty(vis_idx)
        scatter!(p, xd[vis_idx], yd[vis_idx];
            marker_z          = data.times[vis_idx],
            color             = :viridis,
            markersize        = 6,
            markerstrokewidth = 0.5,
            markerstrokecolor = :black,
            alpha             = 0.6,
            label             = "Observations",
            colorbar_title    = "Age (Ma)",
        )
    end

    # Fit path — plotted after points so it sits on top
    plot!(p, xf, yf; color = :black, linewidth = 3.5, label = "Estimated APWP",
        legend = :topleft)

    # Redraw globe boundary on top so it clips cleanly
    plot!(p, cos.(θs), sin.(θs); color = :black, linewidth = 1.5, label = "")

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
