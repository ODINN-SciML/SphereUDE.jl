export plot_sphere, plot_L, plot_cv

"""
Build an orthographic projection function centred at
(`central_latitude`, `central_longitude`). Returns `(lat_deg, lon_deg) ->
(x, y, cos_c)`, where `cos_c > 0` means the point is on the visible
hemisphere.
"""
function _make_ortho(central_latitude::Union{Float64,Nothing}, central_longitude::Union{Float64,Nothing}, data::Union{AbstractData,Nothing} = nothing)
    if (isnothing(central_latitude) || isnothing(central_longitude)) && !isnothing(data)
        mean_dir = mean(data.directions, dims = 2)[:, 1]
        mean_dir ./= norm(mean_dir)
        mean_sph = cart2sph(reshape(mean_dir, 3, 1); radians = false)
        lat_c = isnothing(central_latitude)  ? mean_sph[1, 1] : central_latitude
        lon_c = isnothing(central_longitude) ? mean_sph[2, 1] : central_longitude
    else
        lat_c = isnothing(central_latitude)  ? 0.0 : central_latitude
        lon_c = isnothing(central_longitude) ? 0.0 : central_longitude
    end
    φ_c = deg2rad(lat_c)
    λ_c = deg2rad(lon_c)
    return (lat_deg, lon_deg) -> begin
        φ = deg2rad(lat_deg)
        λ = deg2rad(lon_deg)
        cos_c = sin(φ_c) * sin(φ) + cos(φ_c) * cos(φ) * cos(λ - λ_c)
        x = cos(φ) * sin(λ - λ_c)
        y = cos(φ_c) * sin(φ) - sin(φ_c) * cos(φ) * cos(λ - λ_c)
        return x, y, cos_c
    end
end

"""
Draw the globe background and graticule (lines of latitude/longitude every
30°) under the given orthographic projection `ortho`.
"""
function _plot_globe(ortho; title::String = "", show_coastlines::Bool = false)
    θs = range(0, 2π; length = 361)

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

    if show_coastlines
        _plot_coastlines!(p, ortho)
    end

    return p
end

"""
Scatter the (visible hemisphere of the) data points of `data` onto `p`,
colored by age.
"""
function _plot_data_points!(p, data::AbstractData, ortho)
    X_sph = cart2sph(data.directions; radians = false)
    lats_d = X_sph[1, :]
    lons_d = X_sph[2, :]

    proj_d    = [ortho(lats_d[i], lons_d[i]) for i in eachindex(lats_d)]
    xd        = [q[1] for q in proj_d]
    yd        = [q[2] for q in proj_d]
    visible_d = [q[3] > 0 for q in proj_d]

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
    return p
end

"""
Scatter the resampled directions of `datasets` onto `p` as a faint cloud
around each original observation, giving a visual sense of the noise model
each point was resampled from.
"""
function _plot_resampled_points!(p, datasets::AbstractVector{<:AbstractData}, ortho)
    for (i, ds) in enumerate(datasets)
        X_sph = cart2sph(ds.directions; radians = false)
        lats = X_sph[1, :]
        lons = X_sph[2, :]

        proj    = [ortho(lats[j], lons[j]) for j in eachindex(lats)]
        x_      = [q[1] for q in proj]
        y_      = [q[2] for q in proj]
        vis_idx = findall(q -> q[3] > 0, proj)

        isempty(vis_idx) && continue
        scatter!(p, x_[vis_idx], y_[vis_idx];
            color             = :gray50,
            markersize        = 2,
            markerstrokewidth = 0,
            alpha             = 0.2,
            label             = i == 1 ? "Resampled points" : "",
        )
    end
    return p
end

"""
Plot the fitted path of `results` onto `p` (invisible-hemisphere segments are
broken with `NaN`).
"""
function _plot_fit_path!(
    p,
    results::Results,
    ortho;
    color = :black,
    linewidth = 3.5,
    alpha = 1.0,
    label = "Estimated APWP",
)
    X_fit_sph = cart2sph(results.fit_directions; radians = false)
    lats_f = X_fit_sph[1, :]
    lons_f = X_fit_sph[2, :]

    proj_f = [ortho(lats_f[i], lons_f[i]) for i in eachindex(lats_f)]
    xf     = Float64[q[1] for q in proj_f]
    yf     = Float64[q[2] for q in proj_f]
    vf     = [q[3] > 0 for q in proj_f]
    xf[.!vf] .= NaN
    yf[.!vf] .= NaN

    plot!(p, xf, yf; color = color, linewidth = linewidth, alpha = alpha, label = label)
    return p
end

"""
Plot data points and the fitted path on a globe using an orthographic projection,
matching the visual style of Cartopy's globe view.
`central_latitude` and `central_longitude` control the viewing centre.
"""
function plot_sphere(
    data::AbstractData,
    results::Union{Results,Nothing} = nothing,
    central_latitude::Union{Float64,Nothing} = nothing,
    central_longitude::Union{Float64,Nothing} = nothing;
    show_coastlines::Bool = false,
    saveas::Union{String,Nothing} = nothing,
    title::String = "",
)

    ortho = _make_ortho(central_latitude, central_longitude, data)

    p = _plot_globe(ortho; title = title, show_coastlines = show_coastlines)
    _plot_data_points!(p, data, ortho)
    if !isnothing(results)
        _plot_fit_path!(p, results, ortho)
    end
    plot!(p; legend = :topleft)

    # Redraw globe boundary on top so it clips cleanly
    θs = range(0, 2π; length = 361)
    plot!(p, cos.(θs), sin.(θs); color = :black, linewidth = 1.5, label = "")

    if !isnothing(saveas)
        savefig(p, saveas)
    end
    return p
end

"""
Plot every trajectory in an `EnsambleResult` overlaid on a globe, e.g. to
visualize the spread of resampled fits used for uncertainty quantification
(see [`sample_uq`](@ref)). Pass `main_result` (e.g. the original, non-resampled
fit) to additionally highlight a single reference trajectory on top of the
ensemble. The resampled directions in `ensemble.datasets` are shown as a
faint cloud around each original observation unless
`show_resampled_points = false`.
"""
function plot_sphere(
    data::AbstractData,
    ensemble::EnsambleResult,
    central_latitude::Union{Float64,Nothing} = nothing,
    central_longitude::Union{Float64,Nothing} = nothing;
    main_result::Union{Results,Nothing} = nothing,
    show_resampled_points::Bool = true,
    show_coastlines::Bool = false,
    saveas::Union{String,Nothing} = nothing,
    title::String = "",
)
    ortho = _make_ortho(central_latitude, central_longitude, data)

    p = _plot_globe(ortho; title = title, show_coastlines = show_coastlines)
    if show_resampled_points
        _plot_resampled_points!(p, ensemble.datasets, ortho)
    end
    _plot_data_points!(p, data, ortho)

    for (i, result_i) in enumerate(ensemble.results)
        _plot_fit_path!(p, result_i, ortho;
            color     = :steelblue,
            linewidth = 1.0,
            alpha     = 0.35,
            label     = i == 1 ? "UQ samples" : "",
        )
    end

    if !isnothing(main_result)
        _plot_fit_path!(p, main_result, ortho)
    end
    plot!(p; legend = :topleft)

    # Redraw globe boundary on top so it clips cleanly
    θs = range(0, 2π; length = 361)
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
    Ls = reduce(hcat, (t -> results.regressor(t, results.θ)).(times_smooth))
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


"""
Plot the per-fold validation scores of a [`CVResult`](@ref) (see
[`train_cv`](@ref)) as a function of λ, with λ on a log-scaled x-axis: each
candidate λ's `k_folds` scores are scattered, with the mean and median score
overlaid as lines, and the selected `best_λ` marked with a vertical line.

When `show_all_results = true` and `data` is provided, a second globe plot is
also returned, overlaying the fitted paths for every non-`nothing` entry in
`cv.all_results` (see `refit_all` in [`train_cv`](@ref)), colored from blue
(low λ) to red (high λ), with the best-λ path drawn on top in bold black.
The function then returns `(p_cv, p_sphere)`; otherwise it returns `p_cv` only.
"""
function plot_cv(
    cv::CVResult,
    data::Union{AbstractData,Nothing} = nothing;
    show_all_results::Bool = false,
    central_latitude::Union{Float64,Nothing} = nothing,
    central_longitude::Union{Float64,Nothing} = nothing,
    show_coastlines::Bool = false,
    saveas::Union{String,Nothing} = nothing,
    title::String = "Cross-validation",
)
    λs = cv.λ_grid

    xs = vcat([fill(λs[k], length(cv.scores[k])) for k in eachindex(λs)]...)
    ys = vcat(cv.scores...)

    p_cv = plot(
        xscale = :log10,
        xlabel = "λ",
        ylabel = "Validation loss",
        title = title,
        legend = :topright,
    )

    scatter!(p_cv, xs, ys; color = :gray60, markersize = 4, markerstrokewidth = 0, alpha = 0.5, label = "Per-fold score")
    plot!(p_cv, λs, mean.(cv.scores); color = :blue, linewidth = 2, marker = :circle, label = "Mean")
    plot!(p_cv, λs, median.(cv.scores); color = :red, linewidth = 2, marker = :diamond, label = "Median")
    vline!(p_cv, [cv.best_λ]; color = :black, linestyle = :dash, label = "Best λ")

    if !isnothing(saveas)
        savefig(p_cv, saveas)
    end

    if !show_all_results
        return p_cv
    end

    @assert !isnothing(data) "plot_cv: `data` must be provided when `show_all_results = true`"
    valid_idxs = findall(!isnothing, cv.all_results)
    @assert !isempty(valid_idxs) "plot_cv: `cv.all_results` is all nothing — run train_cv with `refit_all = true`"

    ortho = _make_ortho(central_latitude, central_longitude, data)
    p_sphere = _plot_globe(ortho; title = "$(title) — all λ fits", show_coastlines = show_coastlines)
    _plot_data_points!(p_sphere, data, ortho)

    n_valid = length(valid_idxs)
    palette = cgrad(:RdBu, n_valid; rev = true)
    best_idx = findfirst(==(cv.best_λ), λs)

    for (i, k) in enumerate(valid_idxs)
        is_best = k == best_idx
        is_best && continue  # draw best on top after the loop
        _plot_fit_path!(p_sphere, cv.all_results[k], ortho;
            color     = palette[i],
            linewidth = 1.5,
            alpha     = 0.6,
            label     = "",
        )
    end

    if !isnothing(best_idx) && !isnothing(cv.all_results[best_idx])
        _plot_fit_path!(p_sphere, cv.all_results[best_idx], ortho;
            color     = :black,
            linewidth = 3.5,
            alpha     = 1.0,
            label     = "Best λ = $(cv.best_λ)",
        )
    end

    plot!(p_sphere; legend = :topleft)
    θs = range(0, 2π; length = 361)
    plot!(p_sphere, cos.(θs), sin.(θs); color = :black, linewidth = 1.5, label = "")

    if !isnothing(saveas)
        base, ext = splitext(saveas)
        savefig(p_sphere, base * "_all_results" * ext)
    end

    return p_cv, p_sphere
end
