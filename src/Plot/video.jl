export plot_sphere_video

"""
    plot_sphere_video(data, results, central_latitude, central_longitude; kwargs...)

Animate the APWP solution as time advances from `results.fit_times[1]` to
`results.fit_times[end]`. Each frame draws the fitted path up to the current
time and all data points whose age falls at or before that time.

# Keywords
- `n_frames::Int = 100`: total number of animation frames
- `fps::Int = 20`: frames per second of the output video
- `follow_path::Bool = false`: re-center the globe on the current path tip each
  frame (overrides `central_latitude`/`central_longitude` when true)
- `show_coastlines::Bool = false`: overlay Natural Earth 110 m coastlines
- `saveas::Union{String,Nothing} = nothing`: output path; extension determines
  format — `.gif` → GIF, anything else → MP4
- `title::String = ""`: base title; current time is appended automatically
"""
function plot_sphere_video(
    data::AbstractData,
    results::Results,
    central_latitude::Union{Float64,Nothing} = nothing,
    central_longitude::Union{Float64,Nothing} = nothing;
    n_frames::Int = 100,
    fps::Int = 20,
    follow_path::Bool = false,
    show_coastlines::Bool = false,
    saveas::Union{String,Nothing} = nothing,
    title::String = "",
)
    t_start = results.fit_times[1]
    t_end   = results.fit_times[end]
    frame_times = LinRange(t_start, t_end, n_frames)

    # Precompute spherical coords of fit path and data (reused every frame)
    fit_sph   = cart2sph(results.fit_directions; radians = false)
    fit_lats  = fit_sph[1, :]
    fit_lons  = fit_sph[2, :]

    data_sph  = cart2sph(data.directions; radians = false)
    data_lats = data_sph[1, :]
    data_lons = data_sph[2, :]

    # Fixed projection — computed once when not following the path
    ortho_fixed = follow_path ? nothing : _make_ortho(central_latitude, central_longitude, data)

    anim = @animate for t in frame_times
        path_mask = results.fit_times .<= t

        # Determine projection center for this frame
        if follow_path && any(path_mask)
            tip_idx  = findlast(path_mask)
            tip_sph  = cart2sph(results.fit_directions[:, tip_idx:tip_idx]; radians = false)
            ortho    = _make_ortho(tip_sph[1, 1], tip_sph[2, 1])
        else
            ortho = ortho_fixed
        end

        frame_title = isempty(title) ?
            "t = $(round(t; digits = 1)) Ma" :
            "$(title) — t = $(round(t; digits = 1)) Ma"

        p = _plot_globe(ortho; title = frame_title, show_coastlines = show_coastlines)

        # ── Data points with age ≤ t ─────────────────────────────────────────
        data_mask = data.times .<= t
        vis_idx   = findall(data_mask)
        if !isempty(vis_idx)
            proj_d  = [ortho(data_lats[i], data_lons[i]) for i in vis_idx]
            xd      = Float64[q[1] for q in proj_d]
            yd      = Float64[q[2] for q in proj_d]
            fwd     = findall(q -> q[3] > 0, proj_d)
            if !isempty(fwd)
                scatter!(p, xd[fwd], yd[fwd];
                    marker_z          = data.times[vis_idx[fwd]],
                    color             = :viridis,
                    clims             = (t_start, t_end),
                    markersize        = 6,
                    markerstrokewidth = 0.5,
                    markerstrokecolor = :black,
                    alpha             = 0.6,
                    label             = "Observations",
                    colorbar_title    = "Age (Ma)",
                )
            end
        end

        # ── Fit path up to t ─────────────────────────────────────────────────
        if any(path_mask)
            lf     = fit_lats[path_mask]
            lonf   = fit_lons[path_mask]
            proj_f = [ortho(lf[i], lonf[i]) for i in eachindex(lf)]
            xf     = Float64[q[1] for q in proj_f]
            yf     = Float64[q[2] for q in proj_f]
            vf     = [q[3] > 0 for q in proj_f]
            xf[.!vf] .= NaN
            yf[.!vf] .= NaN
            plot!(p, xf, yf; color = :black, linewidth = 3.5, label = "APWP")

            # Highlight the current tip
            tip_proj = proj_f[end]
            if tip_proj[3] > 0
                scatter!(p, [tip_proj[1]], [tip_proj[2]];
                    color             = :red,
                    markersize        = 9,
                    markerstrokewidth = 1.5,
                    markerstrokecolor = :white,
                    label             = "",
                )
            end
        end

        plot!(p; legend = :topleft)

        # Redraw globe boundary on top so it clips cleanly
        θs = range(0, 2π; length = 361)
        plot!(p, cos.(θs), sin.(θs); color = :black, linewidth = 1.5, label = "")
    end

    if !isnothing(saveas)
        if endswith(saveas, ".gif")
            gif(anim, saveas; fps = fps)
        else
            mp4(anim, saveas; fps = fps)
        end
    end

    return anim
end
