using Shapefile

const _COASTLINE_CACHE = joinpath(first(Base.DEPOT_PATH), "scratchspaces", "SphereUDE", "ne_110m_coastline")

"""
Download the Natural Earth 110 m coastline shapefile on first use and cache it
in the Julia depot's scratch space. Returns the path to the `.shp` file.
"""
function _get_coastline_shp()
    mkpath(_COASTLINE_CACHE)
    shp_path = joinpath(_COASTLINE_CACHE, "ne_110m_coastline.shp")
    if !isfile(shp_path)
        @info "[SphereUDE] Downloading Natural Earth 110m coastline (one-time)..."
        base_url = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/110m_physical/ne_110m_coastline"
        for ext in (".shp", ".dbf", ".shx")
            Base.download(base_url * ext, joinpath(_COASTLINE_CACHE, "ne_110m_coastline" * ext))
        end
    end
    return shp_path
end

"""
Return a vector of index ranges, one per part of a Shapefile `Polyline` geometry.
Shapefile parts are stored as 0-based indices; we convert to 1-based Julia ranges.
"""
function _part_ranges(geom)
    starts = Int.(geom.parts) .+ 1
    ends   = [starts[2:end] .- 1; length(geom.points)]
    return [s:e for (s, e) in zip(starts, ends)]
end

"""
Overlay Natural Earth 110 m coastlines onto an existing globe plot `p` using the
orthographic projection `ortho`. Segments on the back hemisphere are masked with
NaN so they are not drawn.
"""
function _plot_coastlines!(p, ortho; color = :gray40, linewidth = 0.5)
    shp_path = _get_coastline_shp()
    for geom in Shapefile.shapes(Shapefile.Table(shp_path))
        for rng in _part_ranges(geom)
            pts       = geom.points[rng]
            projected = [ortho(pt.y, pt.x) for pt in pts]
            x_        = Float64[q[1] for q in projected]
            y_        = Float64[q[2] for q in projected]
            back      = [q[3] ≤ 0 for q in projected]
            x_[back] .= NaN
            y_[back] .= NaN
            plot!(p, x_, y_; color = color, linewidth = linewidth, label = "")
        end
    end
    return p
end
