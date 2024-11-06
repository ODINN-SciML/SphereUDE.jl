# export mpl_colors, mpl_colormap, sns, ccrs, feature, cmap
export plot_sphere, plot_L

"""
Generate matplotlib grid template for figure
"""
# function generate_sphere_figure(grid::Tuple{Number, Number}, 
#                                 size::Tuple{Number, Number})
#                                 # central_latitude::Number, 
#                                 # central_longitude::Number)
#     fig, axes = plt.subplots(nrows=grid[1], ncols=grid[2], figsize=size)
#     return fig, axes
# end

"""
Create spherical figure with observations (points), latent variable, and fitter path.
"""
function plot_sphere(# ax::PyCall.PyObject, 
                     data::AbstractData,
                     results::AbstractResult,
                    #  X_points::Matrix{Float64}, 
                    #  X_path::Matrix{Float64}, 
                     central_latitude::Float64, 
                     central_longitude::Float64;
                     saveas::Union{String, Nothing},
                     title::String, 
                     matplotlib_rcParams::Union{Dict, Nothing} = nothing)

    # cmap = mpl_colormap.get_cmap("viridis")

    plt.figure(figsize=(10,10))
    ax = plt.axes(projection=ccrs.Orthographic(central_latitude=central_latitude, central_longitude=central_longitude))
    
    # Set default plot parameters. 
    # See https://matplotlib.org/stable/users/explain/customizing.html for customizable optionsz
    if !isnothing(matplotlib_rcParams)
        for (key, value) in matplotlib_rcParams
            @warn "Setting Matplotlib parameters with rcParams currently not working. See following GitHub issue: https://github.com/JuliaPy/PyPlot.jl/issues/525"
            mpl_base.rcParams[key] = value
        end
    end

    ax.coastlines()
    ax.gridlines()
    ax.set_global()

    X_true_points = cart2sph(data.directions, radians=false)
    # X_true_path = cart2sph(X_path, radians=false)
    X_fit_path = cart2sph(results.fit_directions, radians=false)

    # Plots in Python follow the long, lat ordering 

    sns.scatterplot(ax=ax, x = X_true_points[2,:], y=X_true_points[1, :], 
                    hue = data.times, s=150,
                    palette="viridis",
                    transform = ccrs.PlateCarree());
    
    for i in 1:(length(results.fit_times)-1)
        plt.plot([X_fit_path[2,i], X_fit_path[2,i+1]], 
                 [X_fit_path[1,i], X_fit_path[1,i+1]],
                  linewidth=2, color="black",#cmap(norm(results.fit_times[i])),
                  transform = ccrs.Geodetic())
    end
    # plt.title(title, fontsize=20)
    if !isnothing(saveas)
        plt.savefig(saveas, format="pdf")
    end

    return nothing
end


"""
Plot fitted rotation
"""
function plot_L(data::AbstractData, 
                results::AbstractResult;
                saveas::Union{String, Nothing},
                title::String)

    fig, ax = plt.subplots(figsize=(10,5))

    times_smooth = collect(LinRange(results.fit_times[begin], results.fit_times[end], 1000))
    Ls = reduce(hcat, (t -> results.U([t], results.θ, results.st)[1]).(times_smooth))

    angular_velocity = mapslices(x -> norm(x), Ls, dims=1)[1,:]

    ax.plot(times_smooth, angular_velocity, label="Estimated")

    if !isnothing(data.L)
        Ls_true = reduce(hcat, data.L.(times_smooth))
        angular_velocity_true = mapslices(x -> norm(x), Ls_true, dims=1)[1,:]
        ax.plot(times_smooth, angular_velocity_true, label="Reference")
    end
    
    # plt.title("")
    plt.xlabel("Time")
    plt.ylabel("Angular Velocity")
    plt.legend()
    # plt.title(title)

    if !isnothing(saveas)
        plt.savefig(saveas, format="pdf")
    end

    return nothing
end


