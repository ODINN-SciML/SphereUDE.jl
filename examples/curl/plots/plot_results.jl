using Pkg; Pkg.activate(".")

using SphereUDE
using Serialization
using Plots 
using Plots.PlotMeasures
using JLD2
using LinearAlgebra

JLD2.@load "examples/curl/results/results_dict.jld2" 

data_directions_sph = cart2sph(results_dict["directions"], radians=false)
fit_directions_sph = cart2sph(results_dict["fit_directions"], radians=false)

data_lat = data_directions_sph[1,:]
fit_lat  = fit_directions_sph[1,:]

data_lon = data_directions_sph[2,:]
fit_lon  = fit_directions_sph[2,:]

blue_belize = RGBA(41/255, 128/255, 185/255, 1)
blue_midnigth = RGBA(44/255, 62/255, 80/255, 1)
purple_wisteria = RGBA(142/255, 68/255, 173/255, 1)
red_pomegrade = RGBA(192/255, 57/255, 43/255, 1)
orange_carrot = RGBA(230/255, 126/255, 34/255, 1)
green_nephritis = RGBA(39/255, 174/255, 96/255, 1)
green_sea = RGBA(22/255, 160/255, 133/255, 1)
# colors 
lat_color_scatter = blue_belize
lat_color_line = purple_wisteria
lon_color_scatter = red_pomegrade
lon_color_line = orange_carrot
angular_color_scatter = green_nephritis
angular_color_line = green_sea
loss_color = blue_midnigth

# Latitude Plot 

Plots.scatter(results_dict["times"][begin:10:end], data_lat[begin:10:end], label="Reference Latitudes", c=lat_color_scatter, markersize=10)
plot_latitude = Plots.plot!(results_dict["fit_times"], fit_lat, label="Estimated Latitude using SphereUDE", 
                    xlabel="Time", yticks=[-60, -30, 0, 30, 60],
                    ylabel="Latitude (degrees)", ylims=(-60,60), lw = 4, c=lat_color_line,
                    legend=:topleft)
plot!(fontfamily="Computer Modern",
     #title="PIL51",
    titlefontsize=18,
    tickfontsize=15,
    legendfontsize=15,
    guidefontsize=18,
    #ylimits=(0.1,10),
    xlimits=(0,100),
    legend=true,
    margin= 7mm,
    size=(1200,500),
    dpi=600)

Plots.savefig(plot_latitude, "examples/curl/plots/latitude.pdf")

### Longitude Plot 

Plots.scatter(results_dict["times"][begin:10:end], data_lon[begin:10:end], label="Reference Longitudes", c=lon_color_scatter, markersize=10)
plot_longitude = Plots.plot!(results_dict["fit_times"], fit_lon, label="Estimated Longitude using SphereUDE", 
                    xlabel="Time", yticks=[-60 , -30, 0, 30, 60],
                    ylabel="Longitude (degrees)", ylims=(-60,60), lw = 4, c=lon_color_line,
                    legend=:topright)
plot!(fontfamily="Computer Modern",
     #title="PIL51",
    titlefontsize=18,
    tickfontsize=15,
    legendfontsize=15,
    guidefontsize=18,
    #ylimits=(0.1,10),
    xlimits=(0,100),
    margin= 7mm,
    size=(1200,500),
    dpi=600)

Plots.savefig(plot_longitude, "examples/curl/plots/longitude.pdf")


### Angular velocity Plot

angular_velocity = mapslices(x -> norm(x), results_dict["fit_rotations"], dims=1)[:]
angular_velocity_true = 10.0 * π / 180.0

plot_angular = Plots.plot(results_dict["fit_times"], angular_velocity, label="Estimated angular velocity", 
                    xlabel="Age (Myr)", 
                    ylabel="Angular velocity", lw = 5, c=angular_color_scatter,
                    legend=:topright)
hline!([angular_velocity_true], label="Reference angular velocity", lw = 4, c=angular_color_line, ls=:dot)
plot!(fontfamily="Computer Modern",
    #title="PIL51",
    titlefontsize=18,
    tickfontsize=15,
    legendfontsize=15,
    guidefontsize=18,
    ylimits=(0.15,0.201),
    xlimits=(0,100),
    margin= 10mm,
    size=(1200,500),
    dpi=300)


Plots.savefig(plot_angular, "examples/curl/plots/angular.pdf")

### Lat and long of inversion

angular_rotation = results_dict["fit_rotations"] ./ angular_velocity[1,:]
angular_rotation = cart2sph(angular_rotation, radians=false)
lat_velocity = angular_rotation[1, :]
lon_velocity = angular_rotation[2, :]

plot_rotation = Plots.plot(results_dict["fit_times"], lat_velocity, label="Estimated Rotation (latitude)", 
                    xlabel="Age (Myr)", 
                    ylabel="Angular velocity", lw = 5, c=lat_color_scatter,
                    legend=:topright)
plot!(results_dict["fit_times"], lon_velocity, label="Estimated Rotation (longitude)", 
                    xlabel="Age (Myr)", 
                    ylabel="Angle (degrees)", lw = 5, c=lon_color_scatter,
                    legend=:topright)
plot!([0,100], [40, -40], lw = 4, c=lat_color_line, ls=:dot, label="Reference Rotation (latitude)")
hline!([0.0], label="Reference Rotation (longitude)", lw = 4, c=lon_color_line, ls=:dot)

plot!(fontfamily="Computer Modern",
    #title="PIL51",
    titlefontsize=18,
    tickfontsize=15,
    legendfontsize=15,
    guidefontsize=18,
    ylimits=(-41,50),
    #xlimits=(10^(-4),10^(-1)),
    margin= 10mm,
    size=(1200,500),
    dpi=300)
    Plots.savefig(plot_angular, "examples/curl/plots/lat_rotation.pdf")

### Lat and long combined

combo_plot = plot(plot_latitude, plot_longitude, plot_angular, plot_rotation, layout = (4, 1))
plot!(fontfamily="Computer Modern",
     #title="PIL51",
    titlefontsize=18,
    tickfontsize=15,
    legendfontsize=15,
    guidefontsize=18,
    #ylimits=(0.1,10),
    xlimits=(0,100),
    margin= 7mm,
    size=(1400,1600),
    dpi=600)
Plots.savefig(combo_plot, "examples/curl/plots/combo.pdf")

### Loss function

losses = results_dict["losses"]

plot_loss = Plots.plot(1:length(losses), losses, label="Loss Function", 
                    xlabel="Epoch", 
                    ylabel="Loss", lw = 5, c=loss_color,
                    yaxis=:log,
                    # yticks=[1,10,100],
                    xlimits=(0,10000),
                    ylimits=(1e-9, 1e1),
                    xticks=[0,2000,4000,6000,8000, 10000],
                    yticks=[1e-1, 1e-3, 1e-5, 1e-7],
                    legend=:topright)

vspan!(plot_loss, [0,2000], color = :navajowhite3, alpha = 0.2, labels = "ADAM");
vspan!(plot_loss, [2000,10000], color = :navajowhite4, alpha = 0.2, labels = "BFGS");

plot!(fontfamily="Computer Modern",
    #title="PIL51",
    titlefontsize=18,
    tickfontsize=15,
    legendfontsize=15,
    guidefontsize=18,
    #ylimits=(0.1,10),
    #xlimits=(10^(-4),10^(-1)),
    margin= 10mm,
    size=(700,500),
    dpi=300)

Plots.savefig(plot_loss, "examples/curl/plots/loss.pdf")
