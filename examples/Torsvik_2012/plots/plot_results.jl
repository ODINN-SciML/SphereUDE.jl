using Pkg; Pkg.activate(".")

using SphereUDE
using Serialization
using Plots 
using Plots.PlotMeasures
using JLD2
using LinearAlgebra

JLD2.@load "examples/Torsvik_2012/results/results_dict.jld2" 

data_directions_sph = cart2sph(results_dict["directions"], radians=false)
fit_directions_sph = cart2sph(results_dict["fit_directions"], radians=false)

data_lat = data_directions_sph[1,:]
fit_lat  = fit_directions_sph[1,:]

data_lon = data_directions_sph[2,:]
fit_lon  = fit_directions_sph[2,:]

α_error = 140 ./ sqrt.(results_dict["kappas"])

# Latitude Plot 

Plots.scatter(results_dict["times"], data_lat, label="Paleopole Latitudes", yerr=α_error, c=:lightsteelblue2, markersize=5)
plot_latitude = Plots.plot!(results_dict["fit_times"], fit_lat, label="Estimated APWP using SphereUDE", 
                    xlabel="Age (Myr)", yticks=[-90, -60, -30, 0, 30, 60],
                    ylabel="Latitude (degrees)", ylims=(-90,60), lw = 4, c=:brown,
                    legend=:topleft)
plot!(fontfamily="Computer Modern",
     #title="PIL51",
    titlefontsize=18,
    tickfontsize=15,
    legendfontsize=15,
    guidefontsize=18,
    #ylimits=(0.1,10),
    #xlimits=(10^(-4),10^(-1)),
    legend=true,
    margin= 7mm,
    size=(1200,500),
    dpi=600)

Plots.savefig(plot_latitude, "examples/Torsvik_2012/plots/latitude.pdf")

### Longitude Plot 

α_error_lon = α_error ./ cos.(π ./ 180. .* data_lat)

Plots.scatter(results_dict["times"], data_lon, label="Paleopole Longitudes", yerr=α_error_lon, c=:lightsteelblue2, markersize=5)
plot_longitude = Plots.plot!(results_dict["fit_times"], fit_lon, label="Estimated APWP using SphereUDE", 
                    xlabel="Age (Myr)", yticks=[-180, -120, -60 , 0, 60, 120, 180],
                    ylabel="Longitude (degrees)", ylims=(-180,180), lw = 4, c=:brown,
                    legend=:bottomright)
plot!(fontfamily="Computer Modern",
     #title="PIL51",
    titlefontsize=18,
    tickfontsize=15,
    legendfontsize=15,
    guidefontsize=18,
    #ylimits=(0.1,10),
    #xlimits=(10^(-4),10^(-1)),
    margin= 7mm,
    size=(1200,500),
    dpi=600)

Plots.savefig(plot_longitude, "examples/Torsvik_2012/plots/longitude.pdf")

### Lat and long combined

combo_plot = plot(plot_latitude, plot_longitude, layout = (2, 1))
plot!(fontfamily="Computer Modern",
     #title="PIL51",
    titlefontsize=18,
    tickfontsize=15,
    legendfontsize=15,
    guidefontsize=18,
    #ylimits=(0.1,10),
    #xlimits=(10^(-4),10^(-1)),
    margin= 7mm,
    size=(1200,800),
    dpi=600)
Plots.savefig(combo_plot, "examples/Torsvik_2012/plots/latitude_longitude.pdf")


### Angular velocity Plot

angular_velocity = mapslices(x -> norm(x), results_dict["fit_rotations"], dims=1)[:]
angular_velocity_path = [norm(cross(results_dict["fit_directions"][:,i], results_dict["fit_rotations"][:,i] )) for i in axes(results_dict["fit_rotations"],2)]

plot_angular = Plots.plot(results_dict["fit_times"], angular_velocity, label="Maximum total angular velocity", 
                    xlabel="Age (Myr)", 
                    ylabel="Angular velocity (degrees/My)", lw = 5, c=:darkgreen,
                    legend=:topleft)
plot!(results_dict["fit_times"], angular_velocity_path, label="Pole angular velocity", lw = 4, c=:lightseagreen, ls=:dot)
plot!(fontfamily="Computer Modern",
    #title="PIL51",
    titlefontsize=18,
    tickfontsize=15,
    legendfontsize=15,
    guidefontsize=18,
    #ylimits=(0.1,10),
    #xlimits=(10^(-4),10^(-1)),
    margin= 10mm,
    size=(1200,500),
    dpi=300)


Plots.savefig(plot_angular, "examples/Torsvik_2012/plots/angular.pdf")


### Loss function

losses = results_dict["losses"]

plot_loss = Plots.plot(1:length(losses), losses, label="Loss Function", 
                    xlabel="Epoch", 
                    ylabel="Loss", lw = 5, c=:indigo,
                    yaxis=:log,
                    # yticks=[1,10,100],
                    xlimits=(0,10000),
                    ylimits=(10, 1000),
                    xticks=[0,2500,5000,7500,10000],
                    legend=:topright)

vspan!(plot_loss, [0,5000], color = :navajowhite3, alpha = 0.2, labels = "ADAM");
vspan!(plot_loss, [5000,10000], color = :navajowhite4, alpha = 0.2, labels = "BFGS");

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

Plots.savefig(plot_loss, "examples/Torsvik_2012/plots/loss.pdf")
