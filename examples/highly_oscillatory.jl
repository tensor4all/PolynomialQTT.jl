using PolynomialQTT
using CairoMakie
import QuanticsGrids as QG
import TensorCrossInterpolation as TCI
using LinearAlgebra

f(x) = cos(x^2) + sin(π * x)
a, b = -2.0, sqrt(2)
K = 10
R = 8

tt = PolynomialQTT.interpolatesinglescale(f, a, b, R, K)
tt_adaptive = interpolateadaptive(f, a, b, R, K)

grid = QG.DiscretizedGrid(R, a, b)

plotquantics = QG.grididx_to_quantics.(Ref(grid), 1:(2^R))
plotx = QG.grididx_to_origcoord.(Ref(grid), 1:(2^R))
origdata = f.(plotx)
ttdata = tt.(plotquantics)

let
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"f(x)", title = "Single-scale interpolation")
    lines!(ax, plotx, f.(plotx), label = "Original function", linewidth = 2.0, linestyle = :solid)
    lines!(ax, plotx, ttdata, label = L"K=%$K", linewidth = 2.0, linestyle = :dash)
    axislegend(ax, pos = :best)
    fig
end

let
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = L"x", ylabel = "Error", title = "Error single-scale interpolation")
    lines!(ax, plotx, abs.(ttdata .- origdata))
    fig
end

let
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = L"\ell", ylabel = L"\chi_\ell", title = "Bond dimension")
    scatterlines!(ax, 1:(R - 1), TCI.linkdims(tt))
    fig
end
