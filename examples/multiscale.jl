using PolynomialQTT
using CairoMakie
import QuanticsGrids as QG
import TensorCrossInterpolation as TCI 

α = 0.01
f(x) = exp(-0.5*(x/α)^2)
a,b = 0.0, 1.0
K = 25
R = 8

tt = PolynomialQTT.interpolatesinglescale(f, a, b, R, K)

grid = QG.DiscretizedGrid(R, a, b)

plotquantics = QG.grididx_to_quantics.(Ref(grid), 1:2^R)
plotx = QG.grididx_to_origcoord.(Ref(grid), 1:2^R)
origdata = f.(plotx)
ttdata = tt.(plotquantics)

let 
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel=L"x", ylabel=L"f(x)", title="Single-scale interpolation")
    lines!(ax, plotx, f.(plotx), label=L"Origignal function", linewidth=2.0, linestyle=:solid)
    lines!(ax, plotx, ttdata, label=L"K=%$K", linewidth=2.0,linestyle=:dash)
    fig
end

let 
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel=L"x", ylabel="Error", title="Error single-scale interpolation")
    lines!(ax, plotx, abs.(ttdata .- origdata))
    fig
end

let 
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel=L"\ell", ylabel=L"\chi_\ell", title="Bond dimension")
    scatterlines!(ax,  1:R-1, TCI.linkdims(tt))
    fig
end