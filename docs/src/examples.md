# Examples 

## Highly oscillatory function interpolation

In this example, we demonstrate how to do multi-scale interpolation of a highly oscillatory function using Polynomial QTT representation. We consider the function ``f(x) = \cos(x^2) + \sin(πx)`` on the interval ``[-2, \sqrt{2}]`` using ``2^8`` discretization points. 

We begin by loading the necessary packages and defining the function and grid parameters.

```@example oscillatory
using PolynomialQTT
import TensorCrossInterpolation as TCI 

f(x) = cos(x^2) + sin(π * x)
a,b = -2.0, sqrt(2)
K = 10
R = 8;
```
We call the `interpolatesinglescale` function to perform the interpolation. This function takes the function `f`, the interval endpoints `a` and `b`, the number of interpolation nodes `K`, and the number of tensor cores `R` as inputs.

```@example oscillatory
tt = PolynomialQTT.interpolatesinglescale(f, a, b, R, K)
```
This returns a `TensorTrain` object. Using [QuanticsGrids.jl](https://github.com/tensor4all/QuanticsGrids.jl) we can plot the results and compare the interpolated function with the original function. 

```@example oscillatory
import QuanticsGrids as QG
using CairoMakie

grid = QG.DiscretizedGrid(R, a, b)

plotquantics = QG.grididx_to_quantics.(Ref(grid), 1:2^R)
plotx = QG.grididx_to_origcoord.(Ref(grid), 1:2^R)
origdata = f.(plotx)
ttdata = tt.(plotquantics)

let 
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel=L"x", ylabel=L"f(x)", title="Single-scale interpolation")
    lines!(ax, plotx, f.(plotx), label=L"f(x)", linewidth=2.0, linestyle=:solid)
    lines!(ax, plotx, ttdata, label=L"K=%$K", linewidth=2.0,linestyle=:dash)
    axislegend(ax, pos=:best)
    fig
end
```
We see in this case we are able to achieve a good approximation of the highly oscillatory function using only ``K=10`` interpolation nodes. 
