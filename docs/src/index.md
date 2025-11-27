# PolynomialQTT.jl

This is the documentation for [PolynomialQTT.jl](https://github.com/tensor4all/PolynomialQTT.jl).

## Installation

To install PolynomialQTT.jl enter the following commands:

```julia
julia 
] 
add https://github.com/tensor4all/PolynomialQTT.jl
```

### Usage Example

```@example simple
import TensorCrossInterpolation as TCI
import PolynomialQTT

# Single-scale interpolation
f(x) = exp(-x^2)
tt = PolynomialQTT.interpolatesinglescale(f, -2.0, 2.0, 8, 20) # TensorCrossInterpolation.TensorTrain{Float64, 3}

# Multi-scale interpolation with singularity
g(x) = x == 0.0 ? 0.0 : 1/x
tt_ms = PolynomialQTT.interpolatemultiscale(g, 0.0, 1.0, 12, 25, [0.0])
```