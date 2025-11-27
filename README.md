# PolynomialQTT

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://tensor4all.github.io/PolynomialQTT.jl/dev)
[![Build Status](https://github.com/tensor4all/PolynomialQTT.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/tensor4all/PolynomialQTT.jl/actions/workflows/CI.yml?query=branch%3Amain)

Julia implementation of multiscale polynomial interpolation for quantics/quantized tensor trains (QTTs).

## Installation

```julia
julia
] 
add https://github.com/tensor4all/PolynomialQTT.jl
```

## Algorithm

This package implements the multiscale interpolative construction of quantized tensor trains as described in:

[Multiscale interpolative construction of quantized tensor trains](https://arxiv.org/abs/2311.12554) by  Michael Lindsey  

## Features

- Single-scale and multi-scale interpolative construction
- 1D to multi dimensions

## Example

```julia
import TensorCrossInterpolation as TCI
import PolynomialQTT

# Single-scale interpolation
f(x) = exp(-x^2)
tt = PolynomialQTT.interpolatesinglescale(f, -2.0, 2.0, 8, 20) # TensorCrossInterpolation.TensorTrain{Float64, 3}

# Multi-scale interpolation with singularity
g(x) = x == 0.0 ? 0.0 : 1/x
tt_multiscale = PolynomialQTT.interpolatemultiscale(g, 0.0, 1.0, 12, 25, [0.0])
```
