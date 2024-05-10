using Test
import PolynomialQTT
import QuanticsGrids as QG
import TensorCrossInterpolation as TCI

@testset "single-scale interpolation" begin
    R = 4
    a, b = -2.8, float(pi)
    f(x) = exp(-x^2)
    K = 20

    tt = PolynomialQTT.interpolatesinglescale(f, a, b, R, K)
    @test TCI.rank(tt) <= K + 1

    grid = QG.DiscretizedGrid{1}(R, a, b)
    quanticsinds = QG.grididx_to_quantics.(Ref(grid), 1:2^R)
    xs = QG.grididx_to_origcoord.(Ref(grid), 1:2^R)
    origdata = f.(xs)
    ttdata = tt.(quanticsinds)
    @test all(abs.(ttdata .- origdata) .< 1e-10)
end


@testset "multiscale interpolation" begin
    R = 4
    a, b = -2.0, sqrt(2)
    f(x) = exp(-x^2) + abs(x)
    K = 25

    tt = PolynomialQTT.interpolatemultiscale(f, a, b, R, K, Float64[0])
    @test TCI.rank(tt) <= K + 2

    grid = QG.DiscretizedGrid{1}(R, a, b)
    quanticsinds = QG.grididx_to_quantics.(Ref(grid), 1:2^R)
    xs = QG.grididx_to_origcoord.(Ref(grid), 1:2^R)
    origdata = f.(xs)
    ttdata = tt.(quanticsinds)
    @test all(abs.(ttdata .- origdata) .< 1e-12)
end


@testset "multiscale interpolation (1/x)" begin
    R = 12
    a, b = 0.0, 1.0
    f(x) = x == 0.0 ? 0.0 : 1 / x
    K = 25

    tt = PolynomialQTT.interpolatemultiscale(f, a, b, R, K, Float64[0])
    @test TCI.rank(tt) <= K + 2

    grid = QG.DiscretizedGrid{1}(R, a, b)
    quanticsinds = QG.grididx_to_quantics.(Ref(grid), 1:2^R)
    xs = QG.grididx_to_origcoord.(Ref(grid), 1:2^R)
    origdata = f.(xs)
    ttdata = tt.(quanticsinds)
    # skip the first element because it's zero, and use the relative error
    @test all(abs.(ttdata[2:end] .- origdata[2:end]) ./ abs.(origdata[2:end]) .< 1e-12)
end

