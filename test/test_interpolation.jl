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

@testset "single-scale interpolation (N=1)" begin
    R = 4
    a, b = -2.8, float(pi)
    f(x) = exp(-x^2)
    K = 20

    tt = PolynomialQTT.interpolatesinglescale(f, (a,), (b,), R, K)
    @test TCI.rank(tt) <= K + 1

    grid = QG.DiscretizedGrid{1}(R, a, b)
    quanticsinds = QG.grididx_to_quantics.(Ref(grid), 1:2^R)
    xs = QG.grididx_to_origcoord.(Ref(grid), 1:2^R)
    origdata = f.(xs)
    ttdata = tt.(quanticsinds)
    @test all(abs.(ttdata .- origdata) .< 1e-10)
end


@testset "single-scale interpolation (N=2)" begin
    R = 4
    K = 20
    a, b = (-1.0, -1.0), (2.0, 2.0)
    f(x, y) = exp(-x^2 - y^3)

    tt = PolynomialQTT.interpolatesinglescale(f, a, b, R, K)
    @test TCI.rank(tt) <= (K + 1)^2
    @test length(tt) == R

    grid = QG.DiscretizedGrid{2}(R, a, b; unfoldingscheme=:fused)

    quanticsinds = [QG.grididx_to_quantics(grid, (i, j)) for i in 1:2^R, j in 1:2^R]
    xs = [QG.grididx_to_origcoord(grid, (i, j)) for i in 1:2^R, j in 1:2^R]

    origdata = [f(x...) for x in xs]
    ttdata = tt.(quanticsinds)

    @test all(abs.(ttdata .- origdata) .< 1e-10)
end


@testset "single-scale interpolation (N=3)" begin
    R = 4
    K = 15
    a, b = (-1.0, -1.0, 0.0), (2.0, 2.0, 1.0)
    f(x, y, z) = exp(-x^2 - y^3 - 2 * z^2)

    tt = PolynomialQTT.interpolatesinglescale(f, a, b, R, K)
    @test TCI.rank(tt) <= (K + 1)^3
    @test length(tt) == R

    grid = QG.DiscretizedGrid{3}(R, a, b; unfoldingscheme=:fused)

    quanticsinds = [QG.grididx_to_quantics(grid, (i, j, k)) for i in 1:2^R, j in 1:2^R, k in 1:2^R]
    xs = [QG.grididx_to_origcoord(grid, (i, j, k)) for i in 1:2^R, j in 1:2^R, k in 1:2^R]

    origdata = [f(x...) for x in xs]
    ttdata = tt.(quanticsinds)

    @test all(abs.(ttdata .- origdata) .< 1e-7)
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


@testset "_direct_product_coretensors (two tensors)" begin
    χ = 3
    coretensors = [randn(χ, 2, χ) for _ in 1:2]
    c12 = PolynomialQTT._direct_product_coretensors(coretensors)
    c12_ref = Array{Float64,6}(undef, χ, χ, 2, 2, χ, χ)
    for i in 1:χ, j in 1:χ, k in 1:2, l in 1:2, m in 1:χ, n in 1:χ
        c12_ref[i, j, k, l, m, n] = coretensors[1][i, k, m] * coretensors[2][j, l, n]
    end
    @assert vec(c12) ≈ vec(c12_ref)
end


@testset "_direct_product_coretensors (three tensors)" begin
    χ = 3
    coretensors = [randn(χ, 2, χ) for _ in 1:3]
    c123 = PolynomialQTT._direct_product_coretensors(coretensors)
    c123_ref = Array{Float64,9}(undef, χ, χ, χ, 2, 2, 2, χ, χ, χ)
    for i in 1:χ, j in 1:χ, k in 1:χ, l in 1:2, m in 1:2, n in 1:2, o in 1:χ, p in 1:χ, q in 1:χ
        c123_ref[i, j, k, l, m, n, o, p, q] =
            coretensors[1][i, l, o] * coretensors[2][j, m, p] * coretensors[3][k, n, q]
    end
    @assert vec(c123) ≈ vec(c123_ref)
end