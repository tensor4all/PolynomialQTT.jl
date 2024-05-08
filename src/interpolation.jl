struct LagrangePolynomials{T}
    grid::Vector{T}
    baryweights::Vector{T}
end

function (P::LagrangePolynomials{T})(alpha::Int, x::T)::T where {T}
    if abs(x - P.grid[alpha+1]) >= 1e-14
        return prod(x .- P.grid) * P.baryweights[alpha+1] / (x - P.grid[alpha+1])
    else
        return one(T)
    end
end

function getChebyshevGrid(K::Int)::LagrangePolynomials{Float64}
    chebgrid = 0.5 * (1.0 .- cospi.((0:K) / K))
    baryweights = [
        prod(j == m ? 1.0 : 1.0 / (chebgrid[j+1] - chebgrid[m+1]) for m in 0:K)
        for j in 0:K
    ]
    return LagrangePolynomials{Float64}(chebgrid, baryweights)
end

function interpolationtensor(P::LagrangePolynomials{Float64})
    return [
        P(alpha, (sigma + P.grid[beta+1]) / 2)
        for alpha in (0:length(P.grid)-1),
            sigma in [0, 1],
            beta in (0:length(P.grid)-1)
    ]
end

function interpolatesinglescale(
    f,
    a::Float64, b::Float64,
    numbits::Int,
    polynomialdegree::Int
)
    P = getChebyshevGrid(polynomialdegree)
    Aleft = [
        f.(a + (b - a) * (sigma + P.grid[beta+1]) / 2)
        for sigma in [0, 1], beta in 0:polynomialdegree
    ]
    Acenter = interpolationtensor(P)
    Aright = [
        P(alpha, sigma/2)
        for alpha in 0:polynomialdegree, sigma in [0, 1]
    ]

    return TCI.tensortrain(vcat(
        [reshape(Aleft, 1, 2, polynomialdegree+1)],
        fill(Acenter, numbits-2),
        [reshape(Aright, polynomialdegree+1, 2, 1)]
    ))
end
