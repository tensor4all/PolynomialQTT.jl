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

function replacenothing(value::Union{T,Nothing}, default::T)::T where {T}
    if isnothing(value)
        return default
    else
        return value
    end
end

function truncatedsvd(A; tolerance, maxbonddim)
    factorization = LinearAlgebra.svd(A)
    trunci = min(
        replacenothing(findlast(>(tolerance), factorization.S), 1),
        maxbonddim
    )
    return (
        factorization.U[:, 1:trunci],
        Diagonal(factorization.S[1:trunci]) * factorization.Vt[1:trunci, :]
    )
end

function interpolatesinglescale(
    f,
    a::Float64, b::Float64,
    numbits::Int,
    polynomialdegree::Int;
    tolerance=1e-12,
    maxbonddim=typemax(Int)
)
    P = getChebyshevGrid(polynomialdegree)
    Aleft = [
        f.(a + (b - a) * (sigma + P.grid[beta+1]) / 2)
        for sigma in [0, 1], beta in 0:polynomialdegree
    ]
    Acenter = interpolationtensor(P)
    Aright = [
        P(alpha, sigma / 2)
        for alpha in 0:polynomialdegree, sigma in [0, 1]
    ]

    A = reshape(Acenter, polynomialdegree + 1, 2 * (polynomialdegree + 1))
    U, R = truncatedsvd(Aleft; tolerance, maxbonddim)
    train = Array{Float64,3}[reshape(U, 1, 2, size(U, 2))]
    for ell in 2:numbits-1
        U, R = truncatedsvd(
            reshape(R * A, 2 * size(R, 1), polynomialdegree + 1);
            tolerance, maxbonddim
        )
        push!(train, reshape(U, size(last(train), 3), 2, size(U, 2)))
    end
    push!(train, reshape(R * Aright, size(last(train), 3), 2, 1))

    return TCI.tensortrain(train)
end

function indicator(dims, oneindex)
    T = zeros(dims)
    T[oneindex...] = 1.0
    return T
end

function interpolatemultiscale(
    f,
    a::Float64, b::Float64,
    numbits::Int,
    polynomialdegree::Int,
    cusplocations::Vector{Float64}
)
    function issafe(f, interval::Interval{Float64})
        return !any(cusplocations .∈ Ref(interval))
    end

    P = getChebyshevGrid(polynomialdegree)
    intervallist = Interval{Float64}[]
    newintervallist = Interval{Float64}[]
    Acore = interpolationtensor(P)

    Tfirstup = zeros(2, polynomialdegree + 1)
    Tfirstdown = zeros(2, 0)
    for (i, interval) in enumerate(split(Interval{Float64}(a, b)))
        if issafe(f, interval)
            Tfirstup[i, :] = f.(interval.start .+ intervallength(interval) * P.grid)
        else
            Tfirstdown = [Tfirstdown;; indicator((2,), (i,))]
            push!(intervallist, interval)
        end
    end
    Tfirst = [Tfirstup;; Tfirstdown]
    train = [reshape(Tfirst, 1, 2, size(Tfirst, 2))]

    for ell in 2:numbits-1
        newintervallist = Interval{Float64}[]
        qell = length(intervallist)
        F = zeros(qell, 2, polynomialdegree + 1)
        χ = zeros(qell, 2, 0)
        for (i, interval) in enumerate(intervallist)
            for (s, subinterval) in enumerate(split(interval))
                if issafe(f, subinterval)
                    F[i, s, :] = f.(subinterval.start .+ intervallength(subinterval) * P.grid)
                else
                    χ = [χ;;; indicator((qell, 2, 1), (i, s, 1))]
                    push!(newintervallist, subinterval)
                end
            end
        end
        Ak = [Acore; F;;; zeros(polynomialdegree + 1, 2, size(χ, 3)); χ]
        push!(train, Ak)
        intervallist = newintervallist
    end

    Aright = [
        P(alpha, sigma / 2)
        for alpha in 0:polynomialdegree, sigma in [0, 1]
    ]
    F = zeros(length(intervallist), 2)
    for (i, interval) in enumerate(intervallist)
        for (s, subinterval) in enumerate(split(interval))
            F[i, s, 1] = f(subinterval.start)
        end
    end
    Alast = [Aright; F]
    push!(train, reshape(Alast, size(Alast, 1), 2, 1))

    return TCI.tensortrain(train)
end
