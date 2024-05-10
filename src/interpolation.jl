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

function _compress_train(train::Vector{Array{Float64,3}}, tolerance, maxbonddim)
    L = length(train)

    U, R = truncatedsvd(reshape(train[1], :, size(train[1], 3)); tolerance, maxbonddim)
    train_compress = Array{Float64,3}[reshape(U, 1, 2, size(U, 2))]

    localdims = [size(t, 2) for t in train]

    for ell in 2:L-1
        B = R * reshape(train[ell], size(R, 2), :)
        U, R = truncatedsvd(
            reshape(B, size(R, 1) * localdims[ell], :);
            tolerance, maxbonddim
        )
        push!(train_compress, reshape(U, size(last(train_compress), 3), localdims[ell], size(U, 2)))
    end

    lasttensor = R * reshape(train[end], size(train[end], 1), :)
    push!(train_compress, reshape(lasttensor, size(R, 1), localdims[end], 1))

    return train_compress
end

function interpolatesinglescale(
    f,
    a::Float64, b::Float64,
    numbits::Int,
    polynomialdegree::Int;
    tolerance=1e-12,
    maxbonddim=typemax(Int)
)
    _scale(x) = a + (b - a) * x
    P = getChebyshevGrid(polynomialdegree)
    Aleft = [
        f.(_scale.((sigma + P.grid[beta+1]) / 2))
        for sigma in [0, 1], beta in 0:polynomialdegree
    ]
    Acenter = interpolationtensor(P)
    Aright = [
        P(alpha, sigma / 2)
        for alpha in 0:polynomialdegree, sigma in [0, 1]
    ]

    train_ = [
        reshape(Aleft, 1, 2, size(Aleft, 2)),
        fill(Acenter, numbits - 2)...,
        reshape(Aright, size(Aright, 1), 2, 1)
    ]

    if tolerance == 0.0 && maxbonddim == typemax(Int)
        return TCI.tensortrain(deepcopy(train_))
    else
        return TCI.tensortrain(_compress_train(train_, tolerance, maxbonddim))
    end
end

function _direct_product(matrices...)
    N = length(matrices)
    result = matrices[1]
    for i in 2:N
        result = kron(result, matrices[i])
    end
    return result
end


"""
`f` takes `N` Float64 arguments and returns a Float64 value.
"""
function interpolatesinglescale(
    f,
    a::NTuple{N,Float64}, b::NTuple{N,Float64},
    numbits::Int,
    polynomialdegree::Int;
    tolerance=1e-12,
    maxbonddim=typemax(Int),
    unfoldingscheme=:interleaved
) where {N}
    unfoldingscheme == :interleaved || Error("unsupported unfolding scheme")

    _scale(x::NTuple{N,Float64})::NTuple{N,Float64} = tuple((a .+ (b .- a) .* x)...)

    P = getChebyshevGrid(polynomialdegree)

    Aleft = Array{Float64,N + 1}(undef, 2, fill(polynomialdegree + 1, N)...)
    for sigma in [0, 1], betas in Iterators.product(ntuple(x -> 0:polynomialdegree, N)...)
        point = (N == 1) ?
            ((sigma + P.grid[betas[1]+1]) / 2,) :
            tuple((sigma + P.grid[betas[1]+1]) / 2, P.grid[collect(betas[2:end].+1)]...)
        Aleft[sigma+1, (betas .+ 1)...] = f(_scale(point)...)
    end

    identity = Matrix{Float64}(I, polynomialdegree + 1, polynomialdegree + 1)

    Acenter_ = interpolationtensor(P)
    Acenter = Array{Float64,3}[]
    for n in 1:N
        push!(Acenter, zeros(Float64, (polynomialdegree + 1)^N, 2, (polynomialdegree + 1)^N))
        for sigma in [0, 1]
            Acenter[n][:, sigma+1, :] .= _direct_product(fill(identity, n - 1)..., Acenter_[:, sigma+1, :], fill(identity, N - n)...)
        end
    end

    Aright_ = [
        P(alpha, sigma / 2)
        for alpha in 0:polynomialdegree, sigma in [0, 1]
    ]
    Aright = Array{Float64,3}[]
    for n in 1:N
        push!(
            Aright,
            zeros(Float64, (polynomialdegree + 1)^(N - n + 1), 2, (polynomialdegree + 1)^(N - n))
        )
        for sigma in [0, 1]
            Aright[n][:, sigma+1, :] .= _direct_product(Aright_[:, sigma+1], fill(identity, N - n)...)
        end
    end

    train_ = Array{Float64,3}[]
    push!(train_, reshape(Aleft, 1, 2, (polynomialdegree + 1)^N))
    for ell in 1:numbits-1, n in 1:N
        if ell == 1 && n == 1
            continue
        end
        push!(train_, Acenter[n])
    end
    for n in 1:N
        push!(train_, Aright[n])
    end

    if tolerance == 0.0 && maxbonddim == typemax(Int)
        return TCI.tensortrain(deepcopy(train_))
    else
        return TCI.tensortrain(_compress_train(train_, tolerance, maxbonddim))
    end
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