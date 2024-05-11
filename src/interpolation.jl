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
    localdims = [size(t, 2) for t in train]

    U, R = truncatedsvd(reshape(train[1], :, size(train[1], 3)); tolerance, maxbonddim)
    train_compress = Array{Float64,3}[reshape(U, 1, localdims[2], size(U, 2))]

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


# Contract core tensors for difference variables at a given resolution level
function _direct_product_coretensors(coretensors::AbstractArray{Array{T,3}})::Array{T,3} where {T}
    if length(coretensors) == 1
        return coretensors[1]
    end

    # (alpha, s, beta) * (alpha', s', beta') = (alpha, alpha', s, s', beta, beta')
    c12_1 = reshape(coretensors[1], size(coretensors[1])..., 1, 1, 1) .*
            reshape(coretensors[2], 1, 1, 1, size(coretensors[2])...)
    c12_2 = permutedims(c12_1, (1, 4, 2, 5, 3, 6))
    c12 = reshape(
        c12_2,
        size(c12_2, 1) * size(c12_2, 2),
        size(c12_2, 3) * size(c12_2, 4),
        size(c12_2, 5) * size(c12_2, 6)
    )
    return _direct_product_coretensors([c12, coretensors[3:end]...])
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
    unfoldingscheme=:fused
) where {N}
    unfoldingscheme == :fused || Error("unsupported unfolding scheme $(unfoldingscheme)")

    _scale(x::NTuple{N,Float64})::NTuple{N,Float64} = tuple((a .+ (b .- a) .* x)...)
    fillt(N, v) = ntuple(i -> v, N)

    P = getChebyshevGrid(polynomialdegree)

    Aleft = Array{Float64,2N}(undef, fillt(N, 2)..., fillt(N, polynomialdegree + 1)...)
    for sigmas in Iterators.product(ntuple(x -> 0:1, N)...)
        for betas in Iterators.product(ntuple(x -> 0:polynomialdegree, N)...)
            point = tuple(((sigmas[n] + P.grid[betas[n]+1]) / 2 for n in 1:N)...)
            Aleft[(sigmas .+ 1)..., (betas .+ 1)...] = f(_scale(point)...)
        end
    end

    Acenter_ = interpolationtensor(P)
    Acenter = _direct_product_coretensors(fill(Acenter_, N))

    Aright_ = [
        P(alpha, sigma / 2)
        for alpha in 0:polynomialdegree, sigma in [0, 1]
    ]
    Aright = _direct_product_coretensors(fill(reshape(Aright_, size(Aright_)..., 1), N))

    train_ = [
        reshape(Aleft, 1, 2^N, :),
        fill(Acenter, numbits - 2)...,
        Aright
    ]

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