struct LagrangePolynomials{T}
    grid::Vector{T}
    baryweights::Vector{T}
end

function (P::LagrangePolynomials{T})(alpha::Int, x::T)::T where {T}
    if abs(x - P.grid[alpha + 1]) >= 1.0e-14
        return prod(x .- P.grid) * P.baryweights[alpha + 1] / (x - P.grid[alpha + 1])
    else
        return one(T)
    end
end

function getChebyshevGrid(K::Int)::LagrangePolynomials{Float64}
    chebgrid = 0.5 * (1.0 .- cospi.((0:K) / K))
    baryweights = [
        prod(j == m ? 1.0 : 1.0 / (chebgrid[j + 1] - chebgrid[m + 1]) for m in 0:K)
            for j in 0:K
    ]
    return LagrangePolynomials{Float64}(chebgrid, baryweights)
end

function interpolationtensor(P::LagrangePolynomials{Float64})
    return [
        P(alpha, (sigma + P.grid[beta + 1]) / 2)
            for alpha in (0:(length(P.grid) - 1)),
            sigma in [0, 1],
            beta in (0:(length(P.grid) - 1))
    ]
end

function replacenothing(value::Union{T, Nothing}, default::T)::T where {T}
    if isnothing(value)
        return default
    else
        return value
    end
end

function truncatedsvd(A; tolerance, maxbonddim)
    factorization = LinearAlgebra.svd(A)
    trunci = min(
        replacenothing(findlast(>(tolerance * factorization.S[1]), factorization.S), 1),
        maxbonddim
    )
    return (
        factorization.U[:, 1:trunci],
        Diagonal(factorization.S[1:trunci]) * factorization.Vt[1:trunci, :],
    )
end

function _compress_train(train::Vector{Array{Float64, 3}}, tolerance, maxbonddim)
    if tolerance == 0.0 && maxbonddim == typemax(Int)
        return TCI.TensorTrain(deepcopy(train))
    end

    L = length(train)
    localdims = [size(t, 2) for t in train]

    U, R = truncatedsvd(reshape(train[1], :, size(train[1], 3)); tolerance, maxbonddim)
    train_compress = Array{Float64, 3}[reshape(U, 1, localdims[2], size(U, 2))]

    for ell in 2:(L - 1)
        B = R * reshape(train[ell], size(R, 2), :)
        U, R = truncatedsvd(
            reshape(B, size(R, 1) * localdims[ell], :);
            tolerance, maxbonddim
        )
        push!(train_compress, reshape(U, size(last(train_compress), 3), localdims[ell], size(U, 2)))
    end

    lasttensor = R * reshape(train[end], size(train[end], 1), :)
    push!(train_compress, reshape(lasttensor, size(R, 1), localdims[end], 1))

    tt = TCI.TensorTrain(deepcopy(train_compress))
    TCI.compress!(tt, :SVD; tolerance = tolerance, maxbonddim = maxbonddim)
    return tt
end

function interpolatesinglescale(
        f,
        a::Float64, b::Float64,
        numbits::Int,
        polynomialdegree::Int;
        tolerance = 1.0e-12,
        maxbonddim = typemax(Int)
    )
    _scale(x) = a + (b - a) * x
    P = getChebyshevGrid(polynomialdegree)
    Aleft = [
        f.(_scale.((sigma + P.grid[beta + 1]) / 2))
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
        reshape(Aright, size(Aright, 1), 2, 1),
    ]

    return _compress_train(train_, tolerance, maxbonddim)
end


# Contract core tensors for difference variables at a given resolution level
function _direct_product_coretensors(coretensors::AbstractArray{Array{T, 3}})::Array{T, 3} where {T}
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

function interpolatesinglescale(
        f,
        a::NTuple{N, Float64}, b::NTuple{N, Float64},
        numbits::Int,
        polynomialdegree::Int;
        tolerance = 1.0e-12,
        maxbonddim = typemax(Int),
        unfoldingscheme = :fused
    ) where {N}
    unfoldingscheme == :fused || Error("unsupported unfolding scheme $(unfoldingscheme)")

    _scale(x::NTuple{N, Float64})::NTuple{N, Float64} = tuple((a .+ (b .- a) .* x)...)
    fillt(N, v) = ntuple(i -> v, N)

    P = getChebyshevGrid(polynomialdegree)

    Aleft = Array{Float64, 2N}(undef, fillt(N, 2)..., fillt(N, polynomialdegree + 1)...)
    for sigmas in Iterators.product(ntuple(x -> 0:1, N)...)
        for betas in Iterators.product(ntuple(x -> 0:polynomialdegree, N)...)
            point = tuple(((sigmas[n] + P.grid[betas[n] + 1]) / 2 for n in 1:N)...)
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
        Aright,
    ]

    return _compress_train(train_, tolerance, maxbonddim)
end

function indicator(dims, oneindex)
    T = zeros(dims)
    T[oneindex...] = 1.0
    return T
end

function _evalf(f, interval::Interval{Float64}, P::LagrangePolynomials{Float64})
    return f.(interval.start .+ intervallength(interval) * P.grid)
end

function _evalf(f, interval::NInterval{N, Float64}, P::LagrangePolynomials{Float64}) where {N}
    NP = length(P.grid)
    results = Array{Float64, N}(undef, ntuple(i -> NP, N)...)
    scaledgrid = [interval.start[n] .+ intervallength(interval)[n] * P.grid for n in 1:N]

    for inds in Iterators.product(ntuple(x -> 1:NP, N)...)
        points = [g[i] for (g, i) in zip(scaledgrid, inds)]
        results[inds...] = f(points...)
    end
    return results
end

function interpolatemultiscale(
        f,
        a::Float64, b::Float64,
        numbits::Int,
        polynomialdegree::Int,
        cusplocations::Vector{Float64};
        tolerance = 1.0e-12,
        maxbonddim = typemax(Int),
        unfoldingscheme = :fused
    )
    return interpolatemultiscale(
        f, (a,), (b,), numbits, polynomialdegree, [(c,) for c in cusplocations];
        tolerance = tolerance, maxbonddim = maxbonddim, unfoldingscheme = unfoldingscheme
    )
end


function interpolatemultiscale(
        f,
        a::NTuple{N, Float64}, b::NTuple{N, Float64},
        numbits::Int,
        polynomialdegree::Int,
        cusplocations::Vector{NTuple{N, Float64}};
        tolerance = 1.0e-12,
        maxbonddim = typemax(Int),
        unfoldingscheme = :fused
    ) where {N}
    unfoldingscheme == :fused || Error("unsupported unfolding scheme $(unfoldingscheme)")

    function issafe(f, interval::NInterval{N, Float64})
        return !any(cusplocations .∈ Ref(interval))
    end

    P = getChebyshevGrid(polynomialdegree)
    intervallist = NInterval{N, Float64}[]
    newintervallist = NInterval{N, Float64}[]
    Acore_ = interpolationtensor(P)
    Acore = _direct_product_coretensors(fill(Acore_, N))

    Tfirstup = zeros(2^N, (polynomialdegree + 1)^N)
    Tfirstdown = zeros(2^N, 0)
    for (i, interval) in enumerate(split(NInterval{N, Float64}(a, b)))
        if issafe(f, interval)
            Tfirstup[i, :] = _evalf(f, interval, P)
        else
            Tfirstdown = [Tfirstdown;; indicator((2^N,), (i,))]
            push!(intervallist, interval)
        end
    end
    Tfirst = [Tfirstup;; Tfirstdown]
    train = [reshape(Tfirst, 1, 2^N, size(Tfirst, 2))]

    for ell in 2:(numbits - 1)
        newintervallist = NInterval{N, Float64}[]
        qell = length(intervallist)
        F = zeros(qell, 2^N, (polynomialdegree + 1)^N)
        χ = zeros(qell, 2^N, 0)
        for (i, interval) in enumerate(intervallist)
            for (s, subinterval) in enumerate(split(interval))
                if issafe(f, subinterval)
                    F[i, s, :] = _evalf(f, subinterval, P)
                else
                    χ = [χ;;; indicator((qell, 2^N, 1), (i, s, 1))]
                    push!(newintervallist, subinterval)
                end
            end
        end
        Ak = [Acore; F;;; zeros((polynomialdegree + 1)^N, 2^N, size(χ, 3)); χ]
        push!(train, Ak)
        intervallist = newintervallist
    end

    Aright_ = [
        P(alpha, sigma / 2)
            for alpha in 0:polynomialdegree, sigma in [0, 1]
    ]
    Aright = _direct_product_coretensors(fill(reshape(Aright_, size(Aright_)..., 1), N))

    F = zeros(length(intervallist), 2^N)
    for (i, interval) in enumerate(intervallist)
        for (s, subinterval) in enumerate(split(interval))
            F[i, s, 1] = f(subinterval.start...)
        end
    end
    Alast = [Aright; F]
    push!(train, reshape(Alast, size(Alast, 1), 2^N, 1))

    return _compress_train(train, tolerance, maxbonddim)
end

function estimate_interpolation_error(
        f,
        interval::Interval{Float64},
        P::LagrangePolynomials{Float64}
    )::Float64
    values = _evalf(f, interval, P)
    len = intervallength(interval)

    test_points = 0.5 * (1.0 .- cospi.((0:(2 * length(P.grid) - 1)) / (2 * length(P.grid) - 1)))

    max_err = 0.0
    for t in test_points
        x = interval.start + len * t
        interp_val = sum(values[α + 1] * P(α, t) for α in 0:(length(P.grid) - 1))
        max_err = max(max_err, abs(interp_val - f(x)))
    end
    return max_err
end

function estimate_interpolation_error(
        f,
        interval::NInterval{N, Float64},
        P::LagrangePolynomials{Float64}
    )::Float64 where {N}
    values = _evalf(f, interval, P)
    lens = intervallength(interval)
    NP = length(P.grid)

    test_points = 0.5 * (1.0 .- cospi.((0:(2 * NP - 1)) / (2 * NP - 1)))

    max_err = 0.0
    for tinds in Iterators.product(ntuple(_ -> 1:length(test_points), N)...)
        ts = tuple((test_points[i] for i in tinds)...)
        x = tuple((interval.start[n] + lens[n] * ts[n] for n in 1:N)...)

        interp_val = 0.0
        for αinds in Iterators.product(ntuple(_ -> 0:(NP - 1), N)...)
            basis_val = prod(P(αinds[n], ts[n]) for n in 1:N)
            interp_val += values[(αinds .+ 1)...] * basis_val
        end
        max_err = max(max_err, abs(interp_val - f(x...)))
    end
    return max_err
end

function detect_dangerous_intervals!(
        dangerous_at_level::Vector{Vector{NInterval{N, Float64}}},
        f,
        interval::NInterval{N, Float64},
        level::Int,
        maxlevel::Int,
        P::LagrangePolynomials{Float64},
        tol::Float64
    ) where {N}
    if level > maxlevel
        return
    end

    err = estimate_interpolation_error(f, interval, P)

    return if err > tol
        push!(dangerous_at_level[level], interval)
        for subint in split(interval)
            detect_dangerous_intervals!(dangerous_at_level, f, subint, level + 1, maxlevel, P, tol)
        end
    end
end

function is_dangerous(
        interval::NInterval{N, Float64},
        dangerous_list::Vector{NInterval{N, Float64}}
    ) where {N}
    for di in dangerous_list
        if all(abs.(di.start .- interval.start) .< 1.0e-14) &&
                all(abs.(di.stop .- interval.stop) .< 1.0e-14)
            return true
        end
    end
    return false
end

function find_dangerous_index(
        interval::NInterval{N, Float64},
        dangerous_list::Vector{NInterval{N, Float64}}
    )::Int where {N}
    for (i, di) in enumerate(dangerous_list)
        if all(abs.(di.start .- interval.start) .< 1.0e-14) &&
                all(abs.(di.stop .- interval.stop) .< 1.0e-14)
            return i
        end
    end
    return 0
end

function interpolateadaptive(
        f,
        a::Float64, b::Float64,
        numbits::Int,
        polynomialdegree::Int;
        tolerance = 1.0e-12,
        maxbonddim = typemax(Int),
        adaptiveTol = 1.0e-8,
        singularities::Vector{Float64} = Float64[]
    )
    return interpolateadaptive(
        f, (a,), (b,), numbits, polynomialdegree, [(s,) for s in singularities];
        tolerance = tolerance, maxbonddim = maxbonddim, adaptiveTol = adaptiveTol
    )
end

function interpolateadaptive(
        f,
        a::NTuple{N, Float64}, b::NTuple{N, Float64},
        numbits::Int,
        polynomialdegree::Int,
        singularities::Vector{NTuple{N, Float64}} = NTuple{N, Float64}[];
        tolerance = 1.0e-12,
        maxbonddim = typemax(Int),
        adaptiveTol = 1.0e-8,
        unfoldingscheme = :fused
    ) where {N}
    unfoldingscheme == :fused || error("unsupported unfolding scheme $(unfoldingscheme)")

    P = getChebyshevGrid(polynomialdegree)
    domain = NInterval{N, Float64}(a, b)

    dangerous_at_level = [NInterval{N, Float64}[] for _ in 1:numbits]

    for s in singularities
        if s in domain
            _add_singularity_path!(dangerous_at_level, domain, s, numbits)
        end
    end

    for subint in split(domain)
        detect_dangerous_intervals!(dangerous_at_level, f, subint, 1, numbits - 1, P, adaptiveTol)
    end

    if all(isempty, dangerous_at_level)
        return interpolatesinglescale(
            f, a, b, numbits, polynomialdegree;
            tolerance = tolerance, maxbonddim = maxbonddim
        )
    end

    return _build_adaptive_qtt(f, domain, numbits, P, dangerous_at_level, tolerance, maxbonddim)
end

function _add_singularity_path!(
        dangerous_at_level::Vector{Vector{NInterval{N, Float64}}},
        domain::NInterval{N, Float64},
        singularity::NTuple{N, Float64},
        numbits::Int
    ) where {N}
    interval = domain
    for level in 1:(numbits - 1)
        if !is_dangerous(interval, dangerous_at_level[level])
            push!(dangerous_at_level[level], interval)
        end

        subintervals = split(interval)
        for subint in subintervals
            if singularity in subint
                interval = subint
                break
            end
        end
    end
    return
end

function _build_adaptive_qtt(
        f,
        domain::NInterval{N, Float64},
        numbits::Int,
        P::LagrangePolynomials{Float64},
        dangerous_at_level::Vector{Vector{NInterval{N, Float64}}},
        tolerance::Float64,
        maxbonddim::Int
    ) where {N}
    polynomialdegree = length(P.grid) - 1
    Acore_ = interpolationtensor(P)
    Acore = _direct_product_coretensors(fill(Acore_, N))

    Tfirstup = zeros(2^N, (polynomialdegree + 1)^N)
    Tfirstdown = zeros(2^N, 0)
    intervallist = NInterval{N, Float64}[]

    for (i, interval) in enumerate(split(domain))
        if !is_dangerous(interval, dangerous_at_level[1])
            Tfirstup[i, :] = _evalf(f, interval, P)
        else
            Tfirstdown = [Tfirstdown;; indicator((2^N,), (i,))]
            push!(intervallist, interval)
        end
    end

    Tfirst = [Tfirstup;; Tfirstdown]
    train = [reshape(Tfirst, 1, 2^N, size(Tfirst, 2))]

    for ell in 2:(numbits - 1)
        newintervallist = NInterval{N, Float64}[]
        qell = length(intervallist)

        if qell == 0
            push!(train, Acore)
            continue
        end

        F = zeros(qell, 2^N, (polynomialdegree + 1)^N)
        χ = zeros(qell, 2^N, 0)

        for (i, interval) in enumerate(intervallist)
            for (s, subinterval) in enumerate(split(interval))
                if !is_dangerous(subinterval, dangerous_at_level[ell])
                    F[i, s, :] = _evalf(f, subinterval, P)
                else
                    χ = [χ;;; indicator((qell, 2^N, 1), (i, s, 1))]
                    push!(newintervallist, subinterval)
                end
            end
        end

        Ak = [Acore; F;;; zeros((polynomialdegree + 1)^N, 2^N, size(χ, 3)); χ]
        push!(train, Ak)
        intervallist = newintervallist
    end

    Aright_ = [P(alpha, sigma / 2) for alpha in 0:polynomialdegree, sigma in [0, 1]]
    Aright = _direct_product_coretensors(fill(reshape(Aright_, size(Aright_)..., 1), N))

    if isempty(intervallist)
        push!(train, reshape(Aright, size(Aright, 1), 2^N, 1))
    else
        F = zeros(length(intervallist), 2^N)
        for (i, interval) in enumerate(intervallist)
            for (s, subinterval) in enumerate(split(interval))
                F[i, s] = f(subinterval.start...)
            end
        end
        Alast = [Aright; F]
        push!(train, reshape(Alast, size(Alast, 1), 2^N, 1))
    end

    return _compress_train(train, tolerance, maxbonddim)
end
