struct Interval{V}
    start::V
    stop::V

    function Interval{V}(start::V, stop::V) where {V}
        return new{V}(min(start, stop), max(start, stop))
    end
end

function Base.in(v::V, interval::Interval{V}) where {V}
    return v >= interval.start && v < interval.stop
end

function midpoint(interval::Interval{V}) where {V}
    return (interval.start + interval.stop) / 2
end

function split(interval::Interval{V}, value::V = midpoint(interval)) where {V}
    if value in interval
        return [Interval{V}(interval.start, value), Interval{V}(value, interval.stop)]
    else
        return [interval]
    end
end

function intervallength(interval::Interval{V}) where {V}
    return interval.stop - interval.start
end


struct NInterval{N, V}
    start::NTuple{N, V}
    stop::NTuple{N, V}

    function NInterval{N, V}(start::NTuple{N, V}, stop::NTuple{N, V}) where {N, V}
        return new{N, V}(tuple(min.(start, stop)...), tuple(max.(start, stop)...))
    end
end

function Base.in(v::NTuple{N, V}, interval::NInterval{N, V}) where {N, V}
    return all(v .>= interval.start) && all(v .< interval.stop)
end

#function midpoint(interval::NInterval{N,V})::NTuple{N,V} where {N, V}
#return tuple(((interval.start .+ interval.stop) ./ 2)...)
#end

function split(interval::NInterval{N, V}) where {N, V}
    intervals = NInterval{N, V}[]
    halfintervallength = intervallength(interval) ./ 2
    for shifts in Iterators.product(ntuple(x -> 0:1, N)...)
        start_ = interval.start .+ shifts .* halfintervallength
        push!(intervals, NInterval{N, V}(start_, start_ .+ halfintervallength))
    end
    return intervals
end

function intervallength(interval::NInterval{N, V})::NTuple{N, V} where {N, V}
    return tuple((interval.stop .- interval.start)...)
end
