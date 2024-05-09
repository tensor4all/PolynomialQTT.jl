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

function split(interval::Interval{V}, value::V=midpoint(interval)) where {V}
    if value in interval
        return [Interval{V}(interval.start, value), Interval{V}(value, interval.stop)]
    else
        return [interval]
    end
end

function intervallength(interval::Interval{V}) where {V}
    return interval.stop - interval.start
end
