@inline function calc_A_second_order(x::VInt16, ::Val{N}) where N
    p = 15; r = 1; A = Int16(23170); C = Int16(-54292 >> N)
    rounding = one(x) << (p - N - 1)
    (rounding + A + (x * (x * C) >> N) >> r) >> (p - N)
end

@inline function calc_B_second_order(x::VInt16, ::Val{N}) where N
    r = 14; B = Int16(-16384 >> N);
    rounding = one(x) << (r - N - 1)
    (rounding + x * B) >> (r - N)
end

@inline function calc_A(x::VInt16, bits::B) where B <: Union{Val{3}, Val{4}, Val{5}}
    calc_A_second_order(x, bits)
end
@inline function calc_B(x::VInt16, bits::B) where B <: Union{Val{3}, Val{4}, Val{5}}
    calc_B_second_order(x, bits)
end
