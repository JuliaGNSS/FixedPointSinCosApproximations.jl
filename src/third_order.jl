@inline function calc_A_third_order(x::VInt16, ::Val{N}) where N
    p = 15; r = 1; A = Int16(23170); C = Int16(-54292 >> N)
    rounding = one(x) << (p - N - 1)
    (rounding + A + (x * (x * C) >> N) >> r) >> (p - N)
end

@inline function calc_B_third_order(x::VInt16, ::Val{N}) where N
    p = 14; r = 3; B = Int16(-18199); D = Int16(58049 >> N);
    rounding = one(x) << (p - N - 1)
    (rounding + x * (B + (x * (x * D) >> N) >> r) >> N) >> (p - N)
end

@inline function calc_A_third_order(x::VInt32, ::Val{N}) where N
    p = 31; r = 1; A = Int32(1518500249); C = Int32(-3558067408 >> N)
    rounding = one(x) << (p - N - 1)
    (rounding + A + (x * (x * C) >> N) >> r) >> (p - N)
end
@inline function calc_B_third_order(x::VInt32, ::Val{N}) where N
    p = 30; r = 3; B = Int32(-1192627308); D = Int32(3804335470 >> N);
    rounding = one(x) << (p - N - 1)
    (rounding + x * (B + (x * (x * D) >> N) >> r) >> N) >> (p - N)
end

@inline function calc_A(x::VInt16, bits::B) where B <: Union{Val{6}, Val{7}}
    calc_A_third_order(x, bits)
end
@inline function calc_B(x::VInt16, bits::B) where B <: Union{Val{6}, Val{7}}
    calc_B_third_order(x, bits)
end
@inline function calc_A(x::VInt32, bits::B) where B <: Union{Val{8}}
    calc_A_third_order(x, bits)
end
@inline function calc_B(x::VInt32, bits::B) where B <: Union{Val{8}}
    calc_B_third_order(x, bits)
end
