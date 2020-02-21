@inline function calc_A_fourth_order(x::VInt32, ::Val{N}) where N
    p = 31; r = 3; q = 31; A = Int32(1518500249); C = Int32(-1873374594);
    E = Int32(3018908474 >> N)
    rounding = one(x) << (q - N - 1)
    #(rounding + A + (x * (x * (C + (x * (x * E) >> N) >> r) >> N) >> N) >> (p - q)) >> (q - N)
    x² = (x * x) >> N
    (rounding + A + (x² * (C + (x² * E) >> r) >> N) >> (p - q)) >> (q - N)
end

@inline function calc_B_fourth_order(x::VInt32, ::Val{N}) where N
    p = 30; r = 3; B = Int32(-1192627308); D = Int32(3804335470 >> N);
    rounding = one(x) << (p - N - 1)
    #(rounding + x * (B + (x * (x * D) >> N) >> r) >> N) >> (p - N)
    x² = (x * x) >> N
    (rounding + x * (B + (x² * D) >> r) >> N) >> (p - N)
end

@inline function calc_A(x::VInt32, bits::B) where B <: Union{Val{9}, Val{10}}
    calc_A_fourth_order(x, bits)
end
@inline function calc_B(x::VInt32, bits::B) where B <: Union{Val{9}, Val{10}}
    calc_B_fourth_order(x, bits)
end
