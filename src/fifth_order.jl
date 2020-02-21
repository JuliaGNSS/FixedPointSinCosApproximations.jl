@inline function calc_A_fifth_order(x::VInt32, ::Val{N}) where N
    p = 31; r = 3; q = 31; A = Int32(1518500249); C = Int32(-1871437695);
    E = Int32(2956927708 >> N)
    rounding = one(x) << (q - N - 1)
    x² = (x * x) >> N
    (rounding + A + (x² * (C + (x² * E) >> r) >> N) >> (p - q)) >> (q - N)
end

@inline function calc_B_fifth_order(x::VInt32, ::Val{N}) where N
    p = 32; r = 4; q = 30; B = Int32(-1192627308); D = Int32(1960919730);
    F = Int32(-3760127719 >> N);
    rounding = one(x) << (q - N - 1)
    x² = (x * x) >> N
    (rounding + x * (B + (x² * (D + (x² * F) >> r) >> N) >> (p - q)) >> N) >> (q - N)
end

@inline function calc_A(x::VInt32, bits::B) where B <: Union{Val{11}, Val{12}}
    calc_A_fifth_order(x, bits)
end
@inline function calc_B(x::VInt32, bits::B) where B <: Union{Val{11}, Val{12}}
    calc_B_fifth_order(x, bits)
end
