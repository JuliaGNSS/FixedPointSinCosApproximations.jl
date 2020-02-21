@inline function calc_A_sixth(x::VInt32, ::Val{N}) where N
    p = 33; r = 5; q = 31; k = 31; A = Int32(1518500249); C = Int32(-1873374594);
    E = Int32(1540444620); G = Int32(-3966769057 >> N)
    rounding = one(x) << (q - N - 1)
    x² = (x * x) >> N
    (rounding + A + (x² * (C + (x² * (E + (x² * G) >> r) >> N) >> (p - q)) >> N) >> (q - k)) >> (k - N)
end

@inline function calc_B_sixth(x::VInt32, ::Val{N}) where N
    p = 32; r = 4; q = 30; B = Int32(-1192627308); D = Int32(1960919730);
    F = Int32(-3760127719 >> N);
    rounding = one(x) << (q - N - 1)
    x² = (x * x) >> N
    (rounding + x * (B + (x² * (D + (x² * F) >> r) >> N) >> (p - q)) >> N) >> (q - N)
end

@inline function calc_A(x::VInt32, bits::B) where B <: Union{Val{13}, Val{14}}
    calc_A_fifth_order(x, bits)
end
@inline function calc_B(x::VInt32, bits::B) where B <: Union{Val{13}, Val{14}}
    calc_B_fifth_order(x, bits)
end
