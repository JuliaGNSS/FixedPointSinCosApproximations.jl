@inline function get_first_bit_sign(x::VInt16, ::Val{N}) where N
    mysign(x << (16 - N - 1))
end
@inline function get_second_bit_sign(x::VInt16, ::Val{N}) where N
    mysign(x << (16 - N - 2))
end
@inline function get_first_bit_sign(x::VInt32, ::Val{N}) where N
    mysign(x << (32 - N - 1))
end
@inline function get_second_bit_sign(x::VInt32, ::Val{N}) where N
    mysign(x << (32 - N - 2))
end
@inline function get_quarter_angle(x, ::Val{N}) where N
    x & (one(x) << N - one(x)) - one(x) << (N - 1)
end
@inline mysign(x) = vifelse(x >= zero(x), one(x), -one(x))

"""
$(SIGNATURES)

Fixed point cos
"""
@inline function fpcos(phase::P, bits::Val{N}) where {N, P <: Union{VInt16, VInt32}}
    first_bit_sign = get_first_bit_sign(phase, bits)
    second_bit_sign = get_second_bit_sign(phase, bits)
    quarter_angle = get_quarter_angle(phase, bits)
    A = calc_A(quarter_angle, bits)
    B = calc_B(quarter_angle, bits)

    second_bit_sign * (first_bit_sign * A + B)
end

"""
$(SIGNATURES)

Fixed point sin
"""
@inline function fpsin(phase::P, bits::Val{N}) where {N, P <: Union{VInt16, VInt32}}
    first_bit_sign = get_first_bit_sign(phase, bits)
    second_bit_sign = get_second_bit_sign(phase, bits)
    quarter_angle = get_quarter_angle(phase, bits)
    A = calc_A(quarter_angle, bits)
    B = calc_B(quarter_angle, bits)

    second_bit_sign * (A - first_bit_sign * B)
end

"""
$(SIGNATURES)

Fixed point sin and cos
"""
@inline function fpsincos(x::P, bits::Val{N}) where {N, P <: Union{VInt16, VInt32}}
    (fpsin(x, bits), fpcos(x, bits))
end
