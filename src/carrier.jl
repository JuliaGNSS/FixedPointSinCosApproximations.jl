# Drift-free SIMD carrier generation on top of `fpsincos`.
#
# The phase is carried as an exact integer DDA: the phase for sample `i` is
# `div(i * step_numerator, step_denominator)` (plus an offset), never a binary
# fraction, so it never drifts even when the per-sample step is irrational. `fpsincos`
# folds its argument modulo `2^(N+2)` internally (pure bit masking), so the natural
# `T`-width integer overflow of the accumulated phase is benign and no explicit mask is
# needed.
#
# Unlike a LUT, `fpsincos` is plain integer arithmetic + shifts, which SIMD.jl lowers to
# whatever the host supports (AVX-512 / AVX2 / NEON / scalar) — so there is a single
# portable code path and no CPU/backend dispatch.
#
# Type parameters used throughout: `T` = output element type (Int16 / Int32), `N` =
# approximation bits (amplitude `2^N`, full cycle `2^(N+2)` phase units), `U` =
# unsigned(T) (remainder type), `W` = SIMD lane count.

using SIMD: Vec, VecRange, vifelse

export generate_carrier!, generate_carrier, generate_carrier4, cycles_per_sample

# Phase units in one full cycle: a quarter cycle is `2^N` units, so a full cycle is
# `2^(N+2)` (the extra two bits select the quadrant inside `fpsincos`).
@inline phase_units_per_cycle(::Val{N}) where {N} = 1 << (N + 2)

# Default SIMD width: one 512-bit register's worth of the element type. SIMD.jl splits
# this into narrower registers on AVX2 / NEON automatically.
@inline default_lanes(::Type{Int16}) = Val(32)
@inline default_lanes(::Type{Int32}) = Val(16)

"""
    cycles_per_sample(frequency, sampling_frequency)

Normalised frequency `frequency / sampling_frequency` (cycles per sample) — pass the
result as the frequency argument of [`generate_carrier!`](@ref), [`generate_carrier`](@ref)
or [`generate_carrier4`](@ref), e.g.
`generate_carrier!(sin_out, cos_out, Val(13), cycles_per_sample(1000, 2e6))`.
"""
cycles_per_sample(frequency, sampling_frequency) = frequency / sampling_frequency

# Convert an initial `phase` to phase units: an `Integer` is exact phase units, a `Real`
# is cycles (a fraction of a full cycle).
@inline _phase_units(phase::Integer, ::Val{N}) where {N} = Int(phase)
@inline _phase_units(phase::Real, bits::Val{N}) where {N} = round(Int, phase * phase_units_per_cycle(bits))

# ===== generate_carrier! (fill arrays) =====
"""
    generate_carrier!(sin_out, cos_out, bits::Val{N}, step_numerator, step_denominator; phase=0, lanes=…)
    generate_carrier!(sin_out, cos_out, bits::Val{N}, cycles_per_sample::Real;          phase=0, lanes=…)
    generate_carrier!(sin_out, cos_out, bits::Val{N}; frequency, sampling_frequency,    phase=0, lanes=…)

Fill `sin_out`/`cos_out` (element type `Int16` or `Int32`) with an `N`-bit fixed-point
carrier (amplitude `2^N`): `sin_out[n] ≈ 2^N·sin(2π·cycles_per_sample·n)` and likewise
`cos_out`. The phase advances by an exact `step_numerator / step_denominator` phase-units
per sample via a drift-free integer DDA (`2^(N+2)` phase units = one full cycle).
`cycles_per_sample` is the normalised frequency `f/fs`; its step is
`cycles_per_sample · 2^(N+2)`, rationalised internally. The third form takes `frequency`
and `sampling_frequency` directly. `phase` is the initial carrier phase (default 0): an
`Integer` is **phase units** (exact), a `Real` is **cycles**. `lanes` is the SIMD width
as a `Val` (default `Val(32)` for `Int16`, `Val(16)` for `Int32`). Requires
`0 < step_denominator ≤ typemax(T)`.
"""
function generate_carrier!(sin_out::AbstractVector{T}, cos_out::AbstractVector{T},
                           bits::Val{N}, step_numerator::Integer, step_denominator::Integer;
                           phase::Real = 0, lanes::Val = default_lanes(T)) where {T<:Union{Int16,Int32},N}
    length(sin_out) == length(cos_out) || throw(DimensionMismatch("sin/cos lengths differ"))
    (0 < step_denominator ≤ typemax(T)) ||
        throw(ArgumentError("need 0 < step_denominator ≤ typemax($T) = $(typemax(T))"))
    _generate!(sin_out, cos_out, bits, Int(step_numerator), Int(step_denominator),
               _phase_units(phase, bits) % T, lanes, unsigned(T))
    sin_out, cos_out
end
function generate_carrier!(sin_out::AbstractVector{T}, cos_out::AbstractVector{T},
                           bits::Val{N}, normalised_frequency::Real; kw...) where {T<:Union{Int16,Int32},N}
    ratio = rationalize(Int, normalised_frequency * phase_units_per_cycle(bits); tol = 1 / (1 << 20))
    generate_carrier!(sin_out, cos_out, bits, numerator(ratio), denominator(ratio); kw...)
end
function generate_carrier!(sin_out::AbstractVector{T}, cos_out::AbstractVector{T}, bits::Val{N};
                           frequency::Real, sampling_frequency::Real, kw...) where {T<:Union{Int16,Int32},N}
    generate_carrier!(sin_out, cos_out, bits, cycles_per_sample(frequency, sampling_frequency); kw...)
end

# Initialise one DDA state of W lanes whose first lane is sample `start_sample`:
#   phase[j]     = div(step_num*(start_sample+j), step_den) + phase_offset   (mod 2^width(T))
#   remainder[j] = mod(step_num*(start_sample+j), step_den),  j = 0..W-1
# Per-lane reductions use a multiplicative inverse (mul+shift, not idiv) and are
# independent across lanes, so they pipeline.
@inline function _init_state(::Val{W}, ::Type{T}, ::Type{U},
                             step_num, den_inverse, step_den, start_sample, phase_offset) where {W,T,U}
    phase     = Vec{W,T}(ntuple(j -> (div(step_num * (start_sample + j - 1), den_inverse) % T + phase_offset), Val(W)))
    remainder = Vec{W,U}(ntuple(j -> (n = step_num * (start_sample + j - 1); U(n - div(n, den_inverse) * step_den)), Val(W)))
    (phase, remainder)
end

# 4 independent DDA states run interleaved so their loop-carried carry chains
# (add→compare→blend→add) overlap instead of stalling. Hand-unrolled with plain locals —
# a tuple/closure formulation boxes the reassigned state and is far slower.
function _generate!(sin_out::AbstractVector{T}, cos_out, bits::Val{N},
                    step_num, step_den, phase_offset, ::Val{W}, ::Type{U}) where {T,N,W,U}
    stride = 4W
    den_inverse = Base.SignedMultiplicativeInverse(step_den)
    whole_step = div(stride * step_num, step_den) % T       # whole phase-units advanced per stride
    frac_step  = U(mod(stride * step_num, step_den))         # fractional remainder advanced per stride
    modulus    = U(step_den)
    one_T      = one(T)
    phase1, rem1 = _init_state(Val(W), T, U, step_num, den_inverse, step_den, 0,  phase_offset)
    phase2, rem2 = _init_state(Val(W), T, U, step_num, den_inverse, step_den, W,  phase_offset)
    phase3, rem3 = _init_state(Val(W), T, U, step_num, den_inverse, step_den, 2W, phase_offset)
    phase4, rem4 = _init_state(Val(W), T, U, step_num, den_inverse, step_den, 3W, phase_offset)
    num_samples = length(sin_out); chunk_start = 0
    @inbounds while chunk_start + stride <= num_samples
        # store each result immediately (keep ≤2 result vectors live); the four
        # fpsincos evaluations are independent so out-of-order execution overlaps them.
        sin1, cos1 = fpsincos(phase1, bits)
        sin_out[VecRange{W}(chunk_start + 1)]      = sin1; cos_out[VecRange{W}(chunk_start + 1)]      = cos1
        sin2, cos2 = fpsincos(phase2, bits)
        sin_out[VecRange{W}(chunk_start + W + 1)]  = sin2; cos_out[VecRange{W}(chunk_start + W + 1)]  = cos2
        sin3, cos3 = fpsincos(phase3, bits)
        sin_out[VecRange{W}(chunk_start + 2W + 1)] = sin3; cos_out[VecRange{W}(chunk_start + 2W + 1)] = cos3
        sin4, cos4 = fpsincos(phase4, bits)
        sin_out[VecRange{W}(chunk_start + 3W + 1)] = sin4; cos_out[VecRange{W}(chunk_start + 3W + 1)] = cos4
        rem1 += frac_step; rem2 += frac_step; rem3 += frac_step; rem4 += frac_step
        carry1 = rem1 >= modulus; carry2 = rem2 >= modulus; carry3 = rem3 >= modulus; carry4 = rem4 >= modulus
        rem1 = vifelse(carry1, rem1 - modulus, rem1); rem2 = vifelse(carry2, rem2 - modulus, rem2)
        rem3 = vifelse(carry3, rem3 - modulus, rem3); rem4 = vifelse(carry4, rem4 - modulus, rem4)
        phase1 += whole_step; phase2 += whole_step; phase3 += whole_step; phase4 += whole_step
        phase1 = vifelse(carry1, phase1 + one_T, phase1); phase2 = vifelse(carry2, phase2 + one_T, phase2)
        phase3 = vifelse(carry3, phase3 + one_T, phase3); phase4 = vifelse(carry4, phase4 + one_T, phase4)
        chunk_start += stride
    end
    _generate_tail!(sin_out, cos_out, bits, step_num, step_den, phase_offset, chunk_start + 1)
end

@inline function _generate_tail!(sin_out::AbstractVector{T}, cos_out, bits::Val{N},
                                 step_num, step_den, phase_offset, sample) where {T,N}
    @inbounds while sample <= length(sin_out)
        phase = (div(step_num * (sample - 1), step_den) % T + phase_offset)
        s, c = fpsincos(phase, bits)
        sin_out[sample] = s; cos_out[sample] = c
        sample += 1
    end
    sin_out, cos_out
end

# ===== stateful iterator: drift-free phase, yields (sin, cos) Vecs (no array) =====
"""
    CarrierIterator

Iterator returned by [`generate_carrier`](@ref). Yields `(sin, cos)::Tuple{Vec{W,T},Vec{W,T}}`.
"""
struct CarrierIterator{T,N,U,W}
    phase_init::Vec{W,T}
    remainder_init::Vec{W,U}
    whole_step::T
    frac_step::U
    modulus::U
    num_chunks::Int
end

"""
    generate_carrier(bits::Val{N}, step_numerator, step_denominator, num_samples; phase=0, lanes=…, type=Int16)
    generate_carrier(bits::Val{N}, cycles_per_sample::Real,          num_samples; phase=0, lanes=…, type=Int16)
    generate_carrier(bits::Val{N}, num_samples; frequency, sampling_frequency,    phase=0, lanes=…, type=Int16)

Allocation-free iterator over `num_samples ÷ W` chunks (W = `lanes`), each yielding
`(sin, cos)::Tuple{Vec{W,T},Vec{W,T}}` — an `N`-bit fixed-point carrier (amplitude
`2^N`, element type `type`, default `Int16`). Phase advances by an exact
`step_numerator / step_denominator` (or `cycles_per_sample · 2^(N+2)`) phase-units per
sample via a drift-free DDA held in the iteration state — no carrier array is allocated.
Fuse it straight into your own loop (the analogue of `FastSinCos`'s `fast_sincos(::Vec)`,
but a fixed-point kernel):

    accumulator = zero(Int32)
    for (sin_vec, cos_vec) in generate_carrier(Val(13), 0.002, length(signal))
        accumulator += reduce_into_registers(sin_vec, cos_vec, …)   # Vec{W,Int16}
    end

`phase` is the initial carrier phase (default 0): `Integer` = phase units, `Real` =
cycles. Requires `0 < step_denominator ≤ typemax(type)`. Any leftover `num_samples % W`
tail is not produced (handle it yourself if needed).
"""
function generate_carrier(bits::Val{N}, step_numerator::Integer, step_denominator::Integer,
                          num_samples::Integer; phase::Real = 0, type::Type{T} = Int16,
                          lanes::Val = default_lanes(T)) where {N,T<:Union{Int16,Int32}}
    (0 < step_denominator ≤ typemax(T)) ||
        throw(ArgumentError("need 0 < step_denominator ≤ typemax($T) = $(typemax(T))"))
    _make_carrier(bits, Int(step_numerator), Int(step_denominator), Int(num_samples),
                  _phase_units(phase, bits) % T, lanes, T, unsigned(T))
end
function generate_carrier(bits::Val{N}, normalised_frequency::Real, num_samples::Integer; kw...) where {N}
    ratio = rationalize(Int, normalised_frequency * phase_units_per_cycle(bits); tol = 1 / (1 << 20))
    generate_carrier(bits, numerator(ratio), denominator(ratio), num_samples; kw...)
end
function generate_carrier(bits::Val{N}, num_samples::Integer;
                          frequency::Real, sampling_frequency::Real, kw...) where {N}
    generate_carrier(bits, cycles_per_sample(frequency, sampling_frequency), num_samples; kw...)
end

function _make_carrier(bits::Val{N}, step_num, step_den, num_samples, phase_offset,
                       ::Val{W}, ::Type{T}, ::Type{U}) where {N,W,T,U}
    den_inverse = Base.SignedMultiplicativeInverse(step_den)
    phase_init, remainder_init = _init_state(Val(W), T, U, step_num, den_inverse, step_den, 0, phase_offset)
    CarrierIterator{T,N,U,W}(phase_init, remainder_init,
        div(W * step_num, step_den) % T, U(mod(W * step_num, step_den)), U(step_den), num_samples ÷ W)
end

Base.length(it::CarrierIterator) = it.num_chunks
Base.IteratorSize(::Type{<:CarrierIterator}) = Base.HasLength()
Base.eltype(::Type{<:CarrierIterator{T,N,U,W}}) where {T,N,U,W} = Tuple{Vec{W,T},Vec{W,T}}

@inline function Base.iterate(it::CarrierIterator{T,N,U,W},
                              state = (it.phase_init, it.remainder_init, 0)) where {T,N,U,W}
    phase, remainder, chunk = state
    chunk >= it.num_chunks && return nothing
    result = fpsincos(phase, Val(N))
    remainder += it.frac_step
    carry = remainder >= it.modulus
    remainder = vifelse(carry, remainder - it.modulus, remainder)
    phase = vifelse(carry, phase + it.whole_step + one(T), phase + it.whole_step)
    (result, (phase, remainder, chunk + 1))
end

# ===== 4-way interleaved iterator: yields 4 (sin,cos) pairs (4W samples) per step =====
"""
    CarrierIterator4

Iterator returned by [`generate_carrier4`](@ref). Yields `NTuple{4,Tuple{Vec{W,T},Vec{W,T}}}`.
"""
struct CarrierIterator4{T,N,U,W}
    phase1::Vec{W,T}; phase2::Vec{W,T}; phase3::Vec{W,T}; phase4::Vec{W,T}
    rem1::Vec{W,U};   rem2::Vec{W,U};   rem3::Vec{W,U};   rem4::Vec{W,U}
    whole_step::T
    frac_step::U
    modulus::U
    num_steps::Int
end

"""
    generate_carrier4(bits::Val{N}, step_numerator, step_denominator, num_samples; phase=0, lanes=…, type=Int16)
    generate_carrier4(bits::Val{N}, cycles_per_sample, num_samples;                phase=0, lanes=…, type=Int16)
    generate_carrier4(bits::Val{N}, num_samples; frequency, sampling_frequency,    phase=0, lanes=…, type=Int16)

Like [`generate_carrier`](@ref) but yields a 4-tuple of `(sin, cos)` `Vec` pairs per step
(`4W` samples), running four interleaved DDA states so the carry chains overlap. Reaches
the full loop throughput even for trivial consumers such as array fill. **Destructure the
4-tuple in the loop header** —
`for ((s0,c0),(s1,c1),(s2,c2),(s3,c3)) in generate_carrier4(...)` — rather than iterating
it with an inner `for pair in quad` loop, which does not unroll and is much slower.
Produces `num_samples ÷ (4W)` steps; handle any tail yourself.
"""
function generate_carrier4(bits::Val{N}, step_numerator::Integer, step_denominator::Integer,
                           num_samples::Integer; phase::Real = 0, type::Type{T} = Int16,
                           lanes::Val = default_lanes(T)) where {N,T<:Union{Int16,Int32}}
    (0 < step_denominator ≤ typemax(T)) ||
        throw(ArgumentError("need 0 < step_denominator ≤ typemax($T) = $(typemax(T))"))
    _make_carrier4(bits, Int(step_numerator), Int(step_denominator), Int(num_samples),
                   _phase_units(phase, bits) % T, lanes, T, unsigned(T))
end
function generate_carrier4(bits::Val{N}, normalised_frequency::Real, num_samples::Integer; kw...) where {N}
    ratio = rationalize(Int, normalised_frequency * phase_units_per_cycle(bits); tol = 1 / (1 << 20))
    generate_carrier4(bits, numerator(ratio), denominator(ratio), num_samples; kw...)
end
function generate_carrier4(bits::Val{N}, num_samples::Integer;
                           frequency::Real, sampling_frequency::Real, kw...) where {N}
    generate_carrier4(bits, cycles_per_sample(frequency, sampling_frequency), num_samples; kw...)
end

function _make_carrier4(bits::Val{N}, step_num, step_den, num_samples, phase_offset,
                        ::Val{W}, ::Type{T}, ::Type{U}) where {N,W,T,U}
    den_inverse = Base.SignedMultiplicativeInverse(step_den)
    phase1, rem1 = _init_state(Val(W), T, U, step_num, den_inverse, step_den, 0,  phase_offset)
    phase2, rem2 = _init_state(Val(W), T, U, step_num, den_inverse, step_den, W,  phase_offset)
    phase3, rem3 = _init_state(Val(W), T, U, step_num, den_inverse, step_den, 2W, phase_offset)
    phase4, rem4 = _init_state(Val(W), T, U, step_num, den_inverse, step_den, 3W, phase_offset)
    CarrierIterator4{T,N,U,W}(phase1, phase2, phase3, phase4, rem1, rem2, rem3, rem4,
        div(4W * step_num, step_den) % T, U(mod(4W * step_num, step_den)), U(step_den), num_samples ÷ (4W))
end

Base.length(it::CarrierIterator4) = it.num_steps
Base.IteratorSize(::Type{<:CarrierIterator4}) = Base.HasLength()
Base.eltype(::Type{<:CarrierIterator4{T,N,U,W}}) where {T,N,U,W} = NTuple{4,Tuple{Vec{W,T},Vec{W,T}}}

@inline function Base.iterate(it::CarrierIterator4{T,N,U,W},
                              state = (it.phase1, it.phase2, it.phase3, it.phase4,
                                       it.rem1, it.rem2, it.rem3, it.rem4, 0)) where {T,N,U,W}
    phase1, phase2, phase3, phase4, rem1, rem2, rem3, rem4, step_count = state
    step_count >= it.num_steps && return nothing
    pair1 = fpsincos(phase1, Val(N)); pair2 = fpsincos(phase2, Val(N))
    pair3 = fpsincos(phase3, Val(N)); pair4 = fpsincos(phase4, Val(N))
    whole_step = it.whole_step; frac_step = it.frac_step; modulus = it.modulus
    # advance into fresh variables (reassigning the loop-carried state in place pessimises codegen)
    acc1 = rem1 + frac_step; carry1 = acc1 >= modulus
    next_rem1   = vifelse(carry1, acc1 - modulus, acc1)
    next_phase1 = vifelse(carry1, phase1 + whole_step + one(T), phase1 + whole_step)
    acc2 = rem2 + frac_step; carry2 = acc2 >= modulus
    next_rem2   = vifelse(carry2, acc2 - modulus, acc2)
    next_phase2 = vifelse(carry2, phase2 + whole_step + one(T), phase2 + whole_step)
    acc3 = rem3 + frac_step; carry3 = acc3 >= modulus
    next_rem3   = vifelse(carry3, acc3 - modulus, acc3)
    next_phase3 = vifelse(carry3, phase3 + whole_step + one(T), phase3 + whole_step)
    acc4 = rem4 + frac_step; carry4 = acc4 >= modulus
    next_rem4   = vifelse(carry4, acc4 - modulus, acc4)
    next_phase4 = vifelse(carry4, phase4 + whole_step + one(T), phase4 + whole_step)
    ((pair1, pair2, pair3, pair4),
     (next_phase1, next_phase2, next_phase3, next_phase4, next_rem1, next_rem2, next_rem3, next_rem4, step_count + 1))
end
