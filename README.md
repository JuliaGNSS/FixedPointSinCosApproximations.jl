# FixedPointSinCosApproximations

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliagnss.github.io/FixedPointSinCosApproximations.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliagnss.github.io/FixedPointSinCosApproximations.jl/dev)
[![Build Status](https://travis-ci.org/JuliaGNSS/FixedPointSinCosApproximations.jl.svg?branch=master)](https://travis-ci.org/JuliaGNSS/FixedPointSinCosApproximations.jl)
[![Coveralls](https://coveralls.io/repos/github/JuliaGNSS/FixedPointSinCosApproximations.jl/badge.svg?branch=master)](https://coveralls.io/github/JuliaGNSS/FixedPointSinCosApproximations.jl?branch=master)

## Getting started

Install:
```julia
julia> ]
pkg> add FixedPointSinCosApproximations
```

Usage:
```julia
using FixedPointSinCosApproximations, StructArrays
function generate_carrier!(carrier, doppler, sampling_frequency, num_samples, bits::Val{B} = Val(8)) where B
    sampling_frequency_i32 = Base.SignedMultiplicativeInverse(floor(Int32, sampling_frequency))
    scaled_doppler_i32 = floor(Int32, 2π * doppler * 1 << B / π * 2)
    @inbounds for i = Int32(1):Int32(num_samples)
        carrier.im[i], carrier.re[i] = fpsincos(div(scaled_doppler_i32 * i, sampling_frequency_i32), bits)
    end
end

carrier = StructArray(zeros(Complex{Int32}, 2000))

generate_carrier!(carrier, 1000, 2e6, 2000)

using Plots
plot(real.(carrier))
plot!(real.(cos.(2π * 1000 / 2e6 * (1:2000))) * 1 << 8)

using BenchmarkTools
@btime generate_carrier!($carrier, 1000, 2e6, 2000)
    # 1.467 μs (0 allocations: 0 bytes)

carrier_reference = zeros(ComplexF32, 2000)
@btime $carrier_reference .= cis.(2π * 1000 / 2e6 * (1:2000))
    #18.578 μs (0 allocations: 0 bytes)
```

For more speed separate carrier and phase generation:

```julia
function generate_carrier!(phases, carrier, doppler, sampling_frequency, num_samples, bits::Val{B} = Val(8)) where B
    sampling_frequency_i32 = Base.SignedMultiplicativeInverse(floor(Int32, sampling_frequency))
    scaled_doppler_i32 = floor(Int32, 2π * doppler * 1 << B / π * 2)
    @inbounds for i = Int32(1):Int32(num_samples)
        phases[i] = div(scaled_doppler_i32 * i, sampling_frequency_i32)
    end
    @inbounds for i = 1:num_samples
        carrier.im[i], carrier.re[i] = fpsincos(phases[i], bits)
    end
end

phases = zeros(Int32, 2000)

@btime generate_carrier!($phases, $carrier, 1000, 2e6, 2000)
    # 1.034 μs (0 allocations: 0 bytes)
```

## Explicit SIMD with SIMD.jl

`fpsin`, `fpcos` and `fpsincos` accept [SIMD.jl](https://github.com/eschnett/SIMD.jl)
`Vec`s, so you can vectorise the carrier loop explicitly. Because the integer
approximations carry intermediate products in `Int32` only for the higher bit
depths, the lower bit depths use `Int16` — which packs **twice as many lanes per
SIMD register** (e.g. 32×`Int16` vs 16×`Int32` on AVX-512). Pick the lane width
to match your element type:

```julia
using FixedPointSinCosApproximations, SIMD

function fpsincos!(re, im, phases::AbstractVector{T}, bits::Val{B}, ::Val{W}) where {T, B, W}
    @inbounds for i in 1:W:length(phases)
        lane = VecRange{W}(i)
        s, c = fpsincos(phases[lane], bits)
        im[lane] = s
        re[lane] = c
    end
end

# Int16 path (≤ 7 quarter bits) — 32 lanes wide
phases = rand(Int16(-1 << 13):Int16(1 << 13), 16384)
re = similar(phases); im = similar(phases)
fpsincos!(re, im, phases, Val(7), Val(32))

# Int32 path (8–14 quarter bits) — 16 lanes wide
phases32 = rand(Int32(-1 << 20):Int32(1 << 20), 16384)
re32 = similar(phases32); im32 = similar(phases32)
fpsincos!(re32, im32, phases32, Val(8), Val(16))
```

On an AVX-512 machine this runs the kernel at roughly **0.1 ns/sample** (`Int16`,
7 quarter bits) and **0.2 ns/sample** (`Int32`, 8 quarter bits) — faster than a
SIMD `Float32` approximation such as
[FastSinCos.jl](https://github.com/JuliaGNSS/FastSinCos.jl), since the fixed-point
range reduction is a single bit-mask rather than a Cody–Waite reduction. Run
`julia bench/proof.jl` to reproduce the comparison.

## End-to-end: phase generation + sincos

The kernel numbers above feed pre-computed phases. In practice you also pay for
generating the phase. Generate it **drift-free in `Int32`** and narrow to the
sincos argument only at the call. The cheapest way is an incremental
accumulator: hold the per-lane phase in `Int32` (with `fp` fractional bits) and
advance it by a constant `W * delta` each iteration — one vector add, no
per-iteration multiply. Keep it in registers and write straight to the output
arrays, so the only memory traffic per sample is the two output stores (sin and
cos), with zero loads.

```julia
using FixedPointSinCosApproximations, SIMD

# carrier of `freq` cycles/sample; `fp` = fractional bits of the integer phase step
function carrier!(re, im, freq, L, ::Val{n}, ::Val{W}; fp = 16) where {n, W}
    delta = floor(Int32, freq * (1 << (n + 2 + fp)))            # phase step, Int32
    acc   = Vec{W,Int32}(ntuple(j -> Int32(j - 1) * delta, Val(W)))  # phase per lane
    step  = Int32(W) * delta
    @inbounds for k in 0:W:L-1
        s, c = fpsincos(convert(Vec{W,Int16}, acc >> fp), Val(n))    # narrow at the call
        lane = VecRange{W}(k + 1)
        im[lane] = s; re[lane] = c
        acc += step                                             # advance: one add
    end
end

re = zeros(Int16, 2048); im = zeros(Int16, 2048)
carrier!(re, im, 0.002, 2048, Val(7), Val(32))   # Int16 path, 32 lanes wide
```

(Build the lane vector with `ntuple(f, Val(W))`, not `ntuple(f, W)` — the latter
hits a type-unstable fallback for `W > 10` that roughly doubles the kernel time.)

Including phase generation, the `Int32`-phase + `Int16`-sincos pipeline is the
fastest on an AVX-512 machine. Phase generation is a shared cost that narrows the
gap the sincos kernel alone opens up — enough that the all-`Int32` pipeline drops
to roughly float parity:

| pipeline (phase + sincos)                | ns/sample | max \|cos err\| |
| ---------------------------------------- | --------- | --------------- |
| `Int32` phase + `Int16` sincos (32-wide) | **0.13**  | 0.019 (~7 bit)  |
| `Float32` phase + FastSinCos `u100k`     | 0.19      | 3e-4 (~11 bit)  |
| `Float32` phase + FastSinCos `u3500`     | 0.20      | 3.6e-6          |
| `Int32` phase + `Int32` sincos (16-wide) | 0.21      | 0.009 (~8 bit)  |
| `Float32` phase + FastSinCos `u35`       | 0.22      | 1.9e-5 (~23 bit)|

So at a low-bit fixed-point target the `Int16` pipeline is fastest end-to-end
(~1.4× the fastest float); the all-`Int32` pipeline is roughly on par with float;
and float wins decisively on accuracy. Run `julia bench/endtoend.jl` to reproduce.