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