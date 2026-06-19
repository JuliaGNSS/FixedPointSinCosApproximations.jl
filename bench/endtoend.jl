using Pkg
Pkg.activate(@__DIR__)
# Set up deps on first run (FastSinCos is unregistered -> add from GitHub).
try
    @eval using FastSinCos
catch
    Pkg.develop(path = dirname(@__DIR__))
    Pkg.add(["BenchmarkTools", "SIMD"])
    Pkg.add(url = "https://github.com/JuliaGNSS/FastSinCos.jl")
end
using FixedPointSinCosApproximations, FastSinCos, BenchmarkTools, SIMD, Printf

# End-to-end carrier generation: phase computation + sincos, fully vectorised.
# Phase is accumulated drift-free in Int32 (delta carries `fp` fractional bits)
# and narrowed to the sincos argument at the call. Two strategies for advancing
# the phase: a per-iteration multiply, and an incremental accumulator (one add).
# NOTE: build lane-index vectors with ntuple(f, Val(W)) — ntuple(f, W) for W>10
# hits a type-unstable fallback that ~doubles the kernel time.

const L    = 2048
const freq = 0.002
const fp   = 16
delta_i(n) = floor(Int32, freq * (1 << (n + 2 + fp)))
const STEP_F = Float32(2pi * freq)

iota(::Val{W}) where W = Vec{W,Int32}(ntuple(j->Int32(j-1), Val(W)))

# ---- Fixed point, incremental accumulator (Int16 sincos, W lanes) ----
function fp16_inc!(re, im, delta::Int32, ::Val{n}, ::Val{W}) where {n,W}
    acc  = iota(Val(W)) * delta
    step = Int32(W) * delta
    @inbounds for k in 0:W:L-1
        s, c = fpsincos(convert(Vec{W,Int16}, acc >> fp), Val(n))
        lane = VecRange{W}(k+1); im[lane]=s; re[lane]=c
        acc += step
    end
end
# ---- Fixed point, incremental accumulator (Int32 sincos, W lanes) ----
function fp32_inc!(re, im, delta::Int32, ::Val{n}, ::Val{W}) where {n,W}
    acc  = iota(Val(W)) * delta
    step = Int32(W) * delta
    @inbounds for k in 0:W:L-1
        s, c = fpsincos(acc >> fp, Val(n))
        lane = VecRange{W}(k+1); im[lane]=s; re[lane]=c
        acc += step
    end
end
# ---- Float, incremental accumulator ----
function fl_inc!(f, re, im, step::Float32)
    acc = Vec{16,Float32}(ntuple(j->Float32(j-1), Val(16))) * step
    inc = Float32(16) * step
    @inbounds for k in 0:16:L-1
        s, c = f(acc)
        lane = VecRange{16}(k+1); im[lane]=s; re[lane]=c
        acc += inc
    end
end

# validate accuracy vs true cos(2π·freq·i)
function check()
    r16=zeros(Int16,L);i16=zeros(Int16,L); fp16_inc!(r16,i16,delta_i(7),Val(7),Val(32))
    r32=zeros(Int32,L);i32=zeros(Int32,L); fp32_inc!(r32,i32,delta_i(8),Val(8),Val(16))
    rf=zeros(Float32,L);imf=zeros(Float32,L); fl_inc!(fast_sincos_u35,rf,imf,STEP_F)
    e16=maximum(abs(r16[i]/128 - cos(2pi*freq*(i-1))) for i in 1:L)
    e32=maximum(abs(r32[i]/256 - cos(2pi*freq*(i-1))) for i in 1:L)
    ef =maximum(abs(rf[i]      - cos(2pi*freq*(i-1))) for i in 1:L)
    @printf("max |cos error|:  Int16=%.4f  Int32=%.4f  float(u35)=%.6f\n", e16, e32, ef)
end
check()

mn(b)=minimum(b).time
row(n,t)=@printf("  %-36s %7.1f ns   %5.1f ps/elem\n", n, t, t/L*1000)
r16=zeros(Int16,L);i16=zeros(Int16,L); r32=zeros(Int32,L);i32=zeros(Int32,L); rf=zeros(Float32,L);imf=zeros(Float32,L)
println("\nEnd-to-end carrier (phase + sincos), L=$L, incremental phase:")
row("fixed: Int32 phase + Int16 sincos", mn(@benchmark fp16_inc!($r16,$i16,$(delta_i(7)),$(Val(7)),$(Val(32)))))
row("fixed: Int32 phase + Int32 sincos", mn(@benchmark fp32_inc!($r32,$i32,$(delta_i(8)),$(Val(8)),$(Val(16)))))
row("float: F32 phase + u100k",          mn(@benchmark fl_inc!($fast_sincos_u100k,$rf,$imf,$STEP_F)))
row("float: F32 phase + u3500",          mn(@benchmark fl_inc!($fast_sincos_u3500,$rf,$imf,$STEP_F)))
row("float: F32 phase + u35",            mn(@benchmark fl_inc!($fast_sincos_u35,$rf,$imf,$STEP_F)))
