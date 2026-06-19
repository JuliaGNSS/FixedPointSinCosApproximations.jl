# Head-to-head: this package's fpsincos (on SIMD.Vec) vs FastSinCos.jl (SIMD Float32).
# Run from the repo root with:  julia benchmark/proof.jl
using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
using FixedPointSinCosApproximations, FastSinCos, BenchmarkTools, SIMD, Printf

# Uses the REAL package-exported fpsincos directly on SIMD.Vec.
fp!(::Val{W}) where W = function (re, im, p, b::Val{B}=Val(8)) where B
    @inbounds for i = 1:W:length(p)
        l = VecRange{W}(i)
        s, c = fpsincos(p[l], b)
        im[l] = s; re[l] = c
    end
end
fp32! = fp!(Val(16))   # Int32 -> 16-wide
fp16! = fp!(Val(32))   # Int16 -> 32-wide

flt!(f) = function (re, im, p, sc)
    @inbounds for i = 1:16:length(p)
        l = VecRange{16}(i)
        d = convert(Vec{16,Float32}, p[l]) * sc
        s, c = f(d); im[l] = s; re[l] = c
    end
end
fl35! = flt!(fast_sincos_u35); fl3500! = flt!(fast_sincos_u3500); fl100k! = flt!(fast_sincos_u100k)

mn(b) = minimum(b).time
N = 16384
p32 = Int32.(rand(Int32(-1<<20):Int32(1<<20), N))
p16 = Int16.(rand(Int16(-1<<13):Int16(1<<13), N))
r32=zeros(Int32,N);i32=zeros(Int32,N); r16=zeros(Int16,N);i16=zeros(Int16,N); rf=zeros(Float32,N);imf=zeros(Float32,N)
sc = Vec{16,Float32}(Float32((pi/2)/(1<<8)))

t16 = mn(@benchmark fp16!($r16,$i16,$p16,$(Val(7))))
t32 = mn(@benchmark fp32!($r32,$i32,$p32,$(Val(8))))
tk  = mn(@benchmark fl100k!($rf,$imf,$p32,$sc))
t35 = mn(@benchmark fl35!($rf,$imf,$p32,$sc))
t3500 = mn(@benchmark fl3500!($rf,$imf,$p32,$sc))

println("Package fpsincos on SIMD.Vec  vs  FastSinCos  (N=$N, lower ps/elem = faster)\n")
row(n,t,ref)=@printf("  %-28s %7.1f ns  %6.1f ps/elem   %.2fx vs u100k\n", n, t, t/N*1000, ref/t)
row("FP Int16x32 (Val 7)",  t16, tk)
row("FP Int32x16 (Val 8)",  t32, tk)
row("FastSinCos u100k",     tk,  tk)
row("FastSinCos u3500",     t3500, tk)
row("FastSinCos u35",       t35, tk)
