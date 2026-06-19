# Throughput of the drift-free carrier generators (generate_carrier! and the array-free
# iterators). Run from the repo root with:  julia benchmark/carrier.jl
using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
using FixedPointSinCosApproximations, BenchmarkTools, SIMD, Printf

const L = 16384
mn(b) = minimum(b).time
ps(t) = t / L * 1000

# Fill an array via the 4-way interleaved iterator (array-free carrier, then store).
function fill_carrier4!(sin_out, cos_out, iterator, ::Val{W}) where {W}
    sample = 1
    @inbounds for ((s0, c0), (s1, c1), (s2, c2), (s3, c3)) in iterator
        sin_out[VecRange{W}(sample)]        = s0; cos_out[VecRange{W}(sample)]        = c0
        sin_out[VecRange{W}(sample + W)]    = s1; cos_out[VecRange{W}(sample + W)]    = c1
        sin_out[VecRange{W}(sample + 2W)]   = s2; cos_out[VecRange{W}(sample + 2W)]   = c2
        sin_out[VecRange{W}(sample + 3W)]   = s3; cos_out[VecRange{W}(sample + 3W)]   = c3
        sample += 4W
    end
end

# Sum an array-free carrier without ever materialising it (the fusion use case).
function accumulate_carrier(iterator)
    acc = zero(Int32)
    @inbounds for (sin_vec, cos_vec) in iterator
        acc += Int32(sum(sin_vec)) + Int32(sum(cos_vec))
    end
    acc
end

s16 = Vector{Int16}(undef, L); c16 = similar(s16)
s32 = Vector{Int32}(undef, L); c32 = similar(s32)

# An odd denominator (1000) is the drift-prone case a binary fractional accumulator
# would mishandle; the integer DDA stays exact.
num, den = 7, 1000

generate_carrier!(s16, c16, Val(7), num, den)
generate_carrier!(s32, c32, Val(13), num, den)
make_it4()  = generate_carrier4(Val(7), num, den, L; type = Int16, lanes = Val(32))
make_iter() = generate_carrier(Val(7), num, den, L; type = Int16, lanes = Val(32))
fill_carrier4!(s16, c16, make_it4(), Val(32))
accumulate_carrier(make_iter())

t_fill16 = mn(@benchmark generate_carrier!($s16, $c16, Val(7), $num, $den))
t_fill32 = mn(@benchmark generate_carrier!($s32, $c32, Val(13), $num, $den))
t_iter4  = mn(@benchmark fill_carrier4!($s16, $c16, make_it4(), $(Val(32))))
t_acc    = mn(@benchmark accumulate_carrier(make_iter()))

@printf("Drift-free carrier generation (%d samples, %d/%d cycles/sample, lower ps/elem = faster)\n\n", L, num, den)
row(name, t) = @printf("  %-40s %7.1f ns  %6.1f ps/elem\n", name, t, ps(t))
row("generate_carrier!  Int16 Val7 (array)",  t_fill16)
row("generate_carrier!  Int32 Val13 (array)", t_fill32)
row("generate_carrier4  Int16 Val7 (iter→fill)", t_iter4)
row("generate_carrier   Int16 Val7 (iter, no store)", t_acc)
