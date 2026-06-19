using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
using FixedPointSinCosApproximations, Printf

# For each (type, quarter-bits N) measure the current approximation error vs the
# true sin/cos, and compare to the round-to-nearest quantization floor.
# Sweep the full quarter range of phases (one period is 4*2^N).
function analyze(T, N)
    scale = 1 << N
    # full period: phase in [0, 4*2^N); test several periods incl negatives
    xs = T(-2*4*scale):T(2*4*scale)
    maxerr_s = 0.0; maxerr_c = 0.0
    floor_s = 0.0; floor_c = 0.0
    sserr = 0.0; n = 0
    for x in xs
        s_a, c_a = fpsincos(x, Val(N))
        ang = x / scale * (pi/2)
        s_t = sin(ang) * scale
        c_t = cos(ang) * scale
        es = abs(s_a - s_t); ec = abs(c_a - c_t)
        maxerr_s = max(maxerr_s, es); maxerr_c = max(maxerr_c, ec)
        # round-to-nearest floor:
        floor_s = max(floor_s, abs(round(s_t) - s_t))
        floor_c = max(floor_c, abs(round(c_t) - c_t))
        sserr += es^2 + ec^2; n += 2
    end
    (maxerr_s, maxerr_c, sqrt(sserr/n), max(floor_s, floor_c))
end

order(N) = N<=5 ? "2nd" : N<=7 ? "3rd" : N==8 ? "3rd" : N<=10 ? "4th" : N<=12 ? "5th" : "6th(->5th)"

@printf("%-7s %-5s %-12s %10s %10s %10s %10s\n","type","bits","order","max|Δsin|","max|Δcos|","rms","floor")
for (T,Ns) in ((Int16, 3:7), (Int32, 8:14))
    for N in Ns
        ms,mc,rms,fl = analyze(T,N)
        @printf("%-7s %-5d %-12s %10.3f %10.3f %10.3f %10.3f\n", T, N, order(N), ms, mc, rms, fl)
    end
end
