using SIMD

# Scalar reference: the exact integer-DDA phase for each sample, fed through the same
# `fpsincos`. The SIMD kernels must reproduce this bit-for-bit. Computing the phase as
# `div((i-1)*step_num, step_den)` (exact integer division, no running fraction) is what
# makes the carrier drift-free — so matching this reference over a long, odd-denominator
# array is also the drift-free test.
function ref_carrier(::Type{T}, bits::Val{N}, step_num, step_den, n; phase_units = 0) where {T,N}
    sin_out = Vector{T}(undef, n); cos_out = Vector{T}(undef, n)
    for i in 1:n
        phase = (div((i - 1) * step_num, step_den) + phase_units) % T
        sin_out[i], cos_out[i] = fpsincos(phase, bits)
    end
    sin_out, cos_out
end

@testset "carrier generation" begin
    @testset "exact match vs integer-DDA reference (drift-free)" begin
        # Int16 / Int32, several N, power-of-two and odd (drift-prone) denominators.
        cases = ((Int16, 7, 32), (Int16, 5, 32), (Int32, 14, 16), (Int32, 8, 16))
        for (T, N, W) in cases, (step_num, step_den) in ((3, 8), (7, 1000), (123, 4099))
            n = 4096 + 37                       # not a multiple of the 4W stride → exercises the tail
            sg = Vector{T}(undef, n); cg = Vector{T}(undef, n)
            generate_carrier!(sg, cg, Val(N), step_num, step_den; lanes = Val(W))
            sr, cr = ref_carrier(T, Val(N), step_num, step_den, n)
            @test sg == sr
            @test cg == cr
        end
    end

    @testset "accuracy vs true sin/cos (N=$N, $T)" for (T, N, tol) in
            ((Int16, 7, 0.03), (Int32, 14, 3e-4), (Int32, 8, 0.02))
        n = 8192
        cyc = 7 / n                              # 7 whole cycles across the buffer
        sg = Vector{T}(undef, n); cg = Vector{T}(undef, n)
        generate_carrier!(sg, cg, Val(N), cyc)
        amp = 2.0^N
        sin_err = maximum(abs(sg[i] / amp - sin(2π * cyc * (i - 1))) for i in 1:n)
        cos_err = maximum(abs(cg[i] / amp - cos(2π * cyc * (i - 1))) for i in 1:n)
        @test sin_err < tol
        @test cos_err < tol
    end

    @testset "phase offset" begin
        N, T, W = 13, Int32, 16
        n = 2048
        # Integer phase = exact phase units.
        sg = Vector{T}(undef, n); cg = Vector{T}(undef, n)
        generate_carrier!(sg, cg, Val(N), 5, 16; phase = 1234, lanes = Val(W))
        sr, cr = ref_carrier(T, Val(N), 5, 16, n; phase_units = 1234)
        @test sg == sr && cg == cr
        # Real phase = cycles → phase units = round(cycles * 2^(N+2)); first sample is
        # that constant phase.
        generate_carrier!(sg, cg, Val(N), 5, 16; phase = 0.25, lanes = Val(W))
        units = round(Int, 0.25 * (1 << (N + 2)))
        @test (sg[1], cg[1]) == fpsincos((units % T), Val(N))
    end

    @testset "frequency / sampling_frequency form" begin
        N, T = 13, Int32
        n = 4096
        s1 = Vector{T}(undef, n); c1 = Vector{T}(undef, n)
        s2 = Vector{T}(undef, n); c2 = Vector{T}(undef, n)
        generate_carrier!(s1, c1, Val(N); frequency = 1000, sampling_frequency = 2_000_000)
        generate_carrier!(s2, c2, Val(N), cycles_per_sample(1000, 2_000_000))
        @test s1 == s2 && c1 == c2
        @test cycles_per_sample(1000, 2_000_000) == 1000 / 2_000_000
    end

    @testset "generate_carrier iterator matches generate_carrier!" begin
        N, T, W = 7, Int16, 32
        step_num, step_den = 7, 1000
        n = 32 * W                                # exact multiple of W → no tail
        sg = T[]; cg = T[]
        for (sin_vec, cos_vec) in generate_carrier(Val(N), step_num, step_den, n; type = T, lanes = Val(W))
            append!(sg, Tuple(sin_vec)); append!(cg, Tuple(cos_vec))
        end
        @test length(sg) == n
        sr, cr = ref_carrier(T, Val(N), step_num, step_den, n)
        @test sg == sr && cg == cr
    end

    @testset "generate_carrier4 iterator matches generate_carrier!" begin
        N, T, W = 14, Int32, 16
        step_num, step_den = 123, 4099
        n = 16 * (4W)                             # exact multiple of 4W
        sg = T[]; cg = T[]
        for ((s0, c0), (s1, c1), (s2, c2), (s3, c3)) in
                generate_carrier4(Val(N), step_num, step_den, n; type = T, lanes = Val(W))
            append!(sg, Tuple(s0)); append!(cg, Tuple(c0))
            append!(sg, Tuple(s1)); append!(cg, Tuple(c1))
            append!(sg, Tuple(s2)); append!(cg, Tuple(c2))
            append!(sg, Tuple(s3)); append!(cg, Tuple(c3))
        end
        @test length(sg) == n
        sr, cr = ref_carrier(T, Val(N), step_num, step_den, n)
        @test sg == sr && cg == cr
    end

    @testset "iterators are allocation-free" begin
        # Both constructing the iterator and consuming it happen inside a function
        # barrier, so the iterator type is concrete and there is no non-const global
        # read (a global read itself allocates ~16 bytes on Julia 1.10). The loop must
        # not allocate — the whole point of the array-free API.
        consume(it) = (acc = 0; for (s, c) in it; acc += Int(s[1]) + Int(c[1]); end; acc)
        consume4(it) = (acc = 0;
            for ((s0, c0), (s1, c1), (s2, c2), (s3, c3)) in it
                acc += Int(s0[1]) + Int(c0[1]) + Int(s1[1]) + Int(s2[1]) + Int(s3[1])
            end; acc)
        function alloc_iter()
            it = generate_carrier(Val(13), 3, 16, 4096; type = Int32, lanes = Val(16))
            @allocated consume(it)
        end
        function alloc_iter4()
            it = generate_carrier4(Val(13), 3, 16, 4096; type = Int32, lanes = Val(16))
            @allocated consume4(it)
        end
        alloc_iter(); alloc_iter4()               # warm up / compile
        @test alloc_iter() == 0
        @test alloc_iter4() == 0
    end

    @testset "argument validation" begin
        s = Vector{Int16}(undef, 16); c = Vector{Int16}(undef, 16)
        @test_throws DimensionMismatch generate_carrier!(s, Vector{Int16}(undef, 8), Val(7), 1, 4)
        @test_throws ArgumentError generate_carrier!(s, c, Val(7), 1, 0)                 # step_den == 0
        @test_throws ArgumentError generate_carrier!(s, c, Val(7), 1, typemax(Int16) + 1) # step_den too large
    end
end
