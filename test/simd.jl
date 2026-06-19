using SIMD

@testset "SIMD.Vec support" begin
    # fpsin/fpcos/fpsincos on SIMD.Vec must match the scalar results exactly.
    @testset "Int32 x16, quarter bits $B" for B in 8:14
        W = 16
        xs = Int32.(rand(-1 << (B + 3):1 << (B + 3), W * 4))
        for i in 1:W:length(xs)
            v = SIMD.Vec{W,Int32}(ntuple(j -> xs[i + j - 1], W))
            s_ref = [fpsin(xs[i + j - 1], Val(B)) for j in 1:W]
            c_ref = [fpcos(xs[i + j - 1], Val(B)) for j in 1:W]
            @test collect(Tuple(fpsin(v, Val(B)))) == s_ref
            @test collect(Tuple(fpcos(v, Val(B)))) == c_ref
            s, c = fpsincos(v, Val(B))
            @test collect(Tuple(s)) == s_ref
            @test collect(Tuple(c)) == c_ref
        end
    end

    @testset "Int16 x32, quarter bits $B" for B in 3:7
        W = 32
        xs = Int16.(rand(-1 << (B + 3):1 << (B + 3), W * 4))
        for i in 1:W:length(xs)
            v = SIMD.Vec{W,Int16}(ntuple(j -> xs[i + j - 1], W))
            s_ref = [fpsin(xs[i + j - 1], Val(B)) for j in 1:W]
            c_ref = [fpcos(xs[i + j - 1], Val(B)) for j in 1:W]
            @test collect(Tuple(fpsin(v, Val(B)))) == s_ref
            @test collect(Tuple(fpcos(v, Val(B)))) == c_ref
            s, c = fpsincos(v, Val(B))
            @test collect(Tuple(s)) == s_ref
            @test collect(Tuple(c)) == c_ref
        end
    end
end
