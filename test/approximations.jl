@testset "Sin approximation" begin
    @testset "Int16 with quarter bits $i" for i in 3:7
        x = -1 << (3 + i):1 << (3 + i) # 4π
        y_approx = fpsin.(Int16.(x), Val(i))
        y = sin.(x / 1 << i * π / 2) .* 1 << i
        @test all(abs.(y_approx .- y) .< 2.0)
    end

    @testset "Int32 with quarter bits $i" for i in 8:14
        x = -1 << (3 + i):1 << (3 + i) # 4π
        y_approx = fpsin.(Int32.(x), Val(i))
        y = sin.(x / 1 << i * π / 2) .* 1 << i
        @test all(abs.(y_approx .- y) .< 2.5)
    end
end

@testset "Cos approximation" begin
    @testset "Int16 with quarter bits $i" for i in 3:7
        x = -1 << (3 + i):1 << (3 + i) # 4π
        y_approx = fpcos.(Int16.(x), Val(i))
        y = cos.(x / 1 << i * π / 2) .* 1 << i
        @test all(abs.(y_approx .- y) .< 2.0)
    end

    @testset "Int32 with quarter bits $i" for i in 8:14
        x = -1 << (3 + i):1 << (3 + i) # 4π
        y_approx = fpcos.(Int32.(x), Val(i))
        y = cos.(x / 1 << i * π / 2) .* 1 << i
        @test all(abs.(y_approx .- y) .< 2.5)
    end
end
