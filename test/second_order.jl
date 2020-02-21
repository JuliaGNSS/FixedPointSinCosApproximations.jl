@testset "Second order" begin
    @testset "Bits $i" for i in 3:5
        x = 0:1 << i
        y_approx = fpcos.(Int16.(x), Val(i))
        y = cos.(x / 1 << i * π / 2) .* 1 << i
        @test all(abs.(y_approx .- y) .< 2.0)
    end
end
