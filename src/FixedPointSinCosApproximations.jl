module FixedPointSinCosApproximations

    using DocStringExtensions
    import SIMD

    export fpsin,
        fpcos,
        fpsincos

    const VInt16 = Union{Int16,SIMD.Vec{<:Any,Int16}}
    const VInt32 = Union{Int32,SIMD.Vec{<:Any,Int32}}

    include("second_order.jl")
    include("third_order.jl")
    include("fourth_order.jl")
    include("fifth_order.jl")
    include("sixth_order.jl")
    include("approximations.jl")
    include("carrier.jl")

end # module
