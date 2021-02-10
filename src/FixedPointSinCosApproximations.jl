module FixedPointSinCosApproximations

    using LoopVectorization, DocStringExtensions
    using LoopVectorization: Vec

    export fpsin,
        fpcos,
        fpsincos

    const VInt16 = Union{Int16,Vec{<:Any,Int16}}
    const VInt32 = Union{Int32,Vec{<:Any,Int32}}

    include("second_order.jl")
    include("third_order.jl")
    include("fourth_order.jl")
    include("fifth_order.jl")
    include("sixth_order.jl")
    include("approximations.jl")

end # module
