# This file defines the required data structures and methods for vector-valued
# Taylor models.
#
# This file is included from within "Surrogates.jl".
# We therefore can refer to other data structures used there.

@with_kw struct LagrangeModel <: SurrogateModel
    n_out :: Int64 = -1;
    degree :: Int64 = 1;
    fully_linear :: Bool = false;
end
broadcastable( lm :: LagrangeModel ) = Ref(lm);
fully_linear( lm :: LagrangeModel ) = lm.fully_linear;

struct LagrangeMeta end;

function build_model( ac :: AlgoConfig, objf :: VectorObjectiveFunction, cfg :: LagrangeConfig)
    return LagrangeModel(), LagrangeMeta()
end

eval_models( lm :: LagrangeModel, ξ :: Vector{Float64}) = nothing;
get_gradient( lm :: LagrangeModel, ξ :: Vector{Float64}, ℓ :: Int64 ) = nothing;
get_jacobian( lm :: LagrangeModel, ξ :: Vector{Float64}) = nothing;

make_linear!( ::LagrangeModel, ::LagrangeMeta, ::AlgoConfig, ::VectorObjectiveFunction, ::LagrangeConfig, ::Bool ) = false;
improve!( ::LagrangeModel, ::LagrangeMeta, ::AlgoConfig, ::VectorObjectiveFunction, ::LagrangeConfig )  = false;
