# This file defines the required data structures and methods for vector-valued
# Taylor models.
#
# This file is included from within "Surrogates.jl".
# We therefore can refer to other data structures used there.
# NOTE Scaling of training SITES is provided by functions
# (un)scale( mop :: MixedMop, x )
# imported from "Objectives.jl" in "Surrogates.jl"

# import basic definitions from standalone module

fully_linear(m :: RBFModel) = m.fully_linear

import Base: ==
==( config1 :: RbfConfig, config2 :: RbfConfig ) = begin
    if  all( getfield( config1, fname ) == getfield( config2, fname )
            for fname in fieldnames(RbfConfig) if fname != :shape_parameter
        )

        if isa( config1.shape_parameter, Float64 )
            if isa( config2.shape_parameter, Float64 )
                return config1.shape_parameter == config2.shape_parameter
            else
                return all( isapprox.( config1.shape_parameter, config2.shape_parameter.(0:0.1:1.0)))
            end
        else
            return all(
                isapprox.(
                    config1.shape_parameter.(0:0.1:1.0),
                    config2.shape_parameter.(0:0.1:1.0)
                )
            )
        end
    else
        return false
    end
end

@doc "Modify first meta data object to equal second."
function as_second!(destination :: RBFMeta, source :: RBFMeta )
    @unpack_RBFMeta source;
    @pack_RBFMeta! destination;
    return nothing
end

# use functions from base module for evaluation
# (assumes that models are trained on the unit hypercube)
eval_models( m :: RBFModel, x :: Vector{Float64}) = RBF.output(m, x)
eval_models( m :: RBFModel, x :: Vector{Float64}, ℓ :: Int64) = RBF.output(m, ℓ, x)
get_gradient( m :: RBFModel, x :: Vector{Float64}, ℓ :: Int64) = RBF.grad( m, ℓ, x )
get_jacobian( m :: RBFModel, x :: Vector{Float64}) = RBF.jac( m, x )

include("rbf_sampling.jl")

# Redefine interface methods for our own Model type
@forward RbfModel.model fully_linear, eval_models, get_gradient, get_jacobian
@forward RbfModel.model improve!, make_linear!

function build_model( ac :: AlgoConfig, objf :: VectorObjectiveFunction,
        cfg :: RbfConfig, criticality_round :: Bool = false )
    model, meta = build_rbf_model( ac, objf, cfg, criticality_round)
    return RbfModel(model), meta
end
