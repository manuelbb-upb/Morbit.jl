module Morbit 

using Dictionaries

using DocStringExtensions

using Printf: @sprintf

using StaticArrays

# steepest descent and directed search
using LinearAlgebra: norm, pinv, diagm
import LinearAlgebra
import JuMP
import OSQP

# LagrangeModels and PS
import NLopt

using Parameters: @with_kw, @unpack, @pack!

import ThreadSafeDicts: ThreadSafeDict
using Memoization

import UUIDs

import FiniteDiff#erences
const FD = FiniteDiff#erences

import Zygote
#import ForwardDiff
const AD = Zygote.ForwardDiff

import Logging: LogLevel, @logmsg
import Logging

using Lazy: @forward

registered_funcs = Dict{Symbol,Function}() 

include("custom_logging.jl")

include("globals.jl");	# has types meant to be available globally -> import first
include("SuperTypes.jl"); # abstract types and interfaces 

# record keeping (order should not matter)
include("Result.jl")
include("IterDataIterSaveable.jl")
include("Databases.jl")

include("VecFun.jl"); # makes availabe `RefVecFun` & `ExprVecFun` for file `MOP.jl`
include("MOP.jl")

#include("RbfModel.jl")
include("SurrogateContainer.jl")
include(joinpath(@__DIR__, "models", "ExactModel.jl"))
include(joinpath(@__DIR__, "models", "TaylorModel.jl"))
include(joinpath(@__DIR__, "models", "RbfModel.jl"))

include("VarScaler.jl")
include("ConfigImplementations.jl")

include("utilities.jl")

include("FilterImplementation.jl")

include("descent.jl")

include("algorithm.jl")

export AlgorithmConfig, AlgoConfig 
export MOP, add_lower_bound!, add_upper_bound!, del_lower_bound!, del_upper_bound!, add_objective!
export ExactConfig
export TaylorConfig, TaylorCallbackConfig
export RbfConfig
export optimize
export AutoDiffWrapper, FiniteDiffWrapper
end#module
