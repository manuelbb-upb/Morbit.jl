module Morbit 

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

import ForwardDiff
const AD = ForwardDiff

import Logging: LogLevel, @logmsg
import Logging

include("custom_logging.jl")

include("shorthands.jl");
include("Interfaces.jl");

# implementations (order should not matter)
include("VecFunImplementation.jl");
include("MOP.jl")

include("ResultImplementation.jl")
include("DataBaseImplementation.jl")
include("IterDataImplementation.jl")

include("RbfModel.jl")
include("ExactModel.jl")
include("TaylorModel.jl")
include("SurrogateContainerImplementation.jl")

include("ConfigImplementations.jl")

include("FilterImplementation.jl")

include("VarScaler.jl")

# utilities
include("convenience_functions.jl")
include("descent.jl")
#include("saving.jl")
include("utilities.jl")

include("algorithm.jl")

export AlgoConfig #, DefaultConfig
export MOP, add_lower_bound!, add_upper_bound!, del_lower_bound!, del_upper_bound!, add_objective!
export optimize
end#module
