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

using GeneralizedGenerated

using Lazy: @forward

registered_funcs = Dict{Symbol,Function}() 

include("custom_logging.jl")

include("globals.jl");	# has types meant to be available globally -> import first
include("SuperTypes.jl"); # abstract types and interfaces 

# ### Interface Definitions
include("AbstractSurrogateInterface.jl");
include("AbstractMOPInterface.jl");
include("AbstractConfigInterface.jl")
include("AbstractFilterInterface.jl")

include("DiffFn.jl") # required by VecFun.jl
include("VarScaler.jl")

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
include(joinpath(@__DIR__, "models", "LagrangeModel.jl"))

include("ConfigImplementations.jl")

include("utilities.jl")

include("FilterImplementation.jl")

include("descent.jl")

include("algorithm.jl")

for (s1,s2) = [
	(:exact, :ExactConfig),
	(:rbf, :RbfConfig),
	(:lagrange, :LagrangeConfig),
	(:taylor, :TaylorConfig)
]
	add_objective_fn = Symbol("add_$(s1)_objective!")
	add_nl_eq_constraint_fn = Symbol("add_$(s1)_nl_eq_constraint!")
	add_nl_ineq_constraint_fn = Symbol("add_$(s1)_nl_ineq_constraint!")
	
	add_objectives_fn = Symbol("add_$(s1)_objectives!")
	add_nl_eq_constraints_fn = Symbol("add_$(s1)_nl_eq_constraints!")
	add_nl_ineq_constraints_fn = Symbol("add_$(s1)_nl_ineq_constraints!")
	@eval begin 
		function $(add_objective_fn)(mop :: AbstractMOP, f :: Function; kwargs ... )
			return add_objective!(mop, f; model_cfg = $(s2)(), n_out = 1, kwargs...)
		end
		function $(add_nl_eq_constraint_fn)(mop :: AbstractMOP, f :: Function; kwargs ... )
			return add_nl_eq_constraint!(mop, f; model_cfg = $(s2)(), n_out = 1, kwargs... )
		end
		function $(add_nl_ineq_constraint_fn)(mop :: AbstractMOP, f :: Function; kwargs ... )
			return add_nl_ineq_constraint!(mop, f; model_cfg = $(s2)(), n_out = 1, kwargs... )
		end

		function $(add_objectives_fn)(mop :: AbstractMOP, f :: Function; kwargs ... )
			return add_objective!(mop, f; model_cfg = $(s2)(), kwargs...)
		end
		function $(add_nl_eq_constraints_fn)(mop :: AbstractMOP, f :: Function; kwargs ... )
			return add_nl_eq_constraint!(mop, f; model_cfg = $(s2)(), kwargs...)
		end
		function $(add_nl_ineq_constraints_fn)(mop :: AbstractMOP, f :: Function; kwargs ... )
			return add_nl_ineq_constraint!(mop, f; model_cfg = $(s2)(), kwargs...)
		end
		export $(add_objective_fn), $(add_nl_eq_constraint_fn), $(add_nl_ineq_constraint_fn)
		export $(add_objectives_fn), $(add_nl_eq_constraints_fn), $(add_nl_ineq_constraints_fn)
	end
end
export AlgorithmConfig, AlgoConfig 
export MOP, add_lower_bound!, add_upper_bound!, del_lower_bound!, 
	del_upper_bound!, add_objective!, add_nl_eq_constraint!, add_nl_ineq_constraint!,
	add_eq_constraint!, add_ineq_constraint!
export ExactConfig
export TaylorConfig, TaylorCallbackConfig
export RbfConfig
export optimize
export AutoDiffWrapper, FiniteDiffWrapper
end#module
