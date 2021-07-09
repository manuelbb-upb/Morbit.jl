
module Morbit
using Printf: @sprintf

# steepest descent and directed search
using LinearAlgebra: norm, pinv
import JuMP;
import OSQP;

# LagrangeModels and PS
import NLopt;

using Parameters: @with_kw, @unpack, @pack!
using MathOptInterface;
const MOI = MathOptInterface;

import ThreadSafeDicts: ThreadSafeDict;
using Memoize;

import UUIDs;

import FiniteDiff#erences
const FD = FiniteDiff#erences

import ForwardDiff
const AD = ForwardDiff

import Logging: LogLevel, @logmsg
import Logging

include("custom_logging.jl")

include("shorthands.jl");
include("Interfaces.jl");
include("diff_wrappers.jl");

# implementations (order should not matter)
include("VectorObjectiveFunction.jl");
include("MixedMOP.jl");
include("StaticMOP.jl")

include("ResultImplementation.jl")
include("DataBaseImplementation.jl")
include("IterDataImplementation.jl")
include("SurrogatesImplementation.jl");

include("ConfigImplementations.jl")

# utilities
include("adding_objectives.jl");
include("descent.jl")
#include("saving.jl")
include("utilities.jl")

# we expect mop :: MixedMOP, but should work for static MOP if everything 
# is set up properly 
function initialize_data( mop :: AbstractMOP, x0 :: Vec, fx0 :: Vec = Float32[]; 
    algo_config :: Union{AbstractConfig, Nothing} = nothing, 
    populated_db :: Union{AbstractDB, Nothing} = nothing )
    
	if num_objectives(mop) == 0
		error("`mop` has no objectives!")
	end
			
	@warn "The evaluation counter of `mop` is reset."
	reset_evals!( mop )
	# initialize first iteration site
	@assert !isempty( x0 ) "Please provide a non-empty feasible starting point `x0`."

	# for backwards-compatibility with unconstrained problems:
	if num_vars(mop) == 0
		# will error if mop isa StaticMOP
		MOI.add_variables(mop, length(x0))
	end

	x_scaled = scale( x0, mop );
	tfn = TransformerFn(mop)

	# make problem static 
	smop = StaticMOP(mop)

	# initalize first objective vector 
	if isempty( fx0 )
		# if no starting function value was provided, eval objectives
		fx_sorted = eval_all_objectives( smop, x_scaled, tfn );
	else
		fx_sorted = apply_internal_sorting( fx0, smop );
	end 

	# ensure at least half-precision
	F = Base.promote_eltype( x_scaled, fx_sorted, Float32 )
	x = F.(x_scaled)
	fx = F.(fx_sorted)

	if isnothing( algo_config )
		ac = DefaultConfig{F}()
	else
		ac = algo_config
	end

	# initialize iter data obj
	id = init_iter_data( IterData, x, fx, Δ⁰(ac) )
	CT = saveable_type(id)	 # type for database constructor

	# initialize database
	if !isnothing(populated_db)
		# has a database been provided? if yes, prepare (scale vars, sort values)
		data_base = populated_db;
		transform!( data_base, mop )
	else
		result_type = Result{F}
		data_base = init_db( ArrayDB, F, CT );
		set_transformed!(data_base, true)
	end
	#%%
	# make sure, x & fx are in database
	x_id = ensure_contains_values!(data_base, x, fx)
	set_x_index!(id, x_id)

	# make the problem static 
	# initialize surrogate models
	sc = init_surrogates( smop, id, ac );

	return (smop, tfn, ac, data_base, id, sc)
end
end