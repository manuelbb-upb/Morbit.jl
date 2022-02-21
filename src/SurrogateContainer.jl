"Group functions with indices of type `NLIndex` by model config type."
function do_groupings( mop :: AbstractMOP, ac :: AbstractConfig )
	nl_indices = get_nl_function_indices(mop)
 
	if !_combine_models_by_type(ac)
        groupings = [ ModelGrouping( [ind,], get_cfg(_get(mop,ind)) ) for ind in nl_indices ]
		groupings_dict = Dictionary( nl_indices, collect(eachindex(nl_indices)) )
    else
		groupings = ModelGrouping[]
		groupings_dict = Dictionary{NLIndex, Int}()
		
		for objf_ind1 in nl_indices
			objf1 = _get( mop, objf_ind1 )

			# check if there already is a group that `objf1`
			# belongs to and set `group_index` to its position in `groupings`
			group_index = -1
			for (gi, group) in enumerate(groupings)
				if _contains_index( group, objf_ind1 )
					group_index = gi
					break
				end
			end
			# if there is no group with `objf1` in it, 
			# then create a new one and set `group_index` to new, last position
			if group_index < 0
				push!( groupings, ModelGrouping(NLIndex[objf_ind1,], model_cfg(objf1)) )
				group_index = length(groupings)
				insert!( groupings_dict, objf_ind1, group_index)
			end
			
			group = groupings[group_index]
			
			# now, for every remaining function index, check if we can add 
			# it to the group of `objf_ind1`
			for objf_ind2 in nl_indices
				objf2 = _get( mop, objf_ind2 )
				if objf_ind1 != objf_ind2 && combinable( objf1, objf2 ) && !_contains_index(group, objf_ind2)
					push!( group.indices, objf_ind2 )
					insert!(groupings_dict, objf_ind2, group_index)
				end
			end
		end
	end
    return groupings, groupings_dict
end

struct GroupedSurrogates{
	C <: AbstractSurrogateConfig,
	M <: AbstractSurrogate,
	I <: AbstractSurrogateMeta,
	T <: Tuple
}
	cfg :: C 
	model :: M
	meta :: I 

	indices :: T # Tuple{NLIndex}

	num_outputs :: Int 

	# "A dictionary mapping an `NLIndex` to the corresponding model output indices."
	index_outputs_dict :: Dictionary{NLIndex,Vector{Int}}
end

Base.broadcastable( gs :: GroupedSurrogates ) = Ref(gs)

function init_grouped_surrogates(
	cfg, model, meta, indices, no_out = -1;
	index_outputs_dict = nothing, kwargs... 
)

	if no_out < 0
		no_out = sum( num_outputs(ind) for ind = indices ) 
	end

	if isnothing(index_outputs_dict)
		d = Dictionary{NLIndex,Vector{Int}}()
		offset = 1
		for ind = indices
			ind_out = num_outputs(ind)
			insert!(d, ind, collect( offset : offset + num_outputs(ind) - 1 ) )
			offset += ind_out
		end
	else 
		d = index_outputs_dict
	end
	return GroupedSurrogates(cfg, model, meta, Tuple(indices), no_out, d)
end

get_meta( gs :: GroupedSurrogates ) = gs.meta 
get_cfg( gs :: GroupedSurrogates ) = gs.cfg
get_indices( gs :: GroupedSurrogates ) = gs.indices 
num_outputs( gs :: GroupedSurrogates ) = gs.num_outputs
get_model( gs :: GroupedSurrogates ) = gs.model
function get_output_indices( gs :: GroupedSurrogates, ind :: NLIndex ) 
	return gs.index_outputs_dict[ind]
end
fully_linear(gs :: GroupedSurrogates ) = fully_linear(get_model(gs))

struct SurrogateContainer{
	T,D,
    ObjfType, NlEqType, NlIneqType
} <: AbstractSurrogateContainer

	surrogates :: T # GroupedSurrogates Vector

	# dictionary of NLIndex => Int, telling us the position of a single NLIndex in `surrogates`
	surrogates_grouping_dict :: D 

   	objective_functions :: ObjfType	# Dict ObjectiveIndex => AbstractSurrogate
	nl_eq_constraints :: NlEqType
	nl_ineq_constraints :: NlIneqType
end

_sc_outputs(dict) = isempty(dict) ? 0 : sum(num_outputs(ind) for ind=keys(dict))
function Base.show(io::IO, sc :: SurrogateContainer)
    str = "SurrogateContainer with $(length(sc.surrogates)) `GroupedSurrogates`. "
    if !get(io, :compact, false)
        str *= """There are 
        * $(_sc_outputs(sc.objective_functions)) objective outputs,
        * $(_sc_outputs(sc.nl_eq_constraints)) nonlin. equality constraint outputs,
        * $(_sc_outputs(sc.nl_ineq_constraints)) nonlin. inequality constraint outputs."""
    end
    print(io, str)
end

# TODO improve this function
_meta_array_type( sc :: SurrogateContainer ) = typeof( [ get_meta(gs) for gs = sc.surrogates ] )

"""
	_surrogate_from_vec_function( vfun, scal, gs, nl_ind )

Provided a vector function `vfun` that is either of type 
`RefVecFun` or `ExprVecFun` and a `GroupedSurrogates` object `gs`, 
that contains the model for `vfun.nl_index`, create and return 
a `RefSurrogate` or `ExprSurrogate` from `gs`.
"""
function _surrogate_from_vec_function( vfun, scal, gs )
	output_indices = get_output_indices(gs, vfun.nl_index)
	model = get_model( gs )
	cfg = get_cfg(gs)
	if vfun isa RefVecFun
		return RefSurrogate( model, output_indices, vfun.nl_index )
	elseif vfun isa ExprVecFun 
		return ExprSurrogate( model, vfun.expr_str, scal, output_indices, vfun.nl_index )
	end
end

function _surrogate_from_vec_function( vfun, scal, gs_array, groupings_dict )
	#gs_position = findfirst( gs -> vfun.nl_index in get_indices(gs) )
	gs_position = groupings_dict[vfun.nl_index]
	return _surrogate_from_vec_function( vfun, scal, gs_array[gs_position] )
end

function _create_dict(mop, gs_array, groupings_dict, indices, scal, index_type = FunctionIndex )
	if isempty( indices )
		return Base.ImmutableDict{index_type, Nothing}()
	else
		num_indices = length(indices)
		dict_vals = [ _surrogate_from_vec_function( _get(mop,ind), scal, gs_array, groupings_dict ) for ind = indices ]		
		return ArrayDictionary{ index_type, eltype(dict_vals) }( SVector{num_indices}(collect(indices)), dict_vals )
	end
end

function init_surrogate_container( grouped_surrogate_array, groupings_dict, mop :: AbstractMOP, scal :: AbstractVarScaler )
	objective_functions = _create_dict( mop, grouped_surrogate_array, groupings_dict, get_objective_indices(mop), scal, ObjectiveIndex)
	nl_eq_constraints = _create_dict( mop, grouped_surrogate_array, groupings_dict, get_nl_eq_constraint_indices(mop), scal, ConstraintIndex)
	nl_ineq_constraints = _create_dict( mop, grouped_surrogate_array, groupings_dict, get_nl_ineq_constraint_indices(mop), scal, ConstraintIndex)
	return SurrogateContainer( 
		grouped_surrogate_array,
		groupings_dict,
		objective_functions,
		nl_eq_constraints,
		nl_ineq_constraints
	)
end

get_indices( sc :: SurrogateContainer ) = Iterators.flatten( get_indices(gs) for gs = sc.surrogates )
get_objective_indices( sc :: SurrogateContainer ) = keys( sc.objective_functions )
get_nl_eq_constraint_indices( sc :: SurrogateContainer ) = keys( sc.nl_eq_constraints )
get_nl_ineq_constraint_indices( sc :: SurrogateContainer ) = keys( sc.nl_ineq_constraints )
_get_all_indices(sc) = collect( Iterators.flatten([
	get_indices(sc),
	get_objective_indices(sc),
	get_nl_eq_constraint_indices(sc),
	get_nl_ineq_constraint_indices(sc)
]))
get_function_indices(sc) = collect( Iterators.flatten([
	get_objective_indices(sc),
	get_nl_eq_constraint_indices(sc),
	get_nl_ineq_constraint_indices(sc)
]))
get_surrogates( sc :: SurrogateContainer, ind :: NLIndex) = sc.surrogates[ sc.surrogates_grouping_dict[ind] ]
get_surrogates( sc :: SurrogateContainer, ind :: ObjectiveIndex) = sc.objective_functions[ind]
function get_surrogates( sc :: SurrogateContainer, ind :: ConstraintIndex) 
	if ind.type == :nl_eq 
		return sc.nl_eq_constraints[ind]
	else ind.type == :nl_ineq
		return sc.nl_ineq_constraints[ind]
	end
end

function fully_linear(sc :: SurrogateContainer)
	return all( fully_linear( get_surrogates(sc, ind) ) for ind = _get_all_indices(sc) )
end

function set_fully_linear!(sc :: SurrogateContainer, val)
	for ind = _get_all_indices(sc)
		set_fully_linear!( get_surrogates(sc, ind), val )
	end
end

function eval_vec_container_at_func_index_at_scaled_site( 
	sc :: SurrogateContainer, scal :: AbstractVarScaler, x_scaled :: Vec, func_ind :: AnyIndex 
)
	m = get_surrogates( sc, func_ind )
	return _eval_models_vec( m, scal, x_scaled )
end

function eval_container_jacobian_at_func_index_at_scaled_site( 
	sc :: SurrogateContainer, scal :: AbstractVarScaler, x_scaled :: Vec, func_ind :: AnyIndex 
)
	m = get_surrogates( sc, func_ind )
	return get_jacobian( m, scal, x_scaled )
end

for mod_type in ["objective", "nl_eq_constraint", "nl_ineq_constraint"]
	# names of functions to be generated
	get_XXX_indices = Symbol("get_", mod_type, "_indices")
	fully_linear_XXX = Symbol( "fully_linear_", mod_type, "s" )
	get_XXX_optim_handles = Symbol( "get_", mod_type, "s_optim_handles" )
	eval_container_XXX_at_scaled_site = Symbol( "eval_container_", mod_type, "s_at_scaled_site" )
	eval_container_XXX_jacobian_at_scaled_site = Symbol( "eval_container_", mod_type, "s_jacobian_at_scaled_site" )

	@eval begin 
		function $(fully_linear_XXX)( sc :: SurrogateContainer )
			indices = $(get_XXX_indices)( sc )
			return all( fully_linear(get_surrogates(sc, ind)) for ind = indices )
		end

		function $(get_XXX_optim_handles)( sc :: SurrogateContainer, scal :: AbstractVarScaler )
			indices = $(get_XXX_indices)( sc )
			return Iterators.flatten( 
				[ let mod = get_surrogates(sc,ind); 
					[ _get_optim_handle( mod, scal, ℓ) for ℓ = 1:num_outputs(mod) ]
				end for ind = indices ]
			)		
		end

		function $(eval_container_XXX_jacobian_at_scaled_site)( sc :: SurrogateContainer, scal :: AbstractVarScaler, x_scaled)
			indices = $(get_XXX_indices)( sc )
			isempty(indices) && return Matrix{MIN_PRECISION}(undef, 0, length(x_scaled))
			return reduce( vcat, [ eval_container_jacobian_at_func_index_at_scaled_site( sc, scal, x_scaled, ind) for ind = indices ] )
		end

		function $(eval_container_XXX_at_scaled_site)( sc :: SurrogateContainer, scal :: AbstractVarScaler, x_scaled)
			indices = $(get_XXX_indices)( sc )
			isempty(indices) && return MIN_PRECISION[]
			return reduce( vcat, [ eval_vec_container_at_func_index_at_scaled_site( sc, scal, x_scaled, ind) for ind = indices ] )
		end
	end#eval
end#for

@doc "Return a SurrogateContainer initialized from the information provided in `mop`."
function init_surrogates( mop :: AbstractMOP, scal :: AbstractVarScaler, id :: AbstractIterate, 
	ac :: AbstractConfig, groupings :: Vector{ModelGrouping}, groupings_dict, sdb )
    @logmsg loglevel2 "Initializing surrogate models."
    
	# round I of model building: collecting the meta data
    meta_array = AbstractSurrogateMeta[]
    for group in groupings
        meta = prepare_init_model( group.cfg, Tuple(group.indices), mop, scal, id, sdb, ac; meta_array )
        push!(meta_array, meta)
    end
    
    @logmsg loglevel2 "Evaluation of unevaluated results."
    eval_missing!(sdb, mop, scal)
    
	# round II: from meta data and evaluations, construct the surrogates
    gs_array = GroupedSurrogates[]
    for (i,group) in enumerate(groupings)
        _meta = meta_array[i]
        model, meta = init_model( _meta, group.cfg, Tuple(group.indices), mop, scal, id, sdb, ac )
        gs = init_grouped_surrogates( group.cfg, model, meta, group.indices )
        push!(gs_array, gs)
    end

    return init_surrogate_container(gs_array, groupings_dict, mop, scal)
end

for (_fieldname, fnname) = [
	(:objective_functions, :update_objectives!),
	(:nl_eq_constraints, :update_nl_eq_constraints!),
	(:nl_ineq_constraints, :update_nl_ineq_constraints!),
]
	fieldname = Meta.quot(_fieldname)
	@eval function $(fnname)(sc :: SurrogateContainer, updated_indices, scal )
		fn_dict = getfield(sc, $fieldname)
		for (ind, mod) = pairs(fn_dict)
			surrogate_index = sc.surrogates_grouping_dict[ mod.nl_index ] 
			if surrogate_index in updated_indices 
				if mod isa RefSurrogate
					fn_dict[ind] = RefSurrogate(
						get_model(sc.surrogates[ surrogate_index ]), # updated model
						mod.output_indices, mod.nl_index
					)
				elseif mod isa ExprSurrogate
					fn_dict[ind] = ExprSurrogate( 
						get_model(sc.surrogates[ surrogate_index ]),
						mod.expr_str, scal, mod.output_indices, mod.nl_index
					)
				end
			end
		end#for
		return nothing
	end#function
end

#defined below:
function update_surrogates!(sc, mop, scal, id, sdb, ac; kwargs...) end
function improve_surrogates!(sc, mop, scal, id, sdb, ac; kwargs...) end 
for method_name in [:update, :improve]
	_method = Symbol("$(method_name)_surrogates!")
	_mod_prepare_method = Symbol("prepare_$(method_name)_model")
	_mod_method = Symbol("$(method_name)_model")
	_necessity_check = Symbol("requires_$(method_name)")
	@eval function $(_method)( 
		sc :: SurrogateContainer, 
		mop :: AbstractMOP, scal :: AbstractVarScaler, iter_data :: AbstractIterate, 
		sdb, ac :: AbstractConfig;
		ensure_fully_linear = true 
	)
		@logmsg loglevel2 "Updating surrogate models."

		# round I of model building: collecting the meta data
		meta_array = empty( [ get_meta(gs) for gs = sc.surrogates ] )
		updated_indices = Int[]
		for (gi,gs) = enumerate(sc.surrogates)
			indices = get_indices(gs)
			mod = get_model(gs)
			meta = get_meta(gs)
			cfg = get_cfg(gs)
			if $(_necessity_check)(cfg)
				new_meta = $(_mod_prepare_method)( 
					mod, meta, cfg, indices, mop, scal, iter_data, sdb, ac; ensure_fully_linear, meta_array
				)
				push!(meta_array, new_meta)
				push!(updated_indices, gi)
			end
		end
		
		@logmsg loglevel2 "Evaluation of unevaluated results."
		eval_missing!(sdb, mop, scal)
		
		# round II: from meta data and evaluations, update the surrogates
		for (i,gs) = enumerate(sc.surrogates[updated_indices])
			mod = get_model(gs)
			cfg = get_cfg(gs)
			indices = get_indices(gs)
			_meta = meta_array[i]
			model, meta = $(_mod_method)( 
				mod, _meta, cfg, indices, mop, scal, iter_data, sdb, ac 
			)
			new_gs = init_grouped_surrogates( 
				cfg, model, meta, indices; 
				index_outputs_dict = gs.index_outputs_dict 
			)
			
			# replace groupd surrogates
			sc.surrogates[updated_indices[i]] = new_gs
		end

		# finally, update dependent functions 
		update_objectives!(sc, updated_indices, scal)
		update_nl_eq_constraints!(sc, updated_indices, scal)
		update_nl_ineq_constraints!(sc, updated_indices, scal)
		return nothing
	end
end

