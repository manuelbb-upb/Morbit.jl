# An immutable implementation of AbstractMOP for internal use.

# TODO make num_vars a type parameter?
# TODO make vectors (`full_lb` etc.) statically sized?
@with_kw struct StaticMOP{
	F <: AbstractFloat,
	OT <: Tuple{Vararg{<:AbstractObjective}},
	} <: AbstractMOP{false}
	
	num_vars :: Int
	num_objectives :: Int # no of scalar valued outputs 
	
	tuple_of_objectives :: OT
	objf_output_mapping :: Base.ImmutableDict{UInt64,Vector{Int}}

	full_lb :: Vector{F}
	full_ub :: Vector{F}

	full_width :: Vector{F} = full_ub - full_lb
	
	full_lb_int :: Vector{F} = [isinf(l) ? l : 0.0 for l ∈ full_lb ]
	full_ub_int :: Vector{F} = [ isinf(u) ? u : 1.0 for u ∈ full_ub ]

	internal_indices :: Vector{Int} = vcat( (collect(objf_output_mapping[hash(objf)]) for objf in tuple_of_objectives )... )
	reverse_indices :: Vector{Int} = sortperm(internal_indices)
end

function StaticMOP( mop :: AbstractMOP, out_el_type :: Type = Nothing )
	num_out = num_objectives(mop)

	@assert num_out > 0 "There must at least be 1 objective."

	mop_objectives = list_of_objectives(mop)
	if out_el_type <: AbstractFloat
		tuple_of_objectives = Tuple( [ OutTypeWrapper(objf, Vector{out_el_type}) for objf in mop_objectives ] )
	else	
		tuple_of_objectives = Tuple( mop_objectives )
	end

	objf_output_mapping = Base.ImmutableDict( 
		( hash( tuple_of_objectives[i]) => output_indices( mop_objectives[i], mop) for i = eachindex(mop_objectives))...
	)
	return StaticMOP(;
		num_vars = num_vars(mop),
		num_objectives = num_out,
		tuple_of_objectives, 
		objf_output_mapping,
		full_lb = full_lower_bounds(mop),
		full_ub = full_upper_bounds(mop)
	)
end

full_lower_bounds( mop :: StaticMOP ) = mop.full_lb
full_upper_bounds( mop :: StaticMOP ) = mop.full_ub

list_of_objectives( mop :: StaticMOP ) = mop.tuple_of_objectives

num_vars( mop :: StaticMOP ) = mop.num_vars

num_objectives( mop :: StaticMOP ) = mop.num_objectives

output_indices( mop :: StaticMOP ) = mop.internal_indices
output_indices( objf :: AbstractObjective, mop :: StaticMOP ) = mop.objf_output_mapping[ hash(objf) ]

full_lower_bounds_internal( mop :: StaticMOP ) = mop.full_lb_int
full_upper_bounds_internal( mop :: StaticMOP ) = mop.full_ub_int
