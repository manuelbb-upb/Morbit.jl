
# Implementation of the AbstractMOP interface 
# (see file `AbstractMOPInterface.jl`)

@with_kw mutable struct MixedMOP <: AbstractMOP
    vars :: Vector{Int} = [];

    vector_of_objectives :: Vector{ <:AbstractObjective } = AbstractObjective[];
    objf_output_mapping :: Union{Dict{ <:AbstractObjective, Vector{Int} }, Nothing} = nothing;

    lb :: Dict{ Int, Real } = Dict{Int,Real}();
    ub :: Dict{ Int, Real } = Dict{Int,Real}();

    # change whenever vars or constraints are added
    var_state :: UUIDs.UUID = UUIDs.uuid4();      # TODO maybe use automatic hashing? pro: no need to implement everything, cons: slower?
    objf_state :: UUIDs.UUID = UUIDs.uuid4();
end

# legacy constructors
function MixedMOP( lb :: RVec, ub :: RVec )
    @assert length(lb) == length(ub);
    @assert all( lb .<= ub );

    LB = Dict{Int,Real}();
    for (var_int, b) ∈ enumerate(lb)
        if !isinf( b )
            LB[var_int] = b
        end
    end

    UB = Dict{Int,Real}();
    for (var_int, b) ∈ enumerate(ub)
        if !isinf( b )
            UB[var_int] = b
        end
    end

    MixedMOP(;
        vars = collect( 1 : length(lb) ),
        lb = LB,
        ub = UB,
    )
end
#MixedMOP(; lb :: RVec, ub :: RVec ) = MixedMOP( lb, ub );

list_of_objectives( mop :: MixedMOP ) = mop.vector_of_objectives;

@memoize function _output_indices( 
    objf :: AbstractObjective, mop :: MixedMOP, hash :: UUIDs.UUID )
    return mop.objf_output_mapping[ objf ];
end

function output_indices( objf :: AbstractObjective, mop :: MixedMOP )
    return _output_indices(objf, mop, mop.objf_state );
end

function _del!( mop :: MixedMOP, objf :: AbstractObjective )
    position = _objf_index( objf, mop );
    deleteat!( mop.vector_of_objectives, position );
    nothing 
end

function _add!( mop :: MixedMOP, objf :: AbstractObjective, output_indices :: Union{Nothing, Vector{Int}} = nothing )
    if isnothing( output_indices )
        output_indices = let n_out = num_objectives( mop );
            collect( num_objectives + 1 : num_objectives + 1 + num_outputs(objf) )
        end;
    end
    if length(output_indices) != num_outputs(objf)
        error("Number of objective outputs does not match length of output indices!");
    end
    push!(list_of_objectives(mop), objf);
    if isnothing(mop.objf_output_mapping)
        mop.objf_output_mapping = Dict( objf => output_indices );
    else
        mop.objf_output_mapping[objf] = output_indices;
    end
    mop.objf_state = UUIDs.uuid4();
    nothing
end

function MOI.add_variable( mop :: MixedMOP )
    new_var = length(mop.vars) + 1;
    push!( mop.vars, new_var );
    mop.state_hash = UUIDs.uuid4();
    return MOI.VariableIndex( new_var )
end

MOI.get( mop :: MixedMOP, :: MOI.NumberOfVariables ) = length(mop.vars) :: Int;
MOI.get( mop :: MixedMOP, :: MOI.ListOfVariableIndices) = MOI.VariableIndex.(mop.vars);
#=
function MOI.get( mop :: MixedMOP, :: MOI.ListOfConstraints ) 
    constraints = [];
    for (i, var_int) ∈ enumerate(mop.vars) 
        if haskey( mop.lb, var ) && haskey( mop.ub, var )
            push!( constraints, ( MOI.SingleVariable, MOI.Interval ))
        end
    end
    return constraints[];
end
=#

function MOI.add_variables( mop :: MixedMOP, N :: Int )
    new_vars = [length(mop.vars) + i for i = 1 : N];
    push!( mop.vars, new_vars... );
    mop.var_state = UUIDs.uuid4();
    return MOI.VariableIndex.( new_vars )
end

function MOI.supports_constraint( ::MixedMOP, ::Type{MOI.SingleVariable}, ::Type{MOI.Interval} )
    true;
end

function MOI.add_constraint( mop :: MixedMOP, var ::F, bounds ::MOI.Interval ) where {F<:MOI.SingleVariable}
    var_int =  var.variable.value;
    mop.lb[ var_int ] = bounds.lower;
    mop.ub[ var_int ] = bounds.upper;
    mop.var_state = UUIDs.uuid4();
    return MOI.ContstraintIndex{F, MOI.Interval}
end

# TODO support half-open intervals

@memoize function _full_lb( mop :: MixedMOP, hash :: UUIDs.UUID ) 
    [ haskey(mop.lb, var_int) ? mop.lb[var_int] : -Inf for var_int ∈ mop.vars ]
end

function full_lower_bounds( mop :: MixedMOP )
    _full_lb( mop, mop.var_state )
end

@memoize function _full_ub( mop :: MixedMOP, hash :: UUIDs.UUID ) 
    [ haskey(mop.ub, var_int) ? mop.ub[var_int] : Inf for var_int ∈ mop.vars ]
end

function full_upper_bounds( mop :: MixedMOP )
    _full_ub( mop, mop.var_state )
end

# overwrite these derived methods for efficacy 
@memoize function _full_lb_internal(mop :: MixedMOP, hash :: UUIDs.UUID )
    [ isinf(l) ? l : 0.0 for l ∈ full_lower_bounds(mop) ];
end 

@memoize function _full_ub_internal(mop :: MixedMOP, hash :: UUIDs.UUID )
    [ isinf(u) ? u : 1.0 for u ∈ full_upper_bounds(mop) ];
end

full_lower_bounds_internal( mop :: MixedMOP ) = _full_lb_internal( mop , mop.var_state );
full_upper_bounds_internal( mop :: MixedMOP ) = _full_ub_internal( mop , mop.var_state );

# Evaluating and sorting objectives

# overwrite to exploit memoization
@memoize function _output_indices( mop :: MixedMOP, hash :: UUIDs.UUID )
    all_outputs = Int[];
    for objf ∈ list_of_objectives( mop )
        push!( all_outputs, output_indices( objf, mop )...);
    end
    return all_outputs;
end

function output_indices( mop :: MixedMOP )
    return _output_indices(mop, mop.objf_state )
end

@memoize function _reverse_internal_sorting_indices( mop :: MixedMOP, hash :: UUIDs.UUID )
    internal_indices = output_indices(mop);
    return sortperm( internal_indices );
end

function reverse_internal_sorting_indices(mop :: MixedMOP) 
    return _reverse_internal_sorting_indices(mop, mop.objf_state);
end

