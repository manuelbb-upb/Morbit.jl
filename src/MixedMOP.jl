
# Implementation of the AbstractMOP interface 
# Formerly, this was the main problem type used internally.
# But it is not strongly typed and mutable.
# For performance, we convert it to a `StaticMOP` when 
# the user is done.
# (see file `AbstractMOPInterface.jl`)

@with_kw mutable struct MixedMOP <: AbstractMOP{true}
    vars :: Vector{Int} = [];

    vector_of_objectives :: Vector{ <:AbstractObjective } = AbstractObjective[]
    objf_output_mapping :: Dict{Any,Vector{Int}} = Dict{Any,Vector{Int}}()

    lb :: Dict{ Int, Real } = Dict{Int,Real}();
    ub :: Dict{ Int, Real } = Dict{Int,Real}();

    # change whenever vars or constraints are added
    var_state :: UUIDs.UUID = UUIDs.uuid4();      # TODO maybe use automatic hashing? pro: no need to implement everything, cons: slower?
    objf_state :: UUIDs.UUID = UUIDs.uuid4();
end

# legacy constructors
function MixedMOP( lb :: Vec, ub :: Vec )
    @assert length(lb) == length(ub);
    @assert all( lb .<= ub );
    @assert length(lb) > 0;     # for unconstrained problems, we need the number of variables

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

function MixedMOP(n_vars :: Int) 
    return MixedMOP(;
        vars = collect(1:n_vars)
    );
end

list_of_objectives( mop :: MixedMOP ) = mop.vector_of_objectives;

function output_indices( objf :: AbstractObjective, mop :: MixedMOP )
    return mop.objf_output_mapping[ objf ]
end

function _del!( mop :: MixedMOP, objf :: AbstractObjective )
    position = _objf_index( objf, mop );
    deleteat!( mop.vector_of_objectives, position );
    nothing 
end

function _add!( mop :: MixedMOP, objf :: AbstractObjective, out_indices :: Union{Nothing, Vector{Int}} = nothing )
    if isnothing( out_indices )
        out_indices = let n_out = num_objectives( mop );
            collect( n_out + 1 : n_out + num_outputs(objf) )
        end
    end
    if length(out_indices) != num_outputs(objf)
        error("Number of objective outputs does not match length of output indices!");
    end
    push!(list_of_objectives(mop), objf);
    mop.objf_output_mapping[objf] = out_indices;
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

function MOI.get( mop :: MixedMOP, :: MOI.ListOfConstraints ) :: Vector{Tuple}
    constraints = [];
    for (i, var_int) ∈ enumerate(mop.vars) 
        if haskey( mop.lb, var_int ) && haskey( mop.ub, var_int ) &&
            !isinf(mop.lb[var_int]) && !isinf(mop.ub[var_int]) 
            # TODO: what to do with half-open intervals?
            push!( constraints, ( MOI.SingleVariable, MOI.Interval ))
        end
    end
    return constraints;
end

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

@memoize ThreadSafeDict function _full_lb( mop :: MixedMOP, hash :: UUIDs.UUID ) 
    [ haskey(mop.lb, var_int) ? mop.lb[var_int] : -Inf for var_int ∈ mop.vars ]
end

function full_lower_bounds( mop :: MixedMOP )
    _full_lb( mop, mop.var_state )
end

@memoize ThreadSafeDict function _full_ub( mop :: MixedMOP, hash :: UUIDs.UUID ) 
    [ haskey(mop.ub, var_int) ? mop.ub[var_int] : Inf for var_int ∈ mop.vars ]
end

function full_upper_bounds( mop :: MixedMOP )
    _full_ub( mop, mop.var_state )
end

# overwrite these derived methods for efficacy 
@memoize ThreadSafeDict function _full_lb_internal(mop :: MixedMOP, hash :: UUIDs.UUID )
    [ isinf(l) ? l : 0.0 for l ∈ full_lower_bounds(mop) ];
end 

@memoize ThreadSafeDict function _full_ub_internal(mop :: MixedMOP, hash :: UUIDs.UUID )
    [ isinf(u) ? u : 1.0 for u ∈ full_upper_bounds(mop) ];
end

full_lower_bounds_internal( mop :: MixedMOP ) = _full_lb_internal( mop , mop.var_state );
full_upper_bounds_internal( mop :: MixedMOP ) = _full_ub_internal( mop , mop.var_state );

# Evaluating and sorting objectives

# overwrite to exploit memoization
@memoize ThreadSafeDict function _output_indices( mop :: MixedMOP, hash :: UUIDs.UUID )
    all_outputs = Int[];
    for objf ∈ list_of_objectives( mop )
        push!( all_outputs, output_indices( objf, mop )...);
    end
    return all_outputs;
end

function output_indices( mop :: MixedMOP )
    return _output_indices(mop, mop.objf_state )
end

@memoize ThreadSafeDict function _reverse_internal_sorting_indices( mop :: MixedMOP, hash :: UUIDs.UUID )
    internal_indices = output_indices(mop);
    return sortperm( internal_indices );
end

function reverse_internal_sorting_indices(mop :: MixedMOP) 
    return _reverse_internal_sorting_indices(mop, mop.objf_state);
end
