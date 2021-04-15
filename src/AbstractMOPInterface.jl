# needs `AbstractObjective` (which in turn needs the Surrogate Interface)
Broadcast.broadcastable( mop :: AbstractMOP ) = Ref( mop );

# MANDATORY methods
full_lower_bounds( :: AbstractMOP ) = nothing :: RVec;
full_upper_bounds( :: AbstractMOP ) = nothing :: RVec;

list_of_objectives( :: AbstractMOP ) = nothing :: Vector{<:AbstractObjective};

"Remove an objective function from MOP."
_del!(::AbstractMOP, ::AbstractObjective) = nothing :: Nothing;

"Add an objective function to MOP with specified output indices."
_add!(::AbstractMOP, ::AbstractObjective, ::Union{Nothing,Vector{Int}}) = nothing :: Nothing;

# MOI METHODS 
MOI.add_variable( :: AbstractMOP ) = nothing :: MOI.VariableIndex
MOI.add_variables( :: AbstractMOP ) = nothing :: Vector{MOI.VariableIndex}

MOI.get( :: AbstractMOP, :: MOI.NumberOfVariables ) = -1;
MOI.get( :: AbstractMOP, :: MOI.ListOfVariableIndices) = -1;
MOI.get( :: AbstractMOP, :: MOI.ListOfConstraints )::Vector{Tuple} = nothing ;

MOI.supports_constraint( ::AbstractMOP, ::Type{MOI.SingleVariable}, ::Type{MOI.Interval}) = nothing ::Bool;
MOI.add_constraint(::AbstractMOP, func::F, set::S) where {F,S} = nothing :: MOI.ConstraintIndex{F,S} 

# DERIVED methods 
num_vars( mop :: AbstractMOP ) = MOI.get( mop, MOI.NumberOfVariables() );

@memoize ThreadSafeDict function _width( lb :: RVec, ub :: RVec ) :: RVec
    w = ub .- lb
    w[ isinf.(w) ] .= 1
    return w
end

"Number of scalar-valued objectives of the problem."
function num_objectives( mop :: AbstractMOP )
    let objf_list = list_of_objectives(mop);
        isempty(objf_list) ? 0 : sum( num_outputs(objf) for objf ∈ objf_list )
    end
end

function _scale!( x :: RVec, lb :: RVec, ub :: RVec )
    for (i,var_bounds) ∈ enumerate(zip( lb, ub ))
        if !(isinf(var_bounds[1]) || isinf(var_bounds[2]))
            x[i] -= var_bounds[1]
            x[i] /= ( var_bounds[2] - var_bounds[1] )
        end
    end
    nothing
end

function _scale( x :: RVec, lb :: RVec, ub :: RVec )
    χ = copy(x);
    _scale!(χ, lb, ub);
    return χ
end

function _unscale!( x̂ :: RVec, lb :: RVec, ub :: RVec )
    for (i,var_bounds) ∈ enumerate(zip( lb, ub ))
        if !(isinf(var_bounds[1]) || isinf(var_bounds[2]))
            # TODO: Make the component scaling memoized?
            x̂[i] *= (var_bounds[2] - var_bounds[1]) 
            x̂[i] += var_bounds[1];
        end
    end
    nothing
end

function _unscale( x̂ :: RVec, lb :: RVec, ub :: RVec )
    χ̂ = copy( x̂ );
    _unscale!(χ̂, lb, ub);
    return χ̂;
end

"Scale variables fully constrained to a closed interval to [0,1] internally."
function scale( x :: RVec, mop :: AbstractMOP )
    x̂ = copy(x);
    lb, ub = full_lower_bounds(mop), full_upper_bounds(mop);
    _scale!(x̂, lb, ub);
    return x̂
end

"Reverse scaling for fully constrained variables from [0,1] to their former domain."
function unscale( x̂ :: RVec, mop :: AbstractMOP )
    x = copy(x̂);
    lb, ub = full_lower_bounds(mop), full_upper_bounds(mop);
    _unscale!(x, lb, ub);
    return x
end

function scale!( x :: RVec, mop :: AbstractMOP )
    lb, ub = full_lower_bounds(mop), full_upper_bounds(mop);
    _scale!(x, lb, ub);    
end

function unscale!( x̂ :: RVec, mop :: AbstractMOP )
    lb, ub = full_lower_bounds(mop), full_upper_bounds(mop);
    _unscale!( x̂, lb, ub);
end

"Position of `objf` in `list_of_objectives(mop)`."
function _objf_index( objf :: AbstractObjective, mop :: AbstractMOP)
    return findfirst( list_of_objectives(mop) .== objf ); 
end

function output_indices( mop :: AbstractMOP )
    all_outputs = Int[];
    for objf ∈ list_of_objectives( mop )
        push!( all_outputs, output_indices( objf, mop )...);
    end
    return all_outputs;
end

# TODO use memoization in MixedMOP here
function output_indices( objf ::AbstractObjective, mop :: AbstractMOP ) 
    return let first_index = _objf_index(objf,mop);
    collect( first_index : first_index + num_outputs(objf) );
    end
end

"Remove `objf` from `list_of_objectives(mop)` and return its output indices."
function pop_objf!( mop :: AbstractMOP, objf :: AbstractObjective )
    oi = output_indices( objf, mop );
    _del!(mop, objf)
    return oi
end

"Return lower variable bounds for scaled variables."
function full_lower_bounds_internal( mop :: AbstractMOP )
    [ isinf(l) ? l : 0.0 for l ∈ full_lower_bounds(mop) ];
end

"Return upper variable bounds for scaled variables."
function full_upper_bounds_internal( mop :: AbstractMOP )
    [ isinf(u) ? u : 1.0 for u ∈ full_upper_bounds(mop) ];
end

function full_bounds( mop :: AbstractMOP )
    (full_lower_bounds(mop), full_upper_bounds(mop))
end

function full_bounds_internal( mop :: AbstractMOP )
    (full_lower_bounds_internal(mop), full_upper_bounds_internal(mop))
end

"Return lower and upper bound vectors combining global and trust region constraints."
function _local_bounds( x :: RVec, Δ :: Union{Real, RVec}, lb :: RVec, ub :: RVec )
    lb_eff = max.( lb, x .- Δ );
    ub_eff = min.( ub, x .+ Δ );
    return lb_eff, ub_eff 
end

"Local bounds vectors `lb_eff` and `ub_eff` using scaled variable constraints from `mop`."
function local_bounds( mop :: AbstractMOP, x :: RVec, Δ :: Union{Real, RVec} )
    lb, ub = full_lower_bounds_internal( mop ), full_upper_bounds_internal( mop );
    return _local_bounds( x, Δ, lb, ub );
end

"Return smallest positive and biggest negative and `σ₊` and `σ₋` so that `x .+ σ± .* d` stays within bounds."
function _intersect_bounds( x :: RVec, d :: RVec, lb :: RVec, ub :: RVec ) :: Tuple{Real,Real}
    non_zero_dir = ( d .!= 0);
    if any( non_zero_dir )
       # how much can we go in positive direction 
        σ_lb = (lb[non_zero_dir] .- x[non_zero_dir]) ./ d[non_zero_dir];
        σ_ub = (ub[non_zero_dir] .- x[non_zero_dir]) ./ d[non_zero_dir];
       
        # sort so that first column contains the smallest factor 
        # we are allowed to move along each coordinate, second the largest factor
        smallest_largest = sort( [σ_lb σ_ub], dims = 2 );    

        σ_pos = minimum( smallest_largest[:, 2] );
        σ_neg = maximum( smallest_largest[:, 1] );
        return σ_pos, σ_neg
    else
        return typemax(valtype(x)), typemin(valtype(x))
    end
end

function intersect_bounds( mop :: AbstractMOP, x :: RVec, Δ :: Union{Real, RVec}, 
    d :: RVec; return_vals :: Symbol = :both ) :: Union{Real, Tuple{Real,Real}}
    x
    lb_eff, ub_eff = local_bounds( mop, x, Δ );

    σ_pos, σ_neg = _intersect_bounds( x, d, lb_eff, ub_eff );

    if return_vals == :both 
        return σ_pos, σ_neg
    elseif return_vals == :pos
        return σ_pos 
    elseif return_vals == :neg 
        return σ_neg 
    elseif return_vals == :absmax
        if abs(σ_pos) >= abs(σ_neg)
            return σ_pos
        else
            return σ_neg 
        end 
    end
end

# wrapper to unscale x̂ from internal domain and safeguard against boundary violations
struct TransformerFn <: Function 
    mop :: AbstractMOP
end

function _transform_unscale( x̂ :: RVec, lb :: RVec, ub :: RVec )
    x = copy(x̂);
    _unscale!(x, lb, ub);
    return min.( max.( lb , x), ub )
end

using LinearAlgebra: diagm 
function _jacobian_unscaling( tfn :: TransformerFn, x̂ :: RVec)
    # for our simple bounds scaling the jacobian is diagonal.
    lb, ub = full_bounds(tfn.mop)
    w = _width(lb,ub)
    J = diagm(w)
end

"Unscale the point `x̂` from internal to original domain."
function (sgf:: TransformerFn)( x̂ :: RVec )
    lb, ub = full_bounds( sgf.mop );
    return _transform_unscale( x̂, lb, ub);
end

# used in special broadcast to only retrieve bounds once
function ( sgf ::TransformerFn)( X :: RVecArr )
    lb, ub = full_bounds(sgf.mop);
    return [ _transform_unscale( x, lb, ub ) for x ∈ X ]
end

@memoize ThreadSafeDict function _get_transformer_fn( mop :: AbstractMOP )
    TransformerFn(mop)
end

function _add_objective!( mop :: AbstractMOP, T :: Type{<:AbstractObjective},
    func :: Function, model_cfg :: SurrogateConfig, n_out :: Int = 0, 
    can_batch :: Bool = false )

    # use a transformer to be able to directly evaluate scaled variables 
    tfn = _get_transformer_fn(mop);

    fx = can_batch ? BatchObjectiveFunction(func ∘ tfn) : vec ∘ func ∘ tfn;

    objf = _wrap_func( T, fx, model_cfg, num_vars(mop), n_out );
    
    out_indices = let oi = output_indices(mop);
        max_out = isempty( oi ) ? 1 : maximum( oi ) + 1;
        collect(max_out : max_out + n_out - 1)
    end;
    for other_objf ∈ list_of_objectives(mop)
        if combinable( objf, other_objf )
            other_output_indices = pop_objf!( mop, other_objf );
            out_indices = [other_output_indices; out_indices];
            objf = combine(other_objf, objf);
            break;
        end
    end
    _add!(mop, objf, out_indices);
    return num_objectives( mop ); 
end

"Return index vector so that an internal objective vector is sorted according to the order the objectives where added."
function reverse_internal_sorting_indices(mop :: AbstractMOP) 
    internal_indices = output_indices(mop);
    return sortperm( internal_indices );
end

"Sort an interal objective vector so that the objectives are in the order in which they were added."
function reverse_internal_sorting( ŷ :: RVec, mop :: AbstractMOP )
    reverse_indices = reverse_internal_sorting_indices(mop)
    return ŷ[ reverse_indices ];
end

function apply_internal_sorting( y :: RVec, mop :: AbstractMOP )
    return y[ output_indices(mop) ]
end

# custom broadcast to only retrieve sorting indices once
function Broadcast.broadcasted( :: typeof( reverse_internal_sorting ), mop :: AbstractMOP, Ŷ :: RVecArr )
    reverse_indices = reverse_internal_sorting_indices(mop);
    return [ ŷ[reverse_indices] for ŷ ∈ Ŷ];
end

"(Internally) Evaluate all objectives at site `x̂::RVec`. Objective order might differ from order in which they were added."
function eval_all_objectives( mop :: AbstractMOP,  x̂ :: RVec )
    #x = _get_transformer_fn(mop)(x̂);
    vcat( [ eval_objf( objf, x̂ ) for objf ∈ list_of_objectives(mop) ]... )
end

function Broadcast.broadcasted(::typeof(eval_all_objectives), mop :: AbstractMOP, X :: RVecArr )
    if isempty(X)
        return RVec[]
    else
        all_vec_objfs = list_of_objectives(mop);
        b_res = Vector{RVecArr}(undef, length(all_vec_objfs));
        for (i,objf) ∈ enumerate(all_vec_objfs)
            b_res[i] = eval_objf.(objf, X)
        end

        # stack the results
        N = length(X);
        ret_res = Vector{RVec}(undef, N)
        for i = 1:N
            ret_res[i] = vcat( (r[i] for r ∈ b_res )...)
        end
        return ret_res
    end
end

"Evaluate all objectives at site `x̂::RVec` and sort the result according to the order in which objectives were added."
function eval_and_sort_objectives(mop :: AbstractMOP, x̂ :: RVec)
    ŷ = eval_all_objectives(mop, x̂);
    return reverse_internal_sorting( ŷ, mop );
end

function Broadcast.broadcasted( ::typeof(eval_and_sort_objectives), mop :: AbstractMOP, X :: RVecArr )
    R = eval_all_objectives.(mop, X);
    reverse_internal_sorting!.(R, mop)
    return R
end

# Helper functions …
function num_evals( mop :: AbstractMOP ) :: Vector{Int}
    [ num_evals(objf) for objf ∈ list_of_objectives(mop) ]
end

@doc "Set evaluation counter to 0 for each VectorObjectiveFunction in `m.vector_of_objectives`."
function reset_evals!(mop :: AbstractMOP)
    for objf ∈ list_of_objectives( mop )
        num_evals!( objf, 0)
    end
end

function max_evals!( mop :: AbstractMOP, M :: Int64 )
    for objf ∈ list_of_objectives(mop)
        max_evals!( objf, min( max_evals(objf), M ) )
    end
end

#=
@memoize ThreadSafeDict function internal_output_indices_of_type( mop :: AbstractMOP, T::Type{<:SurrogateConfig} )
    indices = Int[];
    offset = 1;
    for ( objf_index, objf ) ∈ enumerate( mop.vector_of_objectives )
        if isa( model_cfg(objf), T ) 
            push!( indices, ( offset : (offset - 1 + num_outputs( objf ) ) )... )
        end
        index_counter += num_outputs(objf)
    end
    return indices
end
=#
#=
@doc "Set field `non_exact_indices` of argument `mop::MixedMOP`."
function set_non_exact_indices!( mop :: MixedMOP )
    mop.non_exact_indices = [];
    index_counter = 1;
    for ( objf_index, objf ) ∈ enumerate( mop.vector_of_objectives )
        if !isa( objf.model_config, ExactConfig )
            push!(mop.non_exact_indices, ( index_counter : ( index_counter + length(objf.internal_indices) - 1))...)
        end
        index_counter += length(objf.internal_indices);
    end
end
=#

# use for finite (e.g. local) bounds only
_rand_box_point(lb::RVec, ub::RVec, type :: Type{<:Real} = Float16)::RVec = lb .+ (ub .- lb) .* rand(type, length(lb));
