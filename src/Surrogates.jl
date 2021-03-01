
#include("RBFModel.jl")
include("ExactModel.jl")
include("TaylorModel.jl")
include("LagrangeModel.jl")
include("newRBF.jl");

struct SurrogateWrapper
    objf :: AbstractObjective;
    model :: SurrogateModel;
    meta :: SurrogateMeta;
end

@with_kw mutable struct SurrogateContainer
    surrogates :: Vector{SurrogateWrapper} = SurrogateWrapper[];
    state :: UUIDs.UUID = UUIDs.uuid4();
end

num_outputs( sw :: SurrogateWrapper ) :: Int = num_outputs(sw.objf);
@memoize function _num_outputs( sc :: SurrogateContainer, hash :: UUIDs.UUID) :: Int
    return sum( num_outputs(sw) for sw ∈ sc.surrogates )
end
num_outputs( sc :: SurrogateContainer ) :: Int = _num_outputs( sc, sc.state );
function fully_linear( sc :: SurrogateContainer )
    return all( fully_linear(sw.model) for sw ∈ sc.surrogates )
end

@memoize function _get_surrogate_from_output_index( sc :: SurrogateContainer, ℓ :: Int, 
    mop :: AbstractMOP, hash :: UUIDs.UUID ) :: Union{Nothing,Tuple{SurrogateWrapper,Int}}
    for sw ∈ sc.surrogates
        objf_out_indices = output_indices( objf, mop );
        l = findfirst( x -> x == ℓ, output_indices(objf,mop) );
        if !isnothing(l)
            return sw,l 
        end
    end
    return nothing
end

function get_surrogate_from_output_index( sc :: SurrogateContainer, ℓ :: Int, 
    mop :: AbstractMOP) :: Union{Nothing,Tuple{SurrogateWrapper,Int}}
    return _get_surrogate_from_output_index(sc, ℓ, mop, sc.state)
end

"Delete surrogate wrapper at position `si` from list `sc.surrogates`."
function delete_surrogate!( sc :: SurrogateContainer, si :: Int ) :: Nothing
    deleteat!( sc.surrogates, si )
    sc.state = UUIDs.uuid4();
    nothing 
end

function add_surrogate!(sc :: SurrogateContainer, sw :: SurrogateWrapper) :: Nothing
    push!(sc.surrogates, sw);
    sc.state = UUIDs.uuid4();
    nothing 
end

function replace_surrogate!(sc :: SurrogateContainer, si :: Int, 
    sw :: SurrogateWrapper) :: Nothing
    
    # A B C D ⇒ deleteat! 3 ⇒ A B D
    deleteat!(sc.surrogates, si);
    # A B D ⇒ insert! 3 ℂ ⇒ A B ℂ D
    insert!(sc.surrogates, si, sw);
    sc.state = UUIDs.uuid4();
    nothing 
end


@doc "Return a SurrogateContainer initialized from the information provided in `mop`."
function init_surrogates( mop :: AbstractMOP, id :: AbstractIterData, ac :: AbstractConfig ) :: SurrogateContainer
    sc = SurrogateContainer();
    for objf ∈ list_of_objectives(mop)
        model, meta = init_model( objf, mop, id, ac );
        add_surrogate!( sc, SurrogateWrapper(objf,model,meta))
    end
    return sc
end

function update_surrogates!( sc :: SurrogateContainer, mop :: AbstractMOP, 
    id :: AbstractIterData, ac :: AbstractConfig; ensure_fully_linear :: Bool = false ) :: Nothing 
    for (si,sw) ∈ enumerate(sc.surrogates)
        new_model, new_meta = update_model( sw.model, sw.objf, sw.meta, mop, id, ac; ensure_fully_linear )
        new_sw = SurrogateWrapper( 
            sw.objf,
            new_model, 
            new_meta
        );
        replace_surrogate!(sc, si, new_sw )
    end
    nothing
end

function improve_surrogates!( sc :: SurrogateContainer, mop :: AbstractMOP, 
    id :: AbstractIterData, ac :: AbstractConfig; ensure_fully_linear :: Bool = false ) :: Nothing 
    for (si,sw) ∈ enumerate(sc.surrogates)
        new_model, new_meta = improve_model( sw.model, sw.objf, sw.meta, mop, id, ac; ensure_fully_linear )
        new_sw = SurrogateWrapper( 
            sw.objf,
            new_model, 
            new_meta
        );
        replace_surrogate!(sc, si, new_sw )
    end
    nothing
end

function eval_models( sc :: SurrogateContainer, x̂ :: RVec ) :: RVec
    vcat( (eval_models(sw.model , x̂) for sw ∈ sc.surrogates )...)
end

function get_jacobian( sc :: SurrogateContainer, x̂ :: RVec) :: RMat
    model_jacobians = [ get_jacobian(sw.model, x̂) for sw ∈ sc.surrogates ]
    vcat( model_jacobians... )
end

#=

function improve!(sc :: SurrogateContainer, non_linear_indices :: Vector{Int64},
        ac :: AlgoConfig )
    for ℓ ∈ non_linear_indices
        model = sc.model_list[ℓ]
        meta = sc.model_meta[ℓ]
        objf = sc.objf_list[ℓ]
        improve!(model, meta, ac, objf, objf.model_config)
    end
end

function make_linear!(sc :: SurrogateContainer, non_linear_indices :: Vector{Int64},
        ac :: AlgoConfig )
    has_changed = false
    for ℓ ∈ non_linear_indices
        model = sc.model_list[ℓ]
        meta = sc.model_meta[ℓ]
        objf = sc.objf_list[ℓ]
        has_changed *= make_linear!(model, meta, ac, objf, objf.model_config)
    end
    return has_changed
end

function eval_models( sc :: SurrogateContainer, x :: Vector{Float64} )
    vcat( ( eval_models(model , x) for model ∈ sc.model_list )...)
end

function get_jacobian( sc :: SurrogateContainer, x :: Vector{Float64})
    model_jacobians = [ get_jacobian(model , x) for model ∈ sc.model_list ]
    vcat( model_jacobians... )
end

# let each surrogate handle its broadcasting behavior itself...
# # (ExactModel might be BatchObjectiveFunction)
function broadcasted( f :: typeof(eval), sc::SurrogateContainer, X :: Vector{Vector{Float64}} )
    [ vcat(z...) for z ∈ zip( [ eval.(model, X) for model ∈ sc.model_list ]... ) ]
end

@doc """
Return `true` (and `[]`) if all models in `sc.model_list` qualify as fully linear.
Return `false` (and a list of the violating indices wrt `sc.model_list`) elsewise.
"""
function fully_linear( sc :: SurrogateContainer )
    linear_flag = true;
    non_linear_indices = Int64[];
    for (model_index, model) ∈ enumerate(sc.model_list)
        if !fully_linear(model)
            linear_flag = false
            push!(non_linear_indices, model_index);
        end
    end
    return linear_flag, non_linear_indices
end

@doc """
Return a function handle to be used with `NLopt` for output `l` of `model`.
Assume `l` to be in [1, …, `model.n_out`].
"""
function get_optim_handle( model :: M where{ M <: SurrogateModel}, l :: Int64 )
    function (x :: Vector{Float64}, g :: Vector{Float64})
        if !isempty(g)
            g[:] = get_gradient( model, x, l)
        end
        return eval_models( model, x, l)
    end
end

@doc """
Return a function handle to be used with `NLopt` for output `ℓ` of `sc`.
Index `ℓ` is assumed to be an internal index in the range of 1,…,n_objfs,
where n_objfs is the total number of (scalarized) objectives stored in `sc`.
"""
function get_optim_handle( sc :: SurrogateContainer, ℓ :: Int64 )
    i,l = sc.output_to_objf_index[ℓ]
    return get_optim_handle( sc.model_list[i], l )
end

@doc """
Return a  gradient handle to be used with `NLopt` for output `ℓ` of `sc`.
Index `ℓ` is assumed to be an internal index in the range of 1,…,n_objfs,
where n_objfs is the total number of (scalarized) objectives stored in `sc`.
"""
function get_gradient( sc :: SurrogateContainer, ξ :: Vector{Float64}, ℓ :: Int64 )
    i,l = sc.output_to_objf_index[ℓ]
    return get_gradient( sc.model_list[i], ξ, l )
end

@doc "Output ℓ of model container at site x."
function eval_models( sc :: SurrogateContainer, x :: Vector{Float64}, ℓ :: Int64 )
    i,l = sc.output_to_objf_index[ℓ]
    return eval_models( sc.model_list[i], x, l )
end
#end#module

=#