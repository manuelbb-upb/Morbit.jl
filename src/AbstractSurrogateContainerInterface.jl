# AbstractSurrogateWrapper
Base.broadcastable( sw :: AbstractSurrogateWrapper ) = Ref(sw)

# constructor
init_wrapper( :: Type{<:AbstractSurrogateWrapper}, objf :: AbstractObjective, mod :: SurrogateModel, meta :: SurrogateMeta ) :: AbstractSurrogateWrapper = nothing

num_outputs( sw :: AbstractSurrogateWrapper ) :: Int = 0
get_objf( sw :: AbstractSurrogateWrapper ) :: AbstractObjective = nothing 
get_model( sw :: AbstractSurrogateWrapper ) :: SurrogateModel = nothing 
get_meta( sw :: AbstractSurrogateWrapper ) :: SurrogateMeta = nothing

# AbstractSurrogateContainer
Base.broadcastable( sc :: AbstractSurrogateContainer ) = Ref(sc)

# constructor 
function init_surrogates( :: Type{<:AbstractSurrogateContainer}, :: AbstractMOP, :: AbstractIterData, 
	:: AbstractDB, :: AbstractConfig )
	return nothing 
end

# update methods
function update_surrogates!( :: AbstractSurrogateContainer, :: AbstractMOP, :: AbstractIterData,
	:: AbstractDB; ensure_fully_linear = false ) 
	return nothing
end

function improve_surrogates!( :: AbstractSurrogateContainer, :: AbstractMOP, :: AbstractIterData,
	:: AbstractDB; ensure_fully_linear = false, kwargs... ) 
	return nothing
end

# mandatory functions

list_of_wrappers( sc :: AbstractSurrogateContainer ) :: AbstractVector{<:AbstractSurrogateWrapper } = nothing

"""
	_output_model_mapping(sc, l)

Return a `Tuple{Int,Int}` with the index `i` of the surrogate model and its output `k`
describing output `l` of an MOP with scalarized outputs `1:l`.

# Example

Suppose an MOP is setup with the 3 scalar objectives f1, f2 and f3 and 
there is one surrogate model modelling f1 and f3 (in that order) and 
a second model for f2. `sc` holds the surrogates in that order. Then:
```julia_repl
julia> _output_model_mapping(sc,1)
(1,1)
julia> _output_model_mapping(sc,2)
(2,1)
julia> _output_model_mapping(sc,3)
(1,2)
```
"""
_output_model_mapping( sc :: AbstractSurrogateContainer, l :: Int ) :: Tuple{Int, Int} = nothing

"""
	_sc_output_model_mapping(sc, ℓ)

Return a `Tuple{Int,Int}` with the index `i` of the surrogate model and its output `k`
describing output `ℓ` of `sc`, i.e., as returned by `eval_models(sc, x)`.
"""
_sc_output_model_mapping( sc :: AbstractSurrogateContainer, ℓ :: Int ) :: Tuple{Int,Int} = nothing

## Derived

# get_wrapper can be improved
get_wrapper(sc :: AbstractSurrogateContainer, i :: Int) = list_of_wrappers(sc)[i]

function num_outputs( sc :: AbstractSurrogateContainer ) :: Int
    return sum( num_outputs(sw) for sw ∈ list_of_wrappers(sc) )
end

function fully_linear( sc :: AbstractSurrogateContainer )
    return all( fully_linear(get_model(sw)) for sw ∈ list_of_wrappers(sc) )
end

"Evaluate all surrogate models stored in `sc` at scaled site `x̂`."
function eval_models( sc :: AbstractSurrogateContainer, x̂ :: Vec )
	return vcat( (eval_models( get_model(sw) , x̂) for sw ∈ list_of_wrappers(sc) )...)
end

"Evaluate Jacobian of surrogate models stored in `sc` at scaled site `x̂`."
function get_jacobian( sc :: AbstractSurrogateContainer, x̂ :: Vec )
    model_jacobians = [ get_jacobian( get_model(sw), x̂) for sw ∈ list_of_wrappers(sc) ]
    return vcat( model_jacobians... )
end

@doc """
Return a function handle to be used with `NLopt` for output `l` of `sc`.
Index `l` is assumed to be an *internal* index in the range of `1, …, n_objfs`,
where `n_objfs` is the total number of (scalarized) objectives modelled by `sc`.
"""
function get_optim_handle( sc :: AbstractSurrogateContainer, l :: Int )
    i, ℓ = sc_output_model_mapping(sc, l)
    sw = get_wrapper( sc, i )
    return _get_optim_handle( get_model(sw), ℓ )
end

function get_saveable_type( sc :: AbstractSurrogateContainer )
	return Tuple{ (get_saveable_type( get_meta(sw) ) for sw in list_of_wrappers(sc))... }
end

get_saveable( :: Nothing ) = nothing
get_saveable_type( :: Nothing ) = Nothing

function get_saveable( sc :: AbstractSurrogateContainer )
	return Tuple( get_saveable( get_meta(sw) ) for sw in list_of_wrappers(sc) )
end

# These are used for PS constraints
# one can in theory provide vector constraints but most solvers fail then
@doc """
    eval_models( sc, x̂, l )

Return model value for output `l` of `sc` at `x̂`.
Index `l` is assumed to be an *internal* index in the range of 1,…,n_objfs,
where n_objfs is the total number of (scalarized) objectives stored in `sc`.
"""
function eval_models( sc :: AbstractSurrogateContainer, x̂ :: Vec, l :: Int )
    i, ℓ = sc_output_model_mapping(sc,l)
    sw = get_wrapper(sc.surrogates, i )
    return eval_models( get_model(sw), x̂, ℓ )
end

@doc """
    get_gradient( sc, x̂, l )

Return a gradient for output `l` of `sc` at `x̂`.
Index `l` is assumed to be an internal index in the range of 1,…,n_objfs,
where n_objfs is the total number of (scalarized) objectives stored in `sc`.
"""
function get_gradient( sc :: AbstractSurrogateContainer, x̂ :: Vec, l :: Int )
    i, ℓ = sc_output_model_mapping( sc, l )
    sw = get_wrapper( sc, i )
    return get_gradient( get_model(sw), x̂, ℓ );
end
