
## Iter Data

mutable struct IterData{ 
        XT <: AbstractDictionary, 
        XS <: AbstractDictionary, 
        YT <: AbstractDictionary, 
        E <: AbstractDictionary, 
        I <: AbstractDictionary,
        ET <: AbstractDictionary, 
        IT <: AbstractDictionary, 
        DT <: NumOrVec
    }
    
    "dict mapping `i::VarInd` to variable value x[i]."
    x :: XT 
    
    "dict mapping `i::VarInd` to scaled variable value x[i]."
    x_scaled :: XS

    "dict mapping `ℓ::ObjectiveIndex` to vector of evaluations."
    fx :: YT

    "dict mapping `ConstraintIndexEq` to vector of evaluations."
    l_e :: E 

    "dict mapping `ConstraintIndexIneq` to vector of evaluations."
    l_i :: I 
    
    "dict mapping `NLConstraintIndexEq` to vector of nl-evaluations."
    c_e :: ET 
    
    "dict mapping `NLConstraintIndexIneq` to vector of nl-evaluations."
    c_i :: IT 
    
    "trust region radius, either number or dict(?)"
    Δ :: DT 
    
    x_index = Int

    # A constructor that ensures the same floating point precision in all values:
    function IterData( 
        x :: AbstractDictionary{VarInd, X},
        x_scaled :: AbstractDictionary{VarInd, S},
        fx :: AbstractDictionary{ObjectiveIndex, Y},
        l_e:: AbstractDictionary{ConstraintIndexEq, E},
        l_i:: AbstractDictionary{ConstraintIndexInEq, I},
        c_e:: AbstractDictionary{NLConstraintIndexEq, NE},
        c_i:: AbstractDictionary{NLConstraintIndexInEq, NI},
        Δ :: DT, x_index :: Int
    ) where{X,S,Y,E,I,NE,NI,DT}
        F = Base.promote_eltype( MIN_PRECISION, X, S, Y, E, I, NE, NI, DT )
        return IterData(
            F.(x), F.(x_scaled), 
            _vec_type( Y, F).(fx),
            _vec_type( E, F).(l_e),
            _vec_type( I, F).(l_i),
            _vec_type( NE, F).(c_e),
            _vec_type( NI, F).(c_i),
            F(Δ), x_index
        )
    end
end

Base.broadcastable( id :: IterData ) = Ref( id )

function _precision( :: IterData{XT,XS,YT,E,I,ET,IT,DT} ) where {XT,XS,YT,E,I,ET,IT,DT} 
    return Base.promote_eltype( MIN_PRECISION, XT,XS,YT,E,I,ET,IT,DT )
end

# ### Getters

# There are Getters for the mathematical objects relevant during optimzation:
"Return current iteration site vector ``xᵗ``."
get_x( id :: IterData, var_inds = nothing ) = collect(id.x)
get_x( id :: IterData, var_inds :: AbstractVector{<:VarInd} ) = (id.x, var_inds)
get_x_dict( id :: IterData ) = id.x

"Return current iteration site vector ``xᵗ``."
get_x_scaled( id :: IterData, var_inds = nothing ) = collect(id.x_scaled)
get_x_scaled( id :: IterData, var_inds :: AbstractVector{<:VarInd} ) = collect(get_indices(id.x_scaled, var_inds))
get_x_scaled_dict( id :: IterData ) = id.x

"Return current value vector ``f(xᵗ)``."
get_fx( id :: IterData, func_inds = nothing ) = collect(id.fx)
get_fx( id :: IterData, func_inds :: AnyIndexIterable ) = collect(get_indices(id.fx, func_inds))
get_fx_dict( id :: IterData ) = id.fx

"Return only the parts of f(x) that are relevant for `func_indices`."
function get_vals( id :: IterData, sdb, func_indices )
    x_index = get_x_index( id, func_indices )
    sub_db = get_sub_db( sdb, func_indices )
    return get_value( sub_db, x_index )
end

"Return value vector of linear equality constraints."
get_eq_const( id :: IterData, func_inds = nothing ) = collect(id.l_e)
get_eq_const( id :: IterData, func_inds :: AnyIndexIterable ) = collect(get_indices(id.l_e, func_inds))
get_eq_const_dict( id :: IterData ) = id.l_e

"Return value vector of linear inequality constraints."
get_ineq_const( id :: IterData, func_inds = nothing ) = collect(id.l_i)
get_ineq_const( id :: IterData, func_inds :: AnyIndexIterable ) = collect(get_indices(id.l_i, func_inds))
get_ineq_const_dict( id :: IterData ) = id.l_i

"Return current equality constraint vector ``cₑ(xᵗ)``."
get_nl_eq_const( id :: IterData, func_inds = nothing ) = collect(id.c_e)
get_nl_eq_const( id :: IterData, func_inds :: AnyIndexIterable ) = collect(get_indices(id.c_e, func_inds))
get_nl_eq_const_dict( id :: IterData ) = id.l_eget_nl_eq_const( id :: IterData ) = id.c_e

"Return current inequality constraint vector ``cᵢ(xᵗ)``."
get_nl_ineq_const( id :: IterData, func_inds = nothing ) = collect(id.c_i)
get_nl_ineq_const( id :: IterData, func_inds :: AnyIndexIterable ) = collect(get_indices(id.c_i, func_inds))
get_nl_ineq_const_dict( id :: IterData ) = id.l_eget_nl_eq_const( id :: IterData ) = id.c_i

"Return current trust region radius (vector) ``Δᵗ``."
get_delta( id :: IterData ) = id.Δ

# We also need the iteration result index for our sub-database.
"Index (or `id`) of current iterate in database."
get_x_index( id:: IterData, ind ) = id.x_indices[ind]
get_x_index( id:: IterData, ind :: AnyIndex ) = id.x_indices[(ind,)]
get_x_index_dict( id :: IterData ) = id.x_indices

function Base.show( io :: IO, id :: I) where I<:IterData
    str = "IterData"
    if !get(io, :compact, false)
        x = get_x(id)
        fx = get_fx(id)
        Δ = get_delta(id)
        c_e = get_nl_eq_const(id)
        c_i = get_nl_ineq_const(id)
        l_e = get_eq_const(id)
        l_i = get_ineq_const(id)
        str *= """ 
        x   $(lpad(eltype(x), 8, " ")) = $(_prettify(x))
        fx  $(lpad(eltype(fx), 8, " ")) = $(_prettify(fx))
        Δ   $(lpad(eltype(Δ), 8, " ")) = $(Δ)
        l_e $(lpad(eltype(l_e), 8, " ")) = $(_prettify(l_e))
        l_i $(lpad(eltype(l_i), 8, " ")) = $(_prettify(l_i))
        c_e $(lpad(eltype(c_e), 8, " ")) = $(_prettify(c_e))
        c_i $(lpad(eltype(c_i), 8, " ")) = $(_prettify(c_i))"""
    end
    print(io, str)
end
### Setters

# Implement the setters and note the leading underscore!
# The algorithm will actually call `set_delta!` instead of `_set_delta!` etc.
# These derived methods are implemented below and ensure that we actually 
# store copies of the input!

"Set current trust region radius (vector?) to `Δ`."
function _set_delta!( id :: IterData, Δ :: NumOrVec ) :: Nothing
    id.Δ = Δ
    return nothing
end

# ## Derived Methods

# The actual setters simply ensure proper copying (if a vector is provided):
set_delta!( id :: IterData, Δ :: NumOrVec ) = _set_delta!(id, copy(Δ))