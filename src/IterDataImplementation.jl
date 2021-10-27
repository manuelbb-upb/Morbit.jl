
####### IterData
mutable struct IterData{ 
        XT <: VecF, YT <: VecF, XS <: VecF,
        E <: VecF, I <: VecF,
        ET <: VecF, IT <: VecF, DT <: NumOrVecF, 
        XIndType, #<: Union{AbstractDict, AbstractDictionary},#{ FunctionIndexTuple, Int } 
    } <: AbstractIterate
    
    x :: XT 
    x_scaled :: XS
    fx :: YT
    l_e :: E 
    l_i :: I 
    c_e :: ET 
    c_i :: IT 
    Δ :: DT 
    
    x_indices :: XIndType
end

# getters 
get_x( id :: IterData ) = id.x
get_fx( id :: IterData ) = id.fx
get_x_scaled( id :: IterData ) = id.x_scaled
get_eq_const( id :: IterData ) = id.l_e
get_ineq_const( id :: IterData ) = id.l_i
get_nl_eq_const( id :: IterData ) = id.c_e
get_nl_ineq_const( id :: IterData ) = id.c_i
get_delta( id :: IterData ) = id.Δ

get_x_index( id:: IterData, indices :: FunctionIndexTuple ) = id.x_indices[indices]
get_x_index_dict( id :: IterData ) = id.x_indices

# setters
function _set_delta!( id :: IterData, Δ :: NumOrVec ) :: Nothing
    id.Δ = Δ
    return nothing
end

function _init_iterate( ::Type{<:IterData}, x :: VecF, 
    x_scaled :: VecF, fx :: VecF,
    l_e :: VecF, l_i :: VecF, 
    c_e :: VecF, c_i :: VecF, Δ :: NumOrVecF, x_index_mapping )
    return IterData(x, x_scaled, fx, l_e, l_i, c_e, c_i, Δ, x_index_mapping)
end

####### IterSaveable

# Also, I store all additional meta data as Float64, input gets automatically converted.
struct IterSaveable{
        XT <: VecF, D <: NumOrVecF,
        XIndType
    } <: AbstractIterSaveable

    iter_counter :: Int
    it_stat :: ITER_TYPE

    x :: XT
    Δ :: D
    x_indices :: XIndType

    # additional information for stamping
    ρ :: Float64
    stepsize :: Float64
    ω :: Float64
end

function get_saveable( :: Type{<:IterSaveable}, id :: AbstractIterate;
    iter_counter, it_stat, rho, steplength, omega)
    return IterSaveable(
        iter_counter, it_stat,
        get_x( id ),
        get_delta( id ),
        get_x_index_dict( id ),
        rho, steplength, omega
    )
end

function get_saveable_type( :: Type{<:IterSaveable}, id :: AbstractIterate )
    return IterSaveable{ 
        typeof( get_x(id) ), 
        typeof( get_delta(id) ),
        typeof( get_x_index_dict(id) )
     }
end
