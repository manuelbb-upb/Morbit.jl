
####### IterData
@with_kw mutable struct IterData{ 
        XT <: VecF, YT <: VecF, XS <: VecF,
        E <: VecF, I <: VecF,
        ET <: VecF, IT <: VecF, DT <: NumOrVecF, 
        XIndType, #<: AbstractDict{ FunctionIndexTuple, Int } 
    } <: AbstractIterData
    
    x :: XT = MIN_PRECISION[]
    fx :: YT = MIN_PRECISION[]
    x_scaled :: XS = MIN_PRECISION[]
    l_e :: E = MIN_PRECISION[]
    l_i :: I = MIN_PRECISION[]
    c_e :: ET = MIN_PRECISION[] 
    c_i :: IT = MIN_PRECISION[]
    Δ :: DT = zero(MIN_PRECISION)
    
    x_indices :: XIndType = Dict()
    num_iterations :: Int = 0
    num_model_improvements :: Int = 0
    it_stat :: ITER_TYPE = SUCCESSFULL
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

it_stat( id :: IterData ) = id.it_stat

# setters
function _set_x!( id :: IterData, x :: Vec ) :: Nothing 
    id.x = x
    return nothing
end

function _set_x_scaled!( id :: IterData, x_scaled :: Vec ) :: Nothing 
    id.x_scaled = x_scaled
    return nothing
end

function _set_fx!( id :: IterData, y :: Vec ) :: Nothing 
    id.fx = y
    return nothing
end

function _set_eq_const!( id :: IterData, c :: Vec ) :: Nothing 
    id.l_e = c
    return nothing
end

function _set_ineq_const!( id :: IterData, c :: Vec ) :: Nothing 
    id.l_i = c
    return nothing
end

function _set_nl_eq_const!( id :: IterData, c :: Vec ) :: Nothing 
    id.c_e = c
    return nothing
end

function _set_nl_ineq_const!( id :: IterData, c :: Vec ) :: Nothing 
    id.c_i = c
    return nothing
end

function _set_delta!( id :: IterData, Δ :: NumOrVec ) :: Nothing
    id.Δ = Δ
    return nothing
end

function set_x_index!( id :: IterData, key_indices :: FunctionIndexTuple, N :: Int ) :: Nothing
    id.x_indices[key_indices] = N
    nothing
end

get_num_iterations( id :: IterData ) :: Int = id.num_iterations
get_num_model_improvements( id :: IterData ) :: Int = id.num_model_improvements

function set_num_iterations!( id :: IterData, N :: Int = 0 ) :: Nothing
    id.num_iterations = N
    return nothing
end

function set_num_model_improvements!( id :: IterData, N :: Int = 0 ) :: Nothing
    id.num_model_improvements = N
    return nothing
end

function it_stat!( id :: IterData, t :: ITER_TYPE ) :: Nothing 
    id.it_stat = t
    return nothing
end

function _init_iter_data( ::Type{<:IterData}, x :: VecF, fx :: VecF,
    x_scaled :: VecF,
    l_e :: VecF, l_i :: VecF, 
    c_e :: VecF, c_i :: VecF, Δ :: NumOrVecF, x_index_mapping; kwargs... )
    return IterData(; x, fx, x_scaled,l_e, l_i, c_e, c_i, Δ, x_indices = x_index_mapping )
end

####### IterSaveable

# Also, I store all additional meta data as Float64, input gets automatically converted.
struct IterSaveable{
        XT <: VecF, YT <: VecF, 
        E <: VecF, I <: VecF,
        ET <: VecF, IT <: VecF, DT <: NumOrVecF } <: AbstractIterSaveable
    
    x :: XT 
    fx :: YT
    l_e :: E
    l_i :: I
    c_e :: ET
    c_i :: IT
    Δ :: DT

    num_iterations :: Int
    num_model_improvements :: Int
    it_stat :: ITER_TYPE 

    # additional information for stamping
    ρ :: Float64
    stepsize :: Float64
    ω :: Float64
    
end

function get_saveable( :: Type{<:IterSaveable}, id :: AbstractIterData;
    ρ = -1.0, stepsize = -1.0, ω = -1.0, kwargs... )
    return IterSaveable(
        get_x(id),
        get_fx(id),
        get_eq_const(id),
        get_ineq_const(id),
        get_nl_eq_const(id),
        get_nl_ineq_const(id),
        get_delta(id),
        get_num_iterations(id),
        get_num_model_improvements(id),
        it_stat(id),
        ρ,stepsize, ω
    )
end
