
####### IterData
@with_kw mutable struct IterData{ XT, YT, DT } <: AbstractIterData{XT,YT,DT}
    x :: XT = MIN_PRECISION[]
    fx :: YT = MIN_PRECISION[]
    Δ :: DT = zero(MIN_PRECISION)
    
    x_index :: Int = -1
    num_iterations :: Int = 0
    num_model_improvements :: Int = 0
    it_stat :: ITER_TYPE = SUCCESSFULL
end

# getters 
get_x( id :: IterData ) = id.x
get_fx( id :: IterData ) = id.fx
get_delta( id :: IterData ) = id.Δ

get_x_index( id:: IterData ) = id.x_index 

it_stat( id :: IterData ) = id.it_stat

# setters
function _set_x!( id :: IterData, x :: Vec ) :: Nothing 
    id.x = x
    return nothing
end

function set_fx!( id :: IterData, y :: Vec ) :: Nothing 
    id.fx = y
    return nothing
end

function set_delta!( id :: IterData, Δ :: NumOrVec ) :: Nothing
    id.Δ = Δ
    return nothing
end


function set_x_index!( id :: IterData, N :: Int ) :: Nothing
    id.x_index = N;
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

function init_iter_data( ::Type{<:IterData}, x :: Vec, fx :: Vec, Δ :: NumOrVec )
    return IterData(; x, fx, Δ )
end

####### IterSaveable

# Note: I don't save x,fx, so i set some arbritrary types for XT and YT
# Also, I store all additional meta data as Float64, input gets automatically converted.
struct IterSaveable{
        DT <: NumOrVecF, C <: ContainerSaveable } <: AbstractIterSaveable{Vec64,Vec64,DT,C}
    
    Δ :: DT

    x_index :: Int
    num_iterations :: Int
    num_model_improvements
    it_stat :: ITER_TYPE 

    # additional information for stamping
    x_trial_index :: Int
    ρ :: Float64
    stepsize :: Float64
    ω :: Float64
    
    model_meta :: C
end

function get_saveable( :: Type{<:IterSaveable}, id :: AbstractIterData;
    x_trial_index, ρ, stepsize, ω, sc = nothing, kwargs... )
    return IterSaveable(
        get_delta(id),
        get_x_index(id),
        get_num_iterations(id),
        get_num_model_improvements(id),
        it_stat(id),
        x_trial_index, ρ,stepsize, ω,
        get_saveable(sc)
    )
end
