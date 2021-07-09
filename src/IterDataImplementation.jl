
####### IterData
@with_kw mutable struct IterData{
        F<:AbstractFloat } <: AbstractIterData{F}
    x :: Vector{F} = F[]
    fx :: Vector{F} = F[]
    Δ :: Union{F, Vector{F}} = zero(F)
    
    x_index :: Int = -1
    num_iterations :: Int = 0;
    num_model_improvements :: Int = 0;
    it_stat :: ITER_TYPE = SUCCESSFULL;

end

get_x( id :: IterData ) = id.x
get_fx( id :: IterData ) = id.fx
get_Δ( id :: IterData ) = id.Δ

# setters
function set_x!( id :: IterData, x̂ :: Vec ) :: Nothing 
    id.x = copy(x̂)
    return anothing
end

function set_fx!( id :: IterData, ŷ :: Vec ) :: Nothing 
    id.fx = copy(ŷ);
    return nothing
end

function set_Δ!( id :: IterData, Δ :: NumOrVec ) :: Nothing
    id.Δ = copy(Δ);
    return nothing
end

get_x_index( id:: IterData ) = id.x_index 

function set_x_index!( id :: IterData, N :: Int ) :: Nothing
    id.x_index = N;
    nothing
end

num_iterations( id :: IterData ) :: Int = id.num_iterations;
num_model_improvements( id :: IterData ) :: Int = id.num_model_improvements;

function inc_iterations!( id :: IterData, N :: Int = 1 ) :: Nothing
    id.num_iterations += N;
    return nothing
end

function inc_model_improvements!( id :: IterData, N :: Int = 1 ) :: Nothing
    id.num_model_improvements += N ;
    return nothing
end

function set_iterations!( id :: IterData, N :: Int = 0 ) :: Nothing
    id.num_iterations = N;
    return nothing
end

function set_model_improvements!( id :: IterData, N :: Int = 0 ) :: Nothing
    id.num_model_improvements = N ;
    return nothing
end

it_stat( id :: IterData ) :: ITER_TYPE = id.it_stat;
function it_stat!( id :: IterData, t :: ITER_TYPE ) :: Nothing 
    id.it_stat = t
    return nothing
end

function init_iter_data( ::T, x :: Vec, fx :: Vec, Δ :: NumOrVec ) where T<:Type{<:IterData}
    F = Base.promote_eltype(x,fx,Δ)
    return IterData(; x = F.(x), fx = F.(x), Δ = F.(Δ))
end

####### IterSaveable
function saveable_type( :: IterData{F} ) where F 
    return IterSaveable{F}
end

@with_kw struct IterSaveable{F<:AbstractFloat } <: AbstractIterSaveable{F}
    Δ :: Union{F, Vector{F}}
    x_index :: Int
    num_iterations :: Int
    num_model_improvements
    it_stat :: ITER_TYPE 

    # additional information for stamping
    x_trial_index :: Int
    ρ :: F
    stepsize :: F
    ω :: F
end

function get_saveable( id :: IterData{F}; x_trial_index, ρ, stepsize, ω ) where F
    return IterSaveable{F}(;
        Δ = get_Δ(id),
        x_index = get_x_index(id),
        num_iterations = num_iterations(id),
        num_model_improvements = num_model_improvements(id),
        it_stat = it_stat(id),
        x_trial_index,
        ρ = F(ρ),
        stepsize = F(stepsize),
        ω = F.(ω)
    )
end
