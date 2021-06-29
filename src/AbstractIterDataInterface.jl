
############################################
# AbstractIterData
Base.broadcastable( id :: AbstractIterData ) = Ref( id )

"Return current iteration site vector ``xᵗ``."
function get_x( :: AbstractIterData{F} ) :: AbstractVector{F} where F 
    return F[]
end

"Return current value vector ``f(xᵗ)``."
function get_fx(  :: AbstractIterData{F} ) :: AbstractVector{F} where F 
    return F[]
end

"Return current trust region radius (vector) ``Δᵗ``."
function get_Δ( :: AbstractIterData{F} ) :: Union{F,AbstractVector{F}} where F
    return zero(F)
end

# setters
# (make sure to store a copy of the very first input vectors!)
function set_x!( :: AbstractIterData, x̂ :: RVec ) :: Nothing 
    nothing
end

function set_fx!( :: AbstractIterData, ŷ :: RVec ) :: Nothing 
    nothing
end

function set_Δ!( id :: AbstractIterData, Δ :: Union{Real, RVec} ) :: Nothing
    nothing
end

   
"Index (or `id`) of ``xᵗ`` in database."
get_x_index( :: AbstractIterData ) :: Int = -1

"Set the index stored in `it_dat` of ``xᵗ`` to `val`."
set_x_index!( it_dat :: AbstractIterData, val :: Int ) :: Nothing = nothing

"Return number of iterations so far."
num_iterations( :: AbstractIterData ) :: Int = 0;
"Return the number of model improvement iterations so far."
num_model_improvements( :: AbstractIterData ) :: Int = 0;

"Increase the iteration counter by `N`."
inc_iterations!( :: AbstractIterData, N :: Int = 1 ) :: Nothing = nothing ;

"Increase the model improvement counter by `N`."
inc_model_improvements!( :: AbstractIterData, N :: Int = 1 ) :: Nothing = nothing;

"Set the iteration counter to `N`."
set_iterations!( :: AbstractIterData, N :: Int = 0 ) :: Nothing = nothing ;

"Set the improvement counter to `N`."
set_model_improvements!( :: AbstractIterData, N :: Int = 0 ) :: Nothing = nothing ;

"Return the iteration classification of `ITER_TYPE`."
it_stat( :: AbstractIterData ) :: ITER_TYPE = SUCCESSFULL;
"Set the iteration classification."
it_stat!( :: AbstractIterData, :: ITER_TYPE ) :: Nothing = nothing;

# generic initializer
function init_iter_data( :: T, x :: Vec, fx :: Vec, Δ :: NumOrVec ) :: T where {T<:Type{<:AbstractIterData}}
	return nothing
end

saveable_type( :: AbstractIterData ) :: Type{<:AbstractIterSavebale} = nothing
get_saveable( id :: AbstractIterData{F}; kwargs... ):: AbstractIterSaveable = nothing

#=
function set_next_iterate!( id :: AbstractIterData, x̂ :: RVec, 
	    ŷ :: RVec ) :: Int
    set_x!(id, x̂)
    set_fx!(id, ŷ)

	return set_x_index!( id, x_index );
end

function keep_current_iterate!( id :: AbstractIterData, x̂ :: RVec, ŷ :: RVec ) :: NothInt
    return new_result!(get_db(id), x̂, ŷ)
end
=#
