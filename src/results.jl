
############################################
# AbstractIterData
Base.broadcastable( id :: AbstractIterData ) = Ref( id )

"Return current iteration site vector."
function xᵗ( :: AbstractIterData{F} ) :: AbstractVector{F} where F 
    return F[]
end

"Return current value vector."
function fxᵗ(  :: AbstractIterData{F} ) :: AbstractVector{F} where F 
    return F[]
end

"Return current trust region radius (vector)."
function Δᵗ( :: AbstractIterData{F} ) :: Union{F,AbstractVector{F}} where F
    return zero(F)
end
    
function db( :: AbstractIterData{F} ) :: AbstractDB{F}
    return MockDB{F}()
end

num_iterations( :: AbstractIterData ) :: Int = 0;
num_model_improvements( :: AbstractIterData ) :: Int = 0;

inc_iterations!( :: AbstractIterData, N :: Int = 1 ) :: Nothing = nothing ;
inc_model_improvements!( :: AbstractIterData, N :: Int = 1 ) :: Nothing = nothing;
e
set_iterations!( :: AbstractIterData, N :: Int = 0 ) :: Nothing = nothing ;
set_model_improvements!( :: AbstractIterData, N :: Int = 0 ) :: Nothing = nothing ;

it_stat( :: AbstractIterData ) :: ITER_TYPE = SUCCESSFULL;
it_stat!( :: AbstractIterData, :: ITER_TYPE ) :: Nothing = nothing;

function xᵗ_index( :: AbstractIterData ) :: NothInt XInt(nothing) end;
function xᵗ_index!( :: AbstractIterData, :: Int ) :: Nothing nothing end;

# setters
# (make sure to store a copy of the input vectors!)
function xᵗ!( id :: AbstractIterData, x̂ :: RVec ) :: Nothing 
    nothing
end

function fxᵗ!( id :: AbstractIterData, ŷ :: RVec ) :: Nothing 
    nothing
end

function Δᵗ!( id :: AbstractIterData, Δ :: Union{Real, RVec} ) :: Nothing
    nothing
end

# generic initializer
init_iter_data( ::Type{<:AbstractIterData}, x :: RVec, fx :: RVec, Δ :: Union{Real, RVec}, 
    db :: Union{AbstractDB,Nothing}) :: AbstractIterData = nothing ;

function set_next_iterate!( id :: AbstractIterData, x̂ :: RVec, 
    ŷ :: RVec ) :: NothInt
    xᵗ!(id, x̂);
    fxᵗ!(id, ŷ);

    x_index = add_result!(db(id), init_res( Result, x̂, ŷ, nothing));
    xᵗ_index!( id, x_index );

    return x_index 
end

function keep_current_iterate!( id :: AbstractIterData, x̂ :: RVec, ŷ :: RVec ) :: NothInt
    return add_result!(db(id),init_res( Result, x̂, ŷ, nothing) );    
end
