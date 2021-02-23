# Implementation of the AbstractObjective interface 
# (see file `AbstractObjectiveInterface.jl`)
@with_kw mutable struct VectorObjectiveFunction <: AbstractObjective
    n_in :: Int = 0; 
    n_out :: Int = 0;
    n_evals :: Int64 = 0;   # true function evaluations (also counts fdm evaluations)

    # max_evals :: Int64 = typemax(Int64);

    model_config :: Union{ Nothing, SurrogateConfig } = nothing;

    function_handle :: Union{F, Nothing} where{F <: Function}  = nothing
end

# Required methods (see file `AbstractObjectiveInterface.jl`)
function _wrap_func( ::Type{VectorObjectiveFunction}, 
        fn::Function, model_cfg::SurrogateConfig,
        n_vars :: Int, n_out :: Int
    ) :: VectorObjectiveFunction
    
    return VectorObjectiveFunction(;
        n_in = n_vars,
        n_out = n_out,
        function_handle = fn, 
        model_config = model_cfg 
    );
end

num_vars( objf :: VectorObjectiveFunction ) = objf.n_in;

num_evals( objf :: VectorObjectiveFunction ) = objf.n_evals;
function num_evals!( objf :: VectorObjectiveFunction, N :: Int )
    objf.n_evals = N;
    nothing
end

num_outputs( objf :: VectorObjectiveFunction ) = objf.n_out;

# can_batch( objf :: VectorObjectiveFunction ) = isa( objf.function_handle, BatchObjectiveFunction );

model_cfg( objf :: VectorObjectiveFunction ) = objf.model_config;

function eval_objf_at_site( objf :: VectorObjectiveFunction, x :: Union{RVec, RVecArr}) :: Union{RVec,RVecArr}
    objf.function_handle(x);
end
