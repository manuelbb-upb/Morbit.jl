# Implementation of the AbstractVecFun interface 
# (see file `AbstractVecFunInterface.jl`)

include("diff_wrappers.jl")

@with_kw struct VecFun{
        SC <: SurrogateConfig,
        D <: Union{Nothing,DiffFn},
        F <: VecFuncWrapper
    } <: AbstractVecFun{SC}

    n_out :: Int = 0

    model_config :: SC

    function_handle :: F 

    diff_wrapper :: D = nothing
end

# Required methods (see file `AbstractVecFunInterface.jl`)
function _wrap_func( ::Type{<:VecFun}, fn :: VecFuncWrapper; 
        model_cfg::SurrogateConfig, n_out :: Int, can_batch :: Bool = false, 
        gradients :: Union{Nothing,Function,AbstractVector{<:Function}} = nothing, 
        jacobian :: Union{Nothing,Function} = nothing, 
        hessians :: Union{Nothing,AbstractVector{<:Function}} = nothing,
        diff_method :: Union{Type{<:DiffFn}, Nothing} = FiniteDiffWrapper
    )
      
    if needs_gradients( model_cfg ) && ( isnothing(gradients) && isnothing(jacobian) )
        if isnothing(diff_method)
            error("""
            According to `model_cfg` we need gradient information.
            You can provide a list of functions with the `gradients` keyword or a 
            `jacobian` function. 
            Alternatively, you can use the keyword argument `diff_method` with 
            `Morbit.FiniteDiffWrapper` or `Morbit.AutoDiffWrapper`.
            """)
        else
            @warn "Using $(diff_method) for gradients."
        end
    end

    if needs_hessians( model_cfg ) && isnothing(hessians)
        if isnothing(diff_method)
            error("""
            According to `model_cfg` we need hessian information.
            You can provide a list of functions with the `hessians` keyword.
            Alternatively, you can use the keyword argument `diff_method` with 
            `Morbit.FiniteDiffWrapper` or `Morbit.AutoDiffWrapper`.
            """)
        else
            @warn "Using $(diff_method) for hessians."
        end
    end

    diff_wrapper = if (needs_gradients(model_cfg) || needs_hessians(model_cfg)) && !isnothing(diff_method)
        diff_method(;
            objf = fn,
            gradients = gradients,
            jacobian = jacobian,
            hessians = hessians
        )
    else
        nothing
    end

    return VecFun(;
        n_out = n_out,
        function_handle = fn, 
        model_config = model_cfg,
        diff_wrapper
    )
end

num_vars( objf :: VecFun ) = objf.n_in
num_outputs( objf :: VecFun ) = objf.n_out

model_cfg( objf :: VecFun ) = objf.model_config

wrapped_function(objf::VecFun) = objf.function_handle

function get_objf_gradient( objf :: VecFun{<:Any, <:DiffFn, <:Any}, x :: Vec, l :: Int = 1 )
    return get_gradient( objf.diff_wrapper, x , l )
end
function get_objf_jacobian( objf :: VecFun{<:Any, <:DiffFn, <:Any}, x :: Vec)
    return get_jacobian( objf.diff_wrapper, x )
end
function get_objf_hessian( objf :: VecFun{<:Any, <:DiffFn, <:Any}, x :: Vec, l :: Int = 1 )
    return get_hessian( objf.diff_wrapper, x , l )
end
