
import FiniteDiff#erences
const FD = FiniteDiff#erences

import ForwardDiff
const AD = ForwardDiff

@doc "Set fields `gradients` or/and `jacobian` of `model_config`."
function set_gradients!(
        model_config :: Union{ExactConfig, TaylorConfig},
        objf :: VectorObjectiveFunction
     )

    # During optimization, gradient information becomes necessary during the
    # descent step calculation. We need the Jacobian of all surrogates combined.
    # Additionally, gradients of the individual objectives may be used in the
    # calculation of local ideal points.
    # This function constructs gradient functions as instructed in `model_config`.
    # Ideally `n_out` distinct gradient handles are already provided.
    # Else this function first creates jacobian functions and then derives gradients
    # as the rows. There is nearly no overhead for `n_out` == 1 but this is
    # really ineffective for vector valued functions. Those should be provided
    # individually in the first place!

    # First check, if maybe function handles have already been provided
    if objf.n_out == 1 && isa( model_config.gradients, F where {F<:Function } )
       model_config.gradients = [ model_config.gradients ]
    elseif objf.n_out == 1 && isa( model_config.jacobian, F where F<:Function )
       model_config.gradients = [ x -> vec(model_config.jacobian(x)), ]
    elseif isa( model_config.gradients, Vector{F} where {F<:Function } )
        if length( model_config.gradients ) != objf.n_out
            @error "Length of gradient array does not match number of vector objective outputs."
        end
    elseif objf.n_out >= 2 && isa( model_config.jacobian, F where{F<:Function} )
        @goto build_grads
    elseif isa( model_config.gradients, Symbol )
        mode = model_config.gradients
        @goto build_jacobian
    elseif isa( model_config.jacobian, Symbol )
        mode = model_config.jacobian
        @goto build_jacobian
    else
        @error "No derivative method (:fdm or :autodiff) or gradient function handle(s) provided."
    end

    return nothing

    @label build_jacobian
    # Build jacobians using ForwardDiff (:autodiff) or FiniteDiff (:fdm)
    if mode == :fdm
        model_config.jacobian = function (x :: Vector{R} where{R<:Real})
            # taking difference quotients of `objf` (instead of
            # `objf.function_handle`) increases `objf.n_evals`
            FD.finite_difference_jacobian( objf, x)
        end
        @goto build_grads
    elseif mode == :autodiff
        if objf.n_out > 1
            model_config.jacobian = function (x :: Vector{R} where{R<:Real})
                AD.jacobian( objf.function_handle, x )
            end
            @goto build_grads
        else
            grad_fun = function (x :: Vector{R} where{R<:Real})
                return AD.gradient( objf.function_handle , x )
            end
            model_config.gradients = [ grad_fun, ];
        end
    else
        @error "No derivative method (:fdm or :autodiff) or gradient function handle(s) provided."
    end
    return nothing

    # Set gradient handles based on the jacobian
    @label build_grads
    @warn "Building gradients from jacobian. This could lead to many unnecessary evaluations!"
    gradients = [];
    for i = 1 : objf.n_out
        push!( gradients, x -> vec(model_config.jacobian(x)[i, :]) )
    end
    model_config.gradients = gradients;
    return nothing
end

# Get derivative information
function get_jacobian( cfg :: Union{ExactConfig, TaylorConfig},
        x :: Vector{R} where{R<:Real} )
    if !isnothing(cfg.jacobian)
        return cfg.jacobian(x)
    else
        return transpose( hcat( (g(x) for g ∈ cfg.gradients )... ) )
    end
end

function set_hessians!(model_config :: TaylorConfig, objf :: VectorObjectiveFunction )
    if objf.n_out == 1 && isa(model_config.hessians, Function)
        model_config.hessians = [ model_config.hessians, ]
        return
    elseif objf.n_out > 1 && isa(model_config.hessians, Vector{F} where F<:Function )
        if length( model_config.hessians ) == objf.n_out
            return
        else
            @error "Nmuber of hessian handles does not match n_out."
        end
    else
        new_hessians = []
        for i = 1 : objf.n_out
            if model_config.hessians == :fdm
                hessian_handle = function (x :: Vector{R} where{R<:Real})
                    # taking difference quotients of `objf` (instead of
                    # `objf.function_handle`) increases `objf.n_evals`
                    FD.finite_difference_hessian( ξ -> objf(ξ)[i], x)
                end
            elseif model_config.hessians == :autodiff
                hessian_handle = function (x :: Vector{R} where{R<:Real})
                    AD.hessian( ξ -> objf.function_handle(ξ)[i], x )
                end
            else
                @error "No Hessian method (:fdm or :autodiff) or function handle for output $i provided."
            end
            push!( new_hessians, hessian_handle );
        end
        model_config.hessians = new_hessians;
        return
    end
end
