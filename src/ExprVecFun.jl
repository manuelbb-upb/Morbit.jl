struct ExprVecFun{R,F,C} <: AbstractVecFun{C}
	function_ref :: R
	generated_function :: F
	num_outputs :: Int
	function ExprVecFun(fr::R, f::F, n) where{R,F}
		C = typeof(model_cfg( fr[] ))
		return new{R,F,C}(fr,f,n)
	end
end

registered_funcs = Dict{Symbol,Function}()

"""
	register_func(func)

Registers the function `func` for subsequent use in a function 
expression.
Note, that using function expressions is really only advisable if 
the "base function" is truly expensive and surrogate modelling remedies 
the performance penalties from parsing strings and `@eval`ing expressions.
"""
function register_func(func)
	global registered_funcs
	@eval registered_funcs[Symbol($func)] = $func
end

"""
	str2func(expr_str, vfunc; register_adjoint = true)

Parse a user provided string describing some function of `x` and 
return the resulting function.
Each occurence of "VREF(x)" in `expr_str` is replaced by a function 
evaluating `vfunc::AbstractVecFun` at `x`.
If `register_adjoint == true`, then we register a custom adjoint for 
`vfunc` that uses the `get_objf_jacobian` method.

The user may also use custom functions in `expr_str` hat have been 
registered with `register_func`.

[`register_func`](@ref)
"""
function str2func(expr_str, vfunc :: AbstractVecFun; register_adjoint = true)
	global registered_funcs

	parsed_expr = Meta.parse(expr_str)
	reg_funcs_expr = Expr[ :($k = $v) for (k,v) = registered_funcs ]

	gen_func = @eval begin 
		let $(reg_funcs_expr...), VREF = ( x -> eval_objf( $vfunc, x ) );
			if $register_adjoint
				Zygote.@adjoint VREF(x) = VREF(x), y -> get_objf_jacobian(VREF, x)'y;
				Zygote.refresh();
			end
			x -> $(parsed_expr)
		end
	end
	return VecFuncWrapper( gen_func )	# TODO can_batch ?
end

function ExprVecFun( vf :: AbstractVecFun, expr_str :: String, n_out :: Int)
	@assert n_out > 0 "Need a positive output number."

	gen_func = str2func( expr_str, vf; register_adjoint = has_gradients(vf) ) 

	return ExprVecFun(Ref(vf),gen_func,n_out)
end

num_outputs(ef :: ExprVecFun) = ef.num_outputs
wrapped_function(ef :: ExprVecFun ) = ef.generated_function

for method_name in [
	:num_vars,:model_cfg,
	]
	@eval function $(method_name)( F :: ExprVecFun, args...; kwargs... )
		return $(method_name)(F.function_ref[], args...; kwargs... )
	end
end

#=
function eval_objf( ef :: ExprVecFun, x :: Vec )
	ef.eval_counter[] += 1 
	return ef.generated_function(x)
end

function Broadcast.broadcasted( ::typeof(eval_objf), objf :: ExprVecFun, X :: VecVec )
	ef.eval_counter[] += length(X)
    return ef.generated_function.(X)
end

num_evals( ef :: ExprVecFun ) = ef.eval_counter
=#

function get_objf_gradient( ef :: ExprVecFun, x :: Vec, ℓ :: Int = 1 )
	return Zygote.gradient( ξ -> ef.generated_function(ξ)[ℓ], x )[1]
end

function get_objf_jacobian( ef :: ExprVecFun, x :: Vec )
	return Zygote.jacobian( ef.generated_function, x )[1]
end

function get_objf_hessian( ef :: ExprVecFun, x :: Vec, ℓ :: Int = 1 )
	return Zygote.gradient( ξ -> ef.generated_function(ξ)[ℓ], x )[1]
end