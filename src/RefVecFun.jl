struct RefVecFun{R,C} <: AbstractVecFun{C}
	function_ref :: R 

	function RefVecFun( fr :: R ) where R
		C = typeof( model_cfg(fr[]) )
		return new{R,C}(fr)
	end
	#TODO use own eval counter like in expr vec fun?
end

RefVecFun( F :: AbstractVecFun ) = RefVecFun( Ref(F) )

for method_name in [
	:num_vars, :num_outputs,:model_cfg,:wrapped_function,
	:get_objf_gradient, :get_objf_jacobian, :get_objf_hessian
	]
	@eval function $(method_name)( F :: RefVecFun, args...; kwargs... )
		return $(method_name)(F.function_ref[], args...; kwargs... )
	end
end