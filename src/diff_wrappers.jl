get_gradient( :: DiffFn, x :: Vec, l :: Int) = nothing
get_jacobian( :: DiffFn, x :: Vec ) = nothing
get_hessian( :: DiffFn, x :: Vec, l :: Int) = nothing

#=
@with_kw struct CallBackWrapper{
        G<:Union{AbstractVector{<:Function},Nothing}, 
        J<:Union{Function, Nothing},
        H<:Union{AbstractVector{<:Function},Nothing}, 
    } <: DiffFn
    objf :: Nothing = nothing
    gradients :: G = nothing
    jacobian :: J = nothing
    hessians :: H = nothing

    function CallBackWrapper{G,J,H}(objf, gradients :: G, jacobian :: J, hessians :: H) where{
        G<:Union{AbstractVector{<:Function},Nothing}, 
        J<:Union{Function, Nothing},
        H<:Union{AbstractVector{<:Function},Nothing}
    }
        @assert !(isnothing(gradients)) || !(isnothing(jacobian)) "Provide either gradients or jacobian."
        return new{G,J,H}(nothing, gradients, jacobian, hessians)
    end
end

CallBackWrapper(objf, gradients :: Function, jacobian, hessian) = CallBackWrapper(objf, [gradients,], jacobian, hessian)

# list of gradients provided
function get_gradient(gw :: CallBackWrapper{<:AbstractVector{<:Function},<:Any,<:Any}, x :: Vec, l :: Int)
    return gw.gradients[l](x)
end

# no list of gradients, but a jacobian handle
function get_gradient(gw :: CallBackWrapper{<:Nothing,<:Function,<:Any}, x :: Vec, l :: Int)
    return vec(gw.jacobian(x)[l,:])
end

# jacobian handle set
function get_jacobian(gw :: CallBackWrapper{<:Any,<:Function,<:Any}, x :: Vec )
    return gw.jacobian(x)
end

# gradient handle set
function get_jacobian(gw :: CallBackWrapper{<:AbstractVector{<:Function},<:Nothing,<:Any}, x :: Vec )
    return mat_from_row_vecs( get_gradient(gw, x, l) for l = eachindex(gw.gradients) )
end

function get_hessian( gw :: CallBackWrapper{<:Any,<:Any,<:Function}, x :: Vec, l :: Int )
    return gw.hessians[l](x)
end
=#

for (WrapperName,gradient_method,jacobian_method,hessian_method) in (
        ( :AutoDiffWrapper, Meta.parse("AD.gradient"), Meta.parse("AD.jacobian"), Meta.parse("AD.hessian") ),
        ( :FiniteDiffWrapper, Meta.parse("FD.finite_difference_gradient"), Meta.parse("FD.finite_difference_jacobian"), Meta.parse("FD.finite_difference_hessian") ),
    )

    @eval begin

        @with_kw struct $(WrapperName){ 
                O <: Union{Nothing,Function},
                G<:Union{AbstractVector{<:Function},Nothing}, 
                J<:Union{Function, Nothing},
                H<:Union{AbstractVector{<:Function},Nothing}, 
            } <: DiffFn
            objf :: O = nothing
            gradients :: G = nothing
            jacobian :: J = nothing
            hessians :: H = nothing
        end

        function $(WrapperName)(objf, gradients :: Function, jacobian, hessians)
            return $(WrapperName)(objf, [gradients,], jacobian, hessians)
        end

        # no jacobian handle set but gradients provided
        function get_jacobian( adw :: $(WrapperName){<:Any,<:AbstractVector,<:Nothing,<:Any}, x :: Vec )
            return mat_from_row_vecs( adw.gradients[l](x) for l = eachindex( adw.gradients ) )
        end

        # jacobian handle set
        function get_jacobian( adw :: $(WrapperName){<:Any,<:Any,<:Function,<:Any}, x :: Vec ) 
            return adw.jacobian( x )
        end

        # fallback to automatic differentiation
        function get_jacobian( adw :: $(WrapperName){<:Any,<:Nothing,<:Nothing,<:Any}, x :: Vec ) 
            if isnothing(adw.objf)
                error("Cannot automatically compute a derivative without an `objf`.")
            end
            return $(jacobian_method)( adw.objf, x )
        end

        # no jacobian handle set but gradients provided
        function get_gradient( adw :: $(WrapperName){<:Any,<:AbstractVector,<:Any,<:Any}, x :: Vec, l :: Int ) #where{O,G<:AbstractVector{<:Function},J<:Nothing,H}
            return adw.gradients[l](x)
        end

        # no gradients but jacobian handle set
        function get_gradient( adw :: $(WrapperName){<:Any,<:Nothing,<:Function,<:Any}, x :: Vec, l :: Int ) #where{O,G,J<:Function,H}
            return vec( adw.jacobian(x)[l,:] )
        end

        # fallback, if no gradients or jacobian are set; not optimized for multiple outputs?
        function get_gradient(adw :: $(WrapperName){<:Any,<:Nothing,<:Nothing,<:Any}, x :: Vec, l :: Int )
            if isnothing(adw.objf)
                error("Cannot automatically compute a derivative without an `objf`.")
            end
            return $(gradient_method)( ξ -> adw.objf(ξ)[l], x )
        end  

        # hessian, if functions are provided
        function get_hessian( adw :: $(WrapperName){<:Any,<:Any,<:Any,<:AbstractVector}, x :: Vec, l :: Int) #where{O,G,J,H<:AbstractVector{<:Function}}
            return adw.hessians[l](x)
        end

        # fallback to automatic differention: 
        function get_hessian( adw :: $(WrapperName){<:Any,<:Any,<:Any,<:Nothing}, x :: Vec, l :: Int)
            return $(hessian_method)( ξ -> adw.objf(ξ)[l], x )
        end
    end#eval 
end#for

function get_hessian( fdw :: FiniteDiffWrapper{<:Any,<:AbstractVector,<:Any,<:Nothing}, x :: Vec, l :: Int)
    return FD.finite_difference_jacobian( ξ -> get_gradient( fdw, ξ, l), x )
end

function get_hessian( fdw :: FiniteDiffWrapper{<:Any,<:Nothing,<:Function,<:Nothing}, x :: Vec, l :: Int)
    return FD.finite_difference_jacobian( ξ -> get_jacobian( fdw, ξ, x )[l,:] )
end