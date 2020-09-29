
scale( ::Val{true}, problem :: MixedMOP, x :: Vector{Float64} ) = ( x .- problem.lb ) ./ ( problem.ub .- problem.lb );
scale( ::Val{false}, problem :: MixedMOP, x :: Vector{Float64} ) = x
scale( problem :: MixedMOP, x :: Vector{Float64} ) = scale( Val( problem.is_constrained ), problem, x)

unscale( ::Val{true}, problem :: MixedMOP, x :: Vector{Float64} ) = problem.lb .+ ( x .* ( problem.ub .- problem.lb ) );
unscale( ::Val{false}, problem :: MixedMOP, x :: Vector{Float64} ) = x
unscale( problem :: MixedMOP, x :: Vector{Float64} ) = unscale( Val( problem.is_constrained ), problem, x)

function eval_expensive_objectives( problem :: MixedMOP, x :: Vector{Float64} )
    X = unscale( problem, x )
    return Vector{Float64}(vcat( [ f(X) for f ∈ problem.vector_of_expensive_funcs ]... ));
end
eval_expensive_objectives( config_struct :: AlgoConfig, x :: Vector{Float64} ) = eval_expensive_objectives( config_struct.problem, x)

function eval_expensive_objectives( problem :: MixedMOP, X :: Vector{Vector{Float64}} )
    mat = hcat( [ f.(X) for f ∈ problem.vector_of_expensive_funcs ]... )
    return Vector{Float64}.(collect( mat[i,:] for i = 1 : size(mat,1) ))
end
eval_expensive_objectives( config_struct :: AlgoConfig, X :: Vector{Vector{Float64}} ) = eval_expensive_objectives( config_struct.problem, X )

function eval_cheap_objectives( problem :: MixedMOP, x :: Vector{Float64} )
    X = unscale( problem, x );
    return Vector{Float64}(vcat( [ f(X) for f ∈ problem.vector_of_cheap_funcs  ]... ));
end
eval_cheap_objectives(config_struct :: AlgoConfig, x :: Vector{Float64} ) = eval_cheap_objectives( config_struct.problem, x)

function eval_cheap_objectives( problem :: MixedMOP, X :: Vector{Vector{Float64}} )
    mat = hcat( [ f.(X) for f ∈ problem.vector_of_cheap_funcs ]... )
    return Vector{Float64}.(collect( mat[i,:] for i = 1 : size(mat,1) ))
end
eval_cheap_objectives( config_struct :: AlgoConfig,  X :: Vector{Vector{Float64}}) = eval_cheap_objectives( config_struct.problem, X)

function eval_all_objectives( problem :: MixedMOP, x :: Vector{Float64} )
    X = unscale( problem, x );
    return Vector{Float64}(vcat( [ f(X) for f ∈ [problem.vector_of_expensive_funcs; problem.vector_of_cheap_funcs] ]... ));
end
eval_all_objectives( config_struct :: AlgoConfig, x :: Vector{Float64} ) = eval_all_objectives( config_struct.problem, x)

function eval_all_objectives( problem :: MixedMOP, X :: Vector{Vector{Float64}} )
    mat = hcat( [ f.(X) for f ∈ [problem.vector_of_expensive_funcs; problem.vector_of_cheap_funcs] ]... )
    return Vector{Float64}.(collect( mat[i,:] for i = 1 : size(mat,1) ))
end
eval_all_objectives( config_struct :: AlgoConfig,  X :: Vector{Vector{Float64}}) =  eval_all_objectives( config_struct.problem, X)

@doc "Return the first `problem.n_exp` components from objective vector `y`."
function expensive_components( y :: Vector{T} where{T<:Real} , config_struct :: AlgoConfig )
    return y[ 1 : config_struct.n_exp ]
end

@doc "Return the last `problem.n_cheap` components from objective vector `y`."
function cheap_components( y :: Vector{T} where{T<:Real} , config_struct :: AlgoConfig )
    return y[ config_struct.n_exp + 1 : end ]
end

# custom broadcasts to exploit parallelized objectives

function broadcasted( f::Union{typeof(eval_expensive_objectives),typeof(eval_cheap_objectives), typeof(eval_all_objectives) }, problem :: MixedMOP, x, args... )
    X = unscale.( problem, x )
    f( problem, X);
end

@doc "Sort image vector `y` to correspond to internal objective sorting."
function apply_internal_sorting(problem :: MixedMOP, y :: Vector{Float64} )
    y[ problem.internal_sorting ]
end

function reverse_internal_sorting(problem :: MixedMOP, y :: Vector{Float64} )
    index_permutation = problem.internal_sorting
    inverse_permutation = sortperm(index_permutation);
    y[ inverse_permutation ]
end

function reverse_internal_sorting(problem :: MixedMOP, Y :: Vector{Vector{Float64}} )
    index_permutation = problem.internal_sorting
    inverse_permutation = sortperm(index_permutation);
    [ y[ inverse_permutation ] for y in Y ]
end

function eval_surrogates( config_struct::AlgoConfig, m :: Union{RBFModel, NamedTuple}, x :: Vector{Float64})
    exp_retval = output( m, x );
    cheap_retval = eval_cheap_objectives( config_struct, x );
    if config_struct.iter_data.update_extrema
        @unpack iter_data = config_struct;
        cheap_retval = scale( cheap_retval, iter_data, config_struct.n_exp + 1 : config_struct.n_exp + config_struct.n_cheap )
    end
    return Float64.(vcat( exp_retval, cheap_retval ))
end

@doc """
    eval_grad( problem :: MixedMOP, m :: Union{RBFModel, NamedTuple}, x :: Vector{Float64}, output_index :: Int64 = 1)

Calculate gradient for (internal) objective with index `output_index` at (scaled) site `x`.
For expensive objectives the gradient is calculated using the surrogate `m`.
Cheap gradients use function handles stored in `problem`.
"""
function eval_grad( config_struct :: AlgoConfig, m :: Union{RBFModel, NamedTuple}, x :: Vector{Float64}, output_index :: Int64 = 1)
    @unpack problem = config_struct;
    if output_index <= config_struct.n_exp
        return grad( m, output_index, x)
    else
        gradval = scale_grad( problem.vector_of_gradient_funcs[ output_index - problem.n_exp ]( unscale( problem, x) ), config_struct.iter_data, output_index )
    end
end

function eval_jacobian( config_struct :: AlgoConfig, m :: Union{RBFModel, NamedTuple} , x :: Vector{Float64} )
    @unpack problem, iter_data = config_struct;
    expensive_jacobian = jac( m, x );

    X = unscale( problem, x);
    cheap_jacobian = ones( problem.n_cheap, length(x) );
    for ℓ = 1 : problem.n_cheap
        true_grad = problem.vector_of_gradient_funcs[ℓ](X);
        cheap_jacobian[ℓ, :] = scale_grad( true_grad, iter_data, ℓ + problem.n_exp )
    end
    return vcat( expensive_jacobian, cheap_jacobian )
end

@doc "Return a function that is suited for NLopt optimization."
function get_optim_handle( config_struct :: AlgoConfig, m :: Union{RBFModel, NamedTuple}, output_index :: Int64 = 1 )
    @unpack iter_data, problem = config_struct;
    if output_index <= config_struct.n_exp
        return function (x :: Vector{T}, g :: Vector{F} ) where{T,F <: Real}
            if !isempty(g)
                g[:] = grad( m, output_index, x )
            end
            output( m, output_index, x )
        end
    else
        ℓ = output_index;
        output_index -= config_struct.n_exp

        this_output = function (x)
            retval = problem.vector_of_cheap_funcs[output_index]( x )
            retval = scale(retval, iter_data, [ℓ])
            return retval
        end
        this_grad = function (x)
            retval = scale_grad( problem.vector_of_gradient_funcs[output_index]( x ), iter_data, ℓ );
            return retval
        end


        return function (x :: Vector{T}, g :: Vector{F} ) where{T,F <: Real}
            X = unscale(problem,x)
            if !isempty(g)
                g[:] = this_grad(X)
            end
            this_output(X)
        end
    end
end
