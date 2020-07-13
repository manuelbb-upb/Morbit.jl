
scale( ::Val{true}, problem :: MixedMOP, x :: Vector{Float64} ) = ( x .- problem.lb ) ./ ( problem.ub .- problem.lb );
scale( ::Val{false}, problem :: MixedMOP, x :: Vector{Float64} ) = x
scale( problem :: MixedMOP, x :: Vector{Float64} ) = scale( Val( problem.is_constrained ), problem, x)

unscale( ::Val{true}, problem :: MixedMOP, x :: Vector{Float64} ) = problem.lb .+ ( x .* ( problem.ub .- problem.lb ) );
unscale( ::Val{false}, problem :: MixedMOP, x :: Vector{Float64} ) = x
unscale( problem :: MixedMOP, x :: Vector{Float64} ) = unscale( Val( problem.is_constrained ), problem, x)

function eval_expensive_objectives( problem :: MixedMOP, x :: Vector{Float64} )
    X = unscale( problem, x )
    return vcat( [ f(X) for f ∈ problem.vector_of_expensive_funcs ]... );
end

function eval_cheap_objectives( problem :: MixedMOP, x :: Vector{Float64} )
    X = unscale( problem, x );
    return vcat( [ f(X) for f ∈ problem.vector_of_cheap_funcs  ]... );
end

function eval_all_objectives( problem :: MixedMOP, x :: Vector{Float64} )
    X = unscale( problem, x );
    return vcat( [ f(X) for f ∈ [problem.vector_of_expensive_funcs; problem.vector_of_cheap_funcs] ]... );
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

function eval_surrogates( problem :: MixedMOP, m :: Union{RBFModel, NamedTuple}, x :: Vector{Float64})
    [ output( m, x ); eval_cheap_objectives( problem, x ) ];
end

@doc """
    eval_grad( problem :: MixedMOP, m :: Union{RBFModel, NamedTuple}, x :: Vector{Float64}, output_index :: Int64 = 1)

Calculate gradient for (internal) objective with index `output_index` at (scaled) site `x`.
For expensive objectives the gradient is calculated using the surrogate `m`.
Cheap gradients use function handles stored in `problem`.
"""
function eval_grad( problem :: MixedMOP, m :: Union{RBFModel, NamedTuple}, x :: Vector{Float64}, output_index :: Int64 = 1)
    if output_index <= problem.n_exp
        return grad( m, output_index, x)
    else
        return problem.vector_of_gradient_funcs[ output_index - problem.n_exp ]( unscale( problem, x) )
    end
end

function eval_jacobian( problem :: MixedMOP, m :: Union{RBFModel, NamedTuple} , x :: Vector{Float64} )
    expensive_jacobian = jac( m, x );

    X = unscale( problem, x);
    cheap_jacobian = ones( problem.n_cheap, length(x) );
    for ℓ = 1 : problem.n_cheap
        cheap_jacobian[ℓ, :] = problem.vector_of_gradient_funcs[ ℓ ]( X )
    end
    return vcat( expensive_jacobian, cheap_jacobian )
end

@doc "Return a function that is suited for NLopt optimization."
function get_optim_handle( problem :: MixedMOP, m :: Union{RBFModel, NamedTuple}, output_index :: Int64 = 1 )
    if output_index <= problem.n_exp
        return function (x :: Vector{T}, g :: Vector{F} ) where{T,F <: Real}
            if !isempty(g)
                g[:] = grad( m, output_index, x )
            end
            output( m, output_index, x )
        end
    else
        output_index -= problem.n_exp
        return function (x :: Vector{T}, g :: Vector{F} ) where{T,F <: Real}
            X = unscale(problem,x)
            if !isempty(g)
                g[:] = problem.vector_of_gradient_funcs[output_index]( X )
            end
            problem.vector_of_cheap_funcs[output_index]( X )
        end
    end
end
