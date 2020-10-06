module RBF
export RBFModel, train!, output, grad, is_valid, jac, min_num_sites

using Parameters: @with_kw, @pack!, @unpack
using LinearAlgebra: norm, Hermitian, lu, qr, cholesky

# If kernel `:k` is conditionally positive definite (cpd) of order `d` then
# Σ_k cₖ k(‖ x - xₖ‖) + p(x)
# allows for a unique interpolation if p(x) is of degree *at least* `d-1`.
const cpd_order( ::Val{:exp} ) = 0 :: Int64;
const cpd_order( ::Val{:multiquadric} ) = 2 :: Int64;
const cpd_order( ::Val{:cubic} ) = 2 :: Int64;
const cpd_order( ::Val{:thin_plate_spline} ) = typemax(Int64) :: Int64;

@with_kw mutable struct RBFModel
    training_sites :: Array{Array{Float64, 1},1} = [];
    training_values :: Array{Array{Float64, 1}, 1} = [];
    n_in :: Int64 = length(training_sites) > 0 ? length(training_sites[1]) : -1;
    kernel :: Symbol = :multiquadric;
    shape_parameter :: Union{Float64, Vector{Float64}} = 1.0;
    fully_linear :: Bool = false;
    polynomial_degree :: Int64 = 1;

    rbf_coefficients :: Array{Float64,2} = Matrix{Float64}(undef,0,0);
    poly_coefficients :: Array{Float64,2} = Matrix{Float64}(undef,0,0);

    function_handle :: Union{Nothing, Function} = nothing;      # TODO deprecate

    #=
    if polynomial_degree < cpd_order( Val(kernel) ) - 1
        @warn "Polynomial degree is to small (should at least be $(cpu_order(kernel) - 1))."
    end
    =#
    @assert polynomial_degree <= 1 "For now only polynomials with degree -1, 0, or 1 are allowed."
end

@doc "Return number of input variables to RBFModel `m`. If `m.n_in` is not set, infer from training sites if possible."
function n_in(m::RBFModel)
    if m.n_in < 0
        if length(m.training_sites) > 0
            m.n_in = length(m.training_sites[0])
        else
            @error "Could not determine number of input variables for RBFModel and infer minimum number of interpolation sites."
        end
    end
    return m.n_in
end

# minimal number of `training_sites` a RBFModel with polynomial tail must have
# (equals number of different sites so that Π has full column rank)
function min_num_sites( m :: RBFModel )
    pd = m.polynomial_degree;
    if pd == -1
        return 0
    elseif pd == 0
        return 1
    else
        return n_in(m) + 1
    end
end

@doc "Return `true` if RBFModel `m` conforms to the requirements by [WILD]."
is_valid( m :: RBFModel ) = m.polynomial_degree >= cpd_order( Val(m.kernel) ) - 1 &&
                            length(m.training_sites) >= min_num_sites(m);

Broadcast.broadcastable(m::RBFModel) = Ref(m);

function (m::RBFModel)(eval_site::Array{Float64, 1})
    m.function_handle(eval_site)  # return an array of arrays for ease of use in other methods
end

@doc "Modify first model to equal second."
function as_second!(destination :: RBFModel, source :: RBFModel )
    @unpack_RBFModel source;
    @pack_RBFModel! destination;
    return nothing
end

# === Evaluate RBF part of the Model ===
kernel( ::Val{:exp}, r, s = 1.0 ) = exp(-(r*s)^2);
kernel( ::Val{:multiquadric} , r, s = 1.0 ) = - sqrt( 1 + (s*r)^2);
kernel( ::Val{:cubic} , r, s = 1.0 ) = (r*s)^3;     # TODO better change this to r^s, s ∈ (2,4)
kernel( ::Val{:thin_plate_spline}, r, s = 1.0) = r == 0 ? 0.0 : r^2 * log(r);

@doc "Return n_sites-array with difference vectors (x_1 - c_{1,m}, …, x_n - c_{n,m}) of length n."
function x_minus_sites( m:: RBFModel, x :: Vector{T} where{T<:Real})
    [ x .- site for site ∈ m.training_sites ]
end

@doc "Return vector of all distance values between x and each center of m"
function center_distances( m :: RBFModel, x :: Vector{T} where{T<:Real} )
    r_vector = norm.( x_minus_sites( m, x), 2 )
end

@doc "Evaluate all ``n_c`` basis functions of m::RBFModel at second argument x and return ``n_c``-Array"
function φ( m::RBFModel, x :: Vector{T} where{T<:Real} )
    kernel.( Val(m.kernel), center_distances(m, x), m.shape_parameter )
end

@doc "Symmetric matrix of every center evaluated in all basis functions."
function get_Φ( m::RBFModel )
    hcat( φ.(m, m.training_sites)... )'
end

@doc "Return ℓ-th output of the RBF Part of the model at site x."
function rbf_output( m::RBFModel, ℓ :: Int64, x :: Vector{T} where{T<:Real})
    φ(m, x)'m.rbf_coefficients[:, ℓ]
end

@doc "Return k-vector of *all* RBF model outputs at site x."
function rbf_output( m::RBFModel, x :: Vector{T} where{T<:Real} )
    vec(φ(m, x)'m.rbf_coefficients) # wrapped in vector for addition with polynomial part
end

# partial derivatives, all missing the factor 2 * ( x_i - c_i ) where c is center site of a single basis function
∂kernel( ::Val{:exp}, r, s = 1.0 ) = - s^2 * kernel( Val(:exp), r, s )
∂kernel( ::Val{:multiquadric}, r, s = 1.0 ) = s^2 / (2 * kernel( Val(:multiquadric), r, s ) );
∂kernel( ::Val{:cubic}, r, s = 1.0 ) = s^3 * ( r + r/2 ) ;
∂kernel( ::Val{:thin_plate_spline}, r, s = 1.0 ) = r == 0 ? 0.0 : log(r) + 1/2;

@doc "Return an n_vars x n_sites matrix where each column is an n-vector of entries 2 coeff(m,ℓ) *(x_1 - c_{1,m}), …, 2*(x_n - c_{n,m}). "
function grad_prefix( m :: RBFModel, ℓ :: Int64, x :: Vector{T} where{T<:Real} )
    differences = x_minus_sites( m, x )
    2 .* hcat( [  m.rbf_coefficients[ i, ℓ ] * differences[i] for i = eachindex(differences) ] ... )
end

@doc "Return an k-array of n_vars x n_sites matrices (1 for each RBF model output) where each column is an n-vector of entries 2 coeff(m,ℓ) *(x_1 - c_{1,m}), …, 2*(x_n - c_{n,m}). "
function grad_prefix( m :: RBFModel, x :: Vector{T} where{T<:Real} )
    differences = x_minus_sites( m, x )
    grad_prefix_matrices = [];
    for ℓ = 1 : length(m.training_values[1])
        grad_pref_ℓ = 2 .* hcat( [  m.rbf_coefficients[ i, ℓ ] * differences[i] for i = eachindex(differences) ] ... )
        push!(grad_prefix_matrices, grad_pref_ℓ)
    end
    return grad_prefix_matrices, differences
end

@doc "Compute gradient term of ℓ-th RBF (scalar) model output and return n-vector."
function rbf_grad( m::RBFModel, ℓ :: Int64, x :: Vector{T} where{T<:Real} )
    (grad_prefix(m, ℓ, x) * ∂kernel.( Val(m.kernel), center_distances( m, x ), m.shape_parameter ))[:]      # n × n_c * n_c × 1
end

@doc "Compute the Jacobian matrix of the RBF part of the model"
function rbf_jacobian( m::RBFModel, x :: Vector{T} where{T<:Real} )
    grad_prefix_matrices, differences = grad_prefix(m, x)
    center_dists = norm.( differences, 2 );
    ∂kernel_eval = ∂kernel.( Val(m.kernel), center_dists, m.shape_parameter )
    gradient_array = [];
    for ℓ = 1 : length(m.training_values[1])
        grad_prefix_ℓ = grad_prefix_matrices[ℓ]
        grad_ℓ = vec(grad_prefix_ℓ * ∂kernel_eval)
        push!(gradient_array, grad_ℓ);
    end
    hcat( gradient_array... )'  # each gradient is a column vector -> stack horizontally and then transpose
end
rbf_jacobian( m::RBFModel, x :: T where{T<:Real} ) = rbf_jacobian( m, [x])

# === Evaluate polynomial part of the model ===
poly(m::RBFModel, ℓ, x, ::Val{-1} ) = 0.0                       # degree -1 ⇒ No polynomial part (ℓ-th output)
poly(m::RBFModel, x, ::Val{-1} ) = 0.0                          # degree -1 ⇒ No polynomial part (all outputs)
poly(m::RBFModel, ℓ, x, ::Val{0} ) = m.poly_coefficients[ℓ]     # degree 0 ⇒ constant (ℓ-th output)
poly(m::RBFModel, x, ::Val{0} ) = m.poly_coefficients[:]        # degree 0 ⇒ constant (all outputs)
poly(m::RBFModel, ℓ, x, ::Val{1} ) = [x;1.0]'m.poly_coefficients[:, ℓ] # affin linear tail (ℓ-th output)
poly(m::RBFModel, x, ::Val{1} ) = vec([x;1.0]'m.poly_coefficients)    # affin linear tail (all outputs)

@doc "Evaluate polynomial tail for output ℓ."
function poly_output( m::RBFModel, ℓ :: Int64, x :: Vector{T} where{T<:Real} )
    poly( m, ℓ, x, Val(m.polynomial_degree) )
end

@doc "k-Vector of evaluations of polynomial tail for all outputs."
function poly_output( m::RBFModel, x :: Vector{T} where{T<:Real} )
    poly( m, x, Val(m.polynomial_degree) )
end

∇poly(m::RBFModel, ℓ :: Int64 , ::Union{Val{-1}, Val{0} } ) = zeros( length( m.training_sites[1] ) )
∇poly(m::RBFModel, ℓ :: Int64 , ::Val{1} ) = m.poly_coefficients[1 : end - 1, ℓ];    # return all coefficients safe c_{n+1}
@doc "Gradient vector for polynomial tail of ℓ-th output."
function poly_grad( m::RBFModel, ℓ :: Int64 , x :: Vector{T} where{T<:Real} )
    ∇poly( m, ℓ, Val(m.polynomial_degree) )
end

function poly_jacobian( m::RBFModel, x :: Vector{T} where{T<:Real} )
    hcat( [poly_grad(m, ℓ, x) for ℓ = 1 : length(m.training_values[1] ) ]... )'
end

# === Combined model output ===
@doc "Evaluate ℓ-th (scalar) model output at vector x."
function output( m::RBFModel, ℓ :: Int64, x :: Vector{T} where{T<:Real} )
    rbf_output( m, ℓ, x ) + poly_output( m, ℓ, x )
end
output( m::RBFModel, ℓ :: Int64, x :: Real ) = output(m, ℓ, [x])

@doc "Evaluate all (scalar) model outputs at vector x and return k-vector of results."
function output( m::RBFModel, x :: Vector{T} where{T<:Real} )
    vec(rbf_output( m, x ) .+ poly_output( m, x ))
end
output( m::RBFModel, x :: Real ) = output(m, [x])

function grad( m::RBFModel, ℓ :: Int64, x :: Vector{T} where{T<:Real} )
    rbf_grad( m, ℓ, x ) + poly_grad( m, ℓ, x)
end
grad(m::RBFModel, ℓ::Int64, x::Real) = grad(m, ℓ, [x])   # if n_vars == 1 and RBFModel is used outside of Optimization

function jac( m::RBFModel, x :: Vector{T} where{T<:Real} )
    rbf_jacobian( m, x ) + poly_jacobian( m , x )
end
jac(m::RBFModel, x::Real) = jac(m,[x])         # if n_vars == 1 and RBFModel is used outside of Optimization

# === Utiliy functions for solving the normal equations
get_Π( m :: RBFModel, ::Val{-1} ) =  Matrix{Float64}( undef, 0, length(m.training_sites) );
get_Π( m :: RBFModel, ::Val{0} ) = ones(1, length(m.training_sites));
get_Π( m :: RBFModel, ::Val{1} ) = [ hcat( m.training_sites...); ones(1,length(m.training_sites)) ];

@doc "Return polynomial base matrix. Ones in last row."
function get_Π(m::RBFModel)
    get_Π( m, Val( m.polynomial_degree ) )
end

@doc "Return column vector to augment the polynomial base matrix `Π` if `y` were added."
function Π_col( m :: RBFModel , y :: Vector{R} where{R<:Real} )
    pd = m.polynomial_degree
    if pd == -1
        return Matrix{Float64}(undef, 0, 1)
    elseif pd == 0
        return 1.0
    elseif pd == 1
        return [ y ; 1.0 ]
    end
end

# see [WILD 2008]
function null_space_coefficients( Q :: T1, R :: T2, Z :: T3, L :: T4, f :: T5, Φ :: T6 ) where{
        T1, T2, T3, T4, T5, T6 <: AbstractArray{Float64}
    }
    rhs = Z'f;
    w = L \ rhs;
    ω = L' \ w;
    λ = Z * ω;  # RBF coefficients

    rhs_poly = Q'*( f.- Φ * λ)
    v = R \ rhs_poly;
    return λ, v
end

@doc "Solve RBF linear equation system using Cholesky based null space method as in [WILD 2008]."
function solve_rbf_problem( Π :: T1, Φ :: T2, f :: T3, :: Val{true} ) where{ T1, T2, T3 <: AbstractArray{Float64}}
    Q, R = qr( Π' );
    R = [
        R;
        zeros( size(Q,1) - size(R,1), size(R,2) )
    ]
    Z = Q[:, size(Π,1) + 1 : end ]

    ZΦZ = Hermitian( Z'Φ*Z );
    L = cholesky( ZΦZ ).L     # should also be empty at this point

    λ, v = null_space_coefficients( Q, R, Z, L, f, Φ)
    return vcat( λ, v )
end

@doc "Solve the RBF linear equation system using LU or QR factorization."
function solve_rbf_problem( Π :: T1, Φ :: T2, f :: T3, :: Val{false} )  where{ T1, T2, T3 <: AbstractArray{Float64}}
    @info("LU or QR")
    Φ_augmented = [ [Φ Π']; [Π zeros(size(Π,1),size(Π,1) )] ];
    f_augmented = [ f;
                    zeros( size(Π,1), size(f,2) ) ];
    local Φ_fact;   # factorized matrix (for min norm solution if Φ is singular)
    try
        Φ_fact = lu( Φ_augmented );
    catch SingularException
        @warn("\tWARNING RBF Matrix singular. Using QR decomposition.")
        Φ_fact = qr(Φ_augmented, Val(true))   # use QR to still obtain a (minimal norm) solution if Φ is near singular
    end
    coefficients = Φ_fact \ f_augmented;   # obtain model coefficients as min norm solution to LGS

    return coefficients
end

# === Training functions ===
function set_coefficients!( m :: RBFModel, coefficients :: AbstractArray{Float64} )
    pd = m.polynomial_degree
    n_vars = n_in(m);
    n_out = length(m.training_values[1]);
    n_sites = length( m.training_sites );
    if pd == -1
        m.rbf_coefficients = reshape(coefficients, (n_sites, n_out));
        m.poly_coefficients =  Matrix{Float64}( undef, 0, n_out );
    elseif pd == 0
        m.rbf_coefficients = reshape(coefficients[1:end-1, :], (n_sites, n_out));
        m.poly_coefficients = reshape(coefficients[ end, : ], (1, n_out));
    elseif pd == 1
        m.rbf_coefficients = reshape(coefficients[1 : n_sites, :], n_sites, n_out);
        m.poly_coefficients = reshape(coefficients[ n_sites + 1 : end, : ], (n_vars + 1, n_out) );
    end # NOTE everything wrapped in reshape for the case that n_out == 1, n_vars == 1

end

@doc "Train (and fully instanciate) a RBFModel instance to best fit its training data."
function train!( m::RBFModel )
    Φ = get_Φ( m );
    Π = get_Π( m );
    RHS = hcat( m.training_values... )';

    coefficients = solve_rbf_problem( Π, Φ, RHS, Val( is_valid(m) ) )
    set_coefficients!(m, coefficients)
    m.function_handle = x -> output(m, x);
    return m
end

# same as train! but using a null space method with data available from 'add_points!' method
# USE ONLY FOR MODELS WITH is_valid(m) == true !!!
function train!(m::RBFModel, Q :: T1 , R :: T2 , Z :: T3, L :: T4, Φ :: T5 ) where{ T1, T2, T3, T4, T5 <: AbstractArray{Float64}}

    RHS = hcat( m.training_values... )';
    λ, v = null_space_coefficients( Q, R, Z, L, RHS, Φ)
    set_coefficients!(m, vcat(λ,v))
    m.function_handle = x -> output(m, x);
    return m
end

end#module
