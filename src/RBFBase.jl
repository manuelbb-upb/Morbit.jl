module RBF
export RBFModel, train!, output, grad, is_valid, jac, min_num_sites

using Parameters: @with_kw, @pack!, @unpack
using LinearAlgebra: norm, Hermitian, lu, qr, cholesky, eigen, I

# If kernel `:k` is conditionally positive definite (cpd) of order `d` then
# Σ_k cₖ k(‖ x - xₖ‖) + p(x)
# allows for a unique interpolation if p(x) is of degree *at least* `d-1`.
cpd_order( ::Val{:exp} ) = 0 :: Int64;
cpd_order( ::Val{:multiquadric} ) = 2 :: Int64;
cpd_order( ::Val{:cubic} ) = 2 :: Int64;
cpd_order( ::Val{:thin_plate_spline} ) = typemax(Int64) :: Int64;

const RVec = Vector{R} where R<:Real;
const RVecArr = Vector{<:RVec};

@with_kw mutable struct RBFModel
    training_sites :: RVecArr = RVec[];
    training_values :: RVecArr = RVec[];
    n_in :: Int64 = length(training_sites) > 0 ? length(training_sites[1]) : -1;
    kernel :: Symbol = :multiquadric;
    shape_parameter :: Union{Real, RVec} = 1;
    fully_linear :: Bool = false;
    polynomial_degree :: Int64 = 1; # -1 means *no polynomial*, not a rational function

    rbf_coefficients :: Array{R,2} where R<:Real = Matrix{Float16}(undef,0,0);
    poly_coefficients :: Array{R,2} where R<:Real = Matrix{Float16}(undef,0,0);

    #function_handle :: Union{Nothing, Function} = nothing;      # TODO deprecate

    @assert polynomial_degree <= 1 "For now only polynomials with degree -1, 0, or 1 are allowed."
    @assert isa(shape_parameter, Real) || length(shape_parameter) == length(training_sites)
end

@doc "Return number of input variables to RBFModel `m`. If `m.n_in` is not set, infer from training sites if possible."
function n_in(m::RBFModel)
    if m.n_in < 0
        if length(m.training_sites) > 0
            m.n_in = length(m.training_sites[0])
        else
            error( "Could not determine number of input variables for RBFModel and 
            infer minimum number of interpolation sites.")
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

num_outputs( m::RBFModel ) = length( m.training_values[end] );

@doc "Return `true` if RBFModel `m` conforms to the requirements by [WILD]."
is_valid( m :: RBFModel ) = m.polynomial_degree >= cpd_order( Val(m.kernel) ) - 1 &&
                            length(m.training_sites) >= min_num_sites(m);

Broadcast.broadcastable(m::RBFModel) = Ref(m);

function (m::RBFModel)(eval_site:: RVec)
    output(m, eval_site)  # return an array of arrays for ease of use in other methods
end

# === Evaluate RBF part of the Model ===
kernel( ::Val{:exp}, r::Real, s::Real = 1 ) = exp(-(r*s)^2);
kernel( ::Val{:multiquadric}, r::Real, s::Real = 1 ) = - sqrt( 1 + (s*r)^2);
kernel( ::Val{:cubic}, r::Real, s::Real = 1 ) = (r*s)^3;     # TODO better change this to r^s, s ∈ (2,4)
function kernel( ::Val{:thin_plate_spline}, r::Real, s::Real = 1)
    r == 0 ? 0 : (-1)^(s+1) * r^(2*s) * log(r)
end

@doc "Return n_sites-array with difference vectors (x_1 - c_{1,m}, …, x_n - c_{n,m}) of length n."
function x_minus_sites( m:: RBFModel, x :: RVec)
    [ x .- site for site ∈ m.training_sites ]
end

@doc "Return vector of all distance values between x and each center of m"
function center_distances( m :: RBFModel, x :: RVec )
    r_vector = norm.( x_minus_sites( m, x), 2 )
end

@doc "Evaluate all ``n_c`` basis functions of m::RBFModel at second argument x
and return ``n_c``-Array"
function φ( m::RBFModel, x :: RVec )
    kernel.( Val(m.kernel), center_distances(m, x), m.shape_parameter )
end

@doc "Symmetric matrix of every center evaluated in all basis functions."
function get_Φ( m::RBFModel )
    hcat( φ.(m, m.training_sites)... )'
end

@doc "Return ℓ-th output of the RBF Part of the model at site x."
function rbf_output( m::RBFModel, ℓ :: Int64, x :: RVec)
    φ(m, x)'m.rbf_coefficients[:, ℓ]
end

@doc "Return k-vector of *all* RBF model outputs at site x."
function rbf_output( m::RBFModel, x :: RVec )
    vec(φ(m, x)'m.rbf_coefficients) # wrapped in vector for addition with polynomial part
end

# d/dr kernel
∂kernel( ::Val{:exp}, r::Real, s::Real = 1 ) = - 2 * r * s^2 * kernel( Val(:exp), r, s )
∂kernel( ::Val{:multiquadric}, r::Real, s::Real = 1 ) = r * s^2 / kernel( Val(:multiquadric), r, s ) ;
∂kernel( ::Val{:cubic}, r::Real, s::Real = 1 ) = 3 * s^3 * r^2 ;
∂kernel( ::Val{:thin_plate_spline}, r::Real, s::Real = 1 ) = r == 0 ? 0 :
    (-1)^(s+1) * r^(2*s-1) * (
        1 +
        2 * s * log(r)
    );
#=
@doc "Return an n_vars x n_sites matrix where column j is an n-vector of entries 2 coeff(j,ℓ) *(x_1 - c_{1,j}), …, 2*(x_n - c_{n,j}). "
function grad_prefix( m :: RBFModel, ℓ :: Int64, x :: RVec )
    differences = x_minus_sites( m, x )
    2 .* hcat( [  m.rbf_coefficients[ i, ℓ ] * differences[i] for i = eachindex(differences) ] ... )
end

@doc "Return an k-array of n_vars x n_sites matrices (1 for each RBF model output) where each column is an n-vector of entries 2 coeff(m,ℓ) *(x_1 - c_{1,m}), …, 2*(x_n - c_{n,m}). "
function grad_prefix( m :: RBFModel, x :: RVec )
    differences = x_minus_sites( m, x )
    grad_prefix_matrices = [];
    for ℓ = 1 : length(m.training_values[1])
        grad_pref_ℓ = 2 .* hcat( [  m.rbf_coefficients[ i, ℓ ] * differences[i] for i = eachindex(differences) ] ... )
        push!(grad_prefix_matrices, grad_pref_ℓ)
    end
    return grad_prefix_matrices, differences
end
=#

@doc "Compute gradient term of ℓ-th RBF (scalar) model output and return n-vector."
function rbf_grad( m::RBFModel, ℓ :: Int64, x :: RVec )
    n_vars = length(m.training_sites[1]);

    difference_vectors = x_minus_sites(m,x); # array with len n_sites and entries n_vars
    distances = norm.(difference_vectors)       # n_sites x 1
    ∂kernel_vals = ∂kernel.( Val(m.kernel), distances, m.shape_parameter )  # n_sites

    coeff_ℓ = m.rbf_coefficients[:,ℓ]           # n_sites x 1
    grad_ℓ = zeros(eltype(x), n_vars);
    for i = 1 : length(distances)
        if distances[i] != 0
            grad_ℓ += coeff_ℓ[i] / distances[i] * ∂kernel_vals[i] .* difference_vectors[i]
        end
    end
    return grad_ℓ
end

@doc "Compute the Jacobian matrix of the RBF part of the model"
function rbf_jacobian( m::RBFModel, x :: RVec )
    n_vars = n_in( m );
    n_out = num_outputs( m );

    difference_vectors = x_minus_sites(m,x); # array with len n_sites and entries n_vars
    distances = norm.(difference_vectors)       # n_sites x 1
    ∂kernel_vals = ∂kernel.( Val(m.kernel), distances, m.shape_parameter )  # n_sites

    rbf_jac = zeros(eltype(x), n_out, n_vars);
    for ℓ = 1 : n_out
        coeff_ℓ = m.rbf_coefficients[:,ℓ]           # n_sites x 1
        grad_ℓ = zeros(eltype(x), n_vars);
        for i = 1 : length(distances)
            if distances[i] != 0
                grad_ℓ += coeff_ℓ[i] / distances[i] * ∂kernel_vals[i] .* difference_vectors[i]
            end
        end
        rbf_jac[ℓ,:] = grad_ℓ;
    end
    return rbf_jac
end
rbf_jacobian( m::RBFModel, x :: Real ) = rbf_jacobian( m, [x])

# === Hessians of the model ===
# note ∂ᵢ ∂kernel(r(x)) = (x_i -c_i) * ∂∂kernel(x)
∂∂kernel( ::Val{:exp}, r :: Real, s :: Real = 1 ) = 2 * s^2 * (2* r^2 * s^2 - 1) * kernel( Val(:exp), r, s );
∂∂kernel( ::Val{:multiquadric}, r :: Real, s :: Real = 1 ) = - s^2 / ((s*r)^2 + 1 )^(3/2) ;
∂∂kernel( ::Val{:cubic}, r :: Real, s :: Real = 1 ) = 6 * s^3 * r;
∂∂kernel( ::Val{:thin_plate_spline}, r :: Real, s :: Real = 1 ) = r == 0 ? 0 :
    (-1)^(s+1) * r^(2*s-2) *(
        4*s - 1 +
        2*(2*s-1)*s*log(r)
    );

function rbf_hessian( m::RBFModel, ℓ :: Int64, x :: RVec )
    n_vars = length(m.training_sites[1])

    coeff_ℓ = m.rbf_coefficients[:, ℓ]              # n_sites
    differences = x_minus_sites(m,x)        # n_sites
    distances = norm.( differences, 2 ); # n_sites

    ∂kernel_eval = ∂kernel.( Val(m.kernel), distances, m.shape_parameter )   # n_sites
    ∂∂kernel_eval = ∂∂kernel.( Val(m.kernel), distances, m.shape_parameter ) # n_sites

    rbf_hessian_ℓ = zeros( T, n_vars, n_vars )
    #rbf_hessian_ℓ .*= sum( coeff_ℓ .* ∂kernel_eval )
    for i = 1 : length(coeff_ℓ)
        φᵢ′ = ∂kernel_eval[i];
        φᵢ′′ = ∂∂kernel_eval[i];
        coeff_ℓ[i]
        rᵢ = distances[i]
        Dᵢ = differences[i];
        if rᵢ != 0
            𝚯ᵢ = ( (φᵢ′ / rᵢ) .* I(n_vars) ) .+ ( ( φᵢ′′ - φᵢ′/rᵢ ) / rᵢ^2  ) .* (Dᵢ * Dᵢ')
        else
            𝚯ᵢ = φᵢ′′ .* I(n_vars)
        end
        rbf_hessian_ℓ .+= coeff_ℓ[i] .* 𝚯ᵢ
    end
    return rbf_hessian_ℓ
end

# === Evaluate polynomial part of the model ===
poly(m::RBFModel, ℓ, x, ::Val{-1} ) = 0                       # degree -1 ⇒ No polynomial part (ℓ-th output)
poly(m::RBFModel, x, ::Val{-1} ) = 0                          # degree -1 ⇒ No polynomial part (all outputs)
poly(m::RBFModel, ℓ, x, ::Val{0} ) = m.poly_coefficients[ℓ]     # degree 0 ⇒ constant (ℓ-th output)
poly(m::RBFModel, x, ::Val{0} ) = m.poly_coefficients[:]        # degree 0 ⇒ constant (all outputs)
poly(m::RBFModel, ℓ, x, ::Val{1} ) = [x;1]'m.poly_coefficients[:, ℓ] # affin linear tail (ℓ-th output)
poly(m::RBFModel, x, ::Val{1} ) = vec([x;1]'m.poly_coefficients)    # affin linear tail (all outputs)

@doc "Evaluate polynomial tail for output ℓ."
function poly_output( m::RBFModel, ℓ :: Int64, x :: RVec )
    poly( m, ℓ, x, Val(m.polynomial_degree) )
end

@doc "k-Vector of evaluations of polynomial tail for all outputs."
function poly_output( m::RBFModel, x :: RVec )
    poly( m, x, Val(m.polynomial_degree) )
end

∇poly(m::RBFModel, ℓ :: Int64 , ::Union{Val{-1}, Val{0} } ) = zeros( length( m.training_sites[1] ) )
∇poly(m::RBFModel, ℓ :: Int64 , ::Val{1} ) = m.poly_coefficients[1 : end - 1, ℓ];    # return all coefficients safe c_{n+1}
@doc "Gradient vector for polynomial tail of ℓ-th output."
function poly_grad( m::RBFModel, ℓ :: Int64 , x :: RVec )
    ∇poly( m, ℓ, Val(m.polynomial_degree) )
end

function poly_jacobian( m::RBFModel, x :: RVec )
    hcat( [poly_grad(m, ℓ, x) for ℓ = 1 : length(m.training_values[1] ) ]... )'
end

#=
## For polynomials of degree at most 1 the hessian is always zero
poly_hessian( m::RBFModel, ℓ::Int64, x::RVec)
=#

# === Combined model output ===
@doc "Evaluate ℓ-th (scalar) model output at vector x."
function output( m::RBFModel, ℓ :: Int64, x :: RVec )
    rbf_output( m, ℓ, x ) + poly_output( m, ℓ, x )
end
#output( m::RBFModel, ℓ :: Int64, x :: Real ) = output(m, ℓ, [x])

@doc "Evaluate all (scalar) model outputs at vector x and return k-vector of results."
function output( m::RBFModel, x :: RVec )
    vec(rbf_output( m, x ) .+ poly_output( m, x ))
end
#output( m::RBFModel, x :: Real ) = output(m, [x])

function grad( m::RBFModel, ℓ :: Int64, x :: RVec )
    rbf_grad( m, ℓ, x ) + poly_grad( m, ℓ, x)
end
#grad(m::RBFModel, ℓ::Int64, x::Real) = grad(m, ℓ, [x])   # if n_vars == 1 and RBFModel is used outside of Optimization

function jac( m::RBFModel, x :: RVec )
    rbf_jacobian( m, x ) + poly_jacobian( m , x )
end
#jac(m::RBFModel, x::Real) = jac(m,[x])         # if n_vars == 1 and RBFModel is used outside of Optimization

function hess(m :: RBFModel, ℓ :: Int64, x :: RVec)
    rbf_hessian(m,ℓ,x) # + poly_hessian == zeros
end
#hess(m::RBFModel, ℓ::Int64, x::Real) = hess(m,ℓ,[x,])         # if n_vars == 1 and RBFModel is used outside of Optimization

# === Utiliy functions for solving the normal equations
# TODO right matrix types
get_Π( m :: RBFModel, ::Val{-1} ) =  Matrix{Real}( undef, 0, length(m.training_sites) );
get_Π( m :: RBFModel, ::Val{0} ) = ones(1, length(m.training_sites));
get_Π( m :: RBFModel, ::Val{1} ) = [ hcat( m.training_sites...); ones(1, length(m.training_sites)) ];

@doc "Return polynomial base matrix. Ones in last row."
function get_Π(m::RBFModel)
    get_Π( m, Val( m.polynomial_degree ) )
end

@doc "Return column vector to augment the polynomial base matrix `Π` if `y` were added."
function Π_col( m :: RBFModel , y :: RVec )
    pd = m.polynomial_degree
    if pd == -1
        return Matrix{Real}(undef, 0, 1)
    elseif pd == 0
        return 1
    elseif pd == 1
        return [ y ; 1 ]
    end
end

# see [WILD 2008]
function null_space_coefficients( Q, R, Z , L, f, Φ )
    rhs = Z'f;
    w = L \ rhs;
    ω = L' \ w;
    λ = Z * ω;  # RBF coefficients

    rhs_poly = Q'*( f.- Φ * λ)
    v = R \ rhs_poly;
    return λ, v
end

@doc "Solve RBF linear equation system using Cholesky based null space method as in [WILD 2008]."
function solve_rbf_problem( Π, Φ, f, :: Val{true} )
    Q, R = qr( Π' );
    R = [
        R;
        zeros( size(Q,1) - size(R,1), size(R,2) )
    ]
    Z = Q[:, size(Π,1) + 1 : end ]

    ZΦZ = Hermitian( Z'Φ*Z );   # ensure it is really Symmetric (rounding errors etc)
    #@show eigen(ZΦZ).values
    try
        L = cholesky( ZΦZ ).L     # should also be empty at this point
        λ, v = null_space_coefficients( Q, R, Z, L, f, Φ)
        return vcat( λ, v )

    catch PosDefException
        return solve_rbf_problem(Π,Φ,f,Val(false))
    end
end

@doc "Solve the RBF linear equation system using LU or QR factorization."
function solve_rbf_problem( Π, Φ, f, :: Val{false} )
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
function set_coefficients!( m :: RBFModel, coefficients :: AbstractArray{R} where R<:Real)
    pd = m.polynomial_degree
    n_vars = n_in(m);
    n_out = num_outputs(m);
    n_sites = length( m.training_sites );
    if pd == -1
        m.rbf_coefficients = reshape(coefficients, (n_sites, n_out));
        m.poly_coefficients =  Matrix{Real}( undef, 0, n_out );
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
    return m
end

# same as train! but using a null space method with data available from 'add_points!' method
# USE ONLY FOR MODELS WITH is_valid(m) == true !!!
function train!(m::RBFModel, Q , R, Z, L, Φ)

    RHS = hcat( m.training_values... )';
    λ, v = null_space_coefficients( Q, R, Z, L, RHS, Φ)
    set_coefficients!(m, vcat(λ,v))
    return m
end

function _test_interpolation( m :: RBFModel )
    for (i,site) in enumerate(m.training_sites)
        @assert all( isapprox.( m(site), m.training_values[i] ) )
    end
end

end#module
