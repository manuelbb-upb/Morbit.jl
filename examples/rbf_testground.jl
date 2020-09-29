using Parameters: @with_kw
using LinearAlgebra
#export RBFModel, train!, output, grad

@with_kw mutable struct RBFModel
    training_sites :: Array{Array{Float64, 1},1} = [];
    training_values :: Array{Array{Float64, 1}, 1} = [];
    kernel :: Symbol = :multiquadric;
    shape_parameter :: Union{Float64, Vector{Float64}} = 1.0;
    fully_linear :: Bool = false;
    polynomial_degree :: Int64 = -1;

    rbf_coefficients :: Array{Float64,2} = Matrix{Float64}(undef,0,0);
    poly_coefficients :: Array{Float64,2} = Matrix{Float64}(undef,0,0);

    function_handle :: Union{Nothing, Function} = nothing;
    @assert polynomial_degree <= 1 "For now only polynomials with degree -1, 0, or 1 are allowed."
end

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

kernel( ::Val{:exp}, r, s = 1.0 ) = exp(-(r*s)^2) :: Float64;
kernel( ::Val{:multiquadric} , r, s = 1.0 ) = - sqrt( 1 + (s*r)^2) :: Float64;
kernel( ::Val{:cubic} , r, s = 1.0 ) = (r*s)^3 :: Float64;
kernel( ::Val{:thin_plate_spline}, r, s = 1.0) = r == 0 ? 0.0 : r^2 * log(r) :: Float64;

@doc "Return n_sites-array with difference vectors (x_1 - c_{1,m}, …, x_n - c_{n,m}) of length n."
function x_minus_sites( m:: RBFModel, x :: Vector{T} where{T<:Real})
    [ x .- site for site ∈ m.training_sites ]
end

@doc "Return vector of all distance values between x and each center of m"
function center_distances( m :: RBFModel, x :: Vector{T} where{T<:Real} )
    r_vector = norm.( x_minus_sites( m, x), 2 )
end

@doc "Evaluate all ``n_c`` basis functions of m::RBFModel at second argument x and return ``n_c``-Array"
function φ( m::RBFModel, x :: Vector{T} where{T<:Real} ) :: Vector{Float64}
    kernel.( Val(m.kernel), center_distances(m, x), m.shape_parameter )
end

@doc "Symmetric matrix of every center evaluated in all basis functions."
function get_Φ( m::RBFModel )
    if isempty(m.training_sites)
        return Matrix{Float64}(undef, 0, 0)
    else
        transpose(hcat( φ.(m, m.training_sites)... ))
    end
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
output( m::NamedTuple, ℓ :: Int64, x :: Real ) = Float64[];

@doc "Evaluate all (scalar) model outputs at vector x and return k-vector of results."
function output( m::RBFModel, x :: Vector{T} where{T<:Real} )
    vec(rbf_output( m, x ) .+ poly_output( m, x ))
end
output( m::RBFModel, x :: Real ) = output(m, [x])
output( m::NamedTuple, x :: Union{Real, Vector{T} where{T<:Real}} ) = Float64[]

function grad( m::RBFModel, ℓ :: Int64, x :: Vector{T} where{T<:Real} )
    rbf_grad( m, ℓ, x ) + poly_grad( m, ℓ, x)
end
grad(m::RBFModel, ℓ::Int64, x::Real) = grad(m, ℓ, [x])   # if n_vars == 1 and RBFModel is used outside of Optimization
grad(m::NamedTuple, ℓ::Int64, x :: Union{Real, Vector{T} where{T<:Real}} ) = Float64[]

function jac( m::RBFModel, x :: Vector{T} where{T<:Real} )
    rbf_jacobian( m, x ) + poly_jacobian( m , x )
end
jac(m::RBFModel, x::Real) = jac(m,[x])         # if n_vars == 1 and RBFModel is used outside of Optimization
jac(m::NamedTuple, x :: Union{Real, Vector{T} where{T<:Real}} ) = Matrix{Float64}(undef, 0, length(x))

# === Utiliy functions for solving the normal equations
get_Π( m :: RBFModel, ::Val{-1} ) =  Matrix{Float64}( undef, 0, length(m.training_sites) );
get_Π( m :: RBFModel, ::Val{0} ) = ones(1, length(m.training_sites));
get_Π( m :: RBFModel, ::Val{1} ) = [ hcat( m.training_sites...); ones(1,length(m.training_sites)) ];

@doc "Return polynomial base matrix. Ones in last row."
function get_Π(m::RBFModel)
    get_Π( m, Val( m.polynomial_degree ) )
end

function solve_rbf_problem( Π, Φ, f )

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
function set_coefficients!( m :: RBFModel, coefficients )
    pd = m.polynomial_degree
    n_vars = length(m.training_sites[1]);
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

    coefficients = solve_rbf_problem( Π, Φ, RHS )
    set_coefficients!(m, coefficients)
    m.function_handle = x -> output(m, x);
    return m
end

# same as train! but using a null space method with data available from 'additional_points!' method
function train!(m::RBFModel, Π, Q, R, Z, L)
    if m.polynomial_degree == 1
        n_out = length(m.training_values[1])
        Φ = get_Φ( m );
        n_sites = length(m.training_sites)

        f = hcat( m.training_values... )';

        R = Matrix{Float64}(I, size(Q,1), size(R,1)) * R;   # get full R matrix ( not truncated R matrix)

        # compute rbf coefficients using a null space method and provided basis Z
        rhs = Z'f;
        w = L \ rhs;
        ω = L' \ w;
        λ = Z * ω;  # actual coefficients
        # compute polynomial coefficients
        rhs_poly = Q'*(f.-Φ*λ)
        ν = R \ rhs_poly;

        coefficients = vcat( λ, ν );

        # adjust for offset in data sites
        coeff_offset = (m.training_sites[1]'coefficients[n_sites+1:end-1, :])[1];
        #@show coeff_offset
        coefficients[end, :] .-= coeff_offset

        set_coefficients!(m, coefficients)
        m.function_handle = x -> output(m, x);

        return m
    else
        train!(m)
    end
end

##############

function Π_col( m, y )
    pd = m.polynomial_degree
    n_vars = length(m.training_sites[1]);
    if pd == -1
        return Matrix{Float64}(undef, 0, 1)
    elseif pd == 0
        return 1.0
    elseif pd == 1
        return [ y ; 1.0 ]
    end
end

##################################################
nvars = 2
pdeg = 1

sites = [ rand(2) for i = 1 : 3 ]#(pdeg > 0 ? npolypoints : 1 )]
vals = (x -> [sum( x.^2 )]).(sites)

x = sites[1]

m = RBFModel(
    training_sites = sites,
    training_values = vals,
    kernel = :exp,
    polynomial_degree = pdeg
)

Π = get_Π(m)
Φ = get_Φ(m)

Q,R = qr( Π' )
R = [
    R;
    zeros( size(Q,1) - size(R,1), size(R,2) )
]
Z = Q[:, 2 + 1 : end ]

ZΦZ = Hermitian(Z'Φ*Z)

L = cholesky(ZΦZ).L
Lⁱ = inv(L);

φ₀ = Φ[1,1]
#f = collect( vals... )

new_sites = [ rand(2) for i = 1 : 5]

θ_pivot_cholesky = 1e-7;

np = size(R,2)
k = 0   # number of accepted sites
for y in new_sites
    global Φ, Q, R, Z, Π, ZΦZ, L, Lⁱ, φ₀, θ_pivot_cholesky
    φy = φ(m, y)
    Φy = [
        [Φ φy];
        [φy' φ₀]
    ]

    πy = Π_col( m, y );
    Ry = [
        R ;
        πy'
    ]

    # perform some givens rotations to turn last row in Ry to zeros
    row_index = size( Ry, 1)
    G = Matrix(I, row_index, row_index)
    for j = 1 : np  # column index
        g = givens( Ry[j,j], Ry[row_index, j], j, row_index )[1];
        Ry = g*Ry;
        G = g*G;
    end
    Gᵀ = transpose(G)
    g̃ = Gᵀ[1 : end-1, end];   #last column
    ĝ = Gᵀ[end, end];

    Qg = Q*g̃;
    v_y = Z'*( Φ*Qg + φy .* ĝ );
    σ_y = Qg'*Φ*Qg + (2*ĝ)* φy'*Qg + ĝ^2*φ₀;

    τ_y² = σ_y - norm( Lⁱ * v_y, 2 )^2

    if τ_y² > θ_pivot_cholesky
        τ_y = sqrt(τ_y²)

        R = Ry;

        z = [
            Q * g̃;
            ĝ
        ]
        Z = [
            [
                Z;
                zeros(1, size(Z,2))
            ] z
        ]
        #=
        L = [
            [ L zeros(size(L,1), 1) ];
            [ v_y'Lⁱ' τ_y ]
        ]
        =#

        Lⁱ = [
            [Lⁱ zeros(size(Lⁱ,1),1)];
            [ -(v_y'Lⁱ'Lⁱ)./τ_y 1/τ_y ]
        ]
        #=
        ZΦZ = [
            [ ZΦZ v_y ];
            [ v_y' σ_y ]
        ]
        =#
        Q = [
            [ Q zeros( size(Q,1), 1) ];
            [ zeros(1, size(Q,2)) 1.0 ]
        ] * Gᵀ

        Π = [ Π πy ];
        Φ = Φy;

        push!(m.training_sites, y)
        push!(m.training_values, [sum(y.^2)] )
    end

end
