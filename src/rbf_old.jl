
#=

function (m::RBFModel)(eval_site::Array{Float64, 1})
    m.function_handle(eval_site)  # return an array of arrays for ease of use in other methods
end

const exp_kernel(r,s) = exp(-(r/s)^2);
const multiquadric_kernel(r,s) = - sqrt( 1 + (s*r)^2);
const cubic_kernel(r,s) = (r*s)^3;
const thin_plate_spline_kernel(r,s) = r == 0 ? 0.0 : r^2 * log(r);

# derivatives wrt r
const ∂exp_kernel(r,s) = - s^2 * exp_kernel(r,s);
const ∂multiquadric_kernel(r,s) = s^2 / (2 * multiquadric_kernel(r,s));
const ∂cubic_kernel(r,s) = s^3 * ( r + r/2 ) ;
const ∂thin_plate_spline_kernel(r,s) = r == 0 ? 0.0 : log(r) + 1/2;


#=
function build_Φ( sites, kernel_name = "multiquadric", shape_parameter = 1)
    φ = get_basis_vector_function( sites, kernel_name, shape_parameter )
    Φ = hcat( ( φ.(sites))... );             # apply φ to all sites and form rbf matrix
    return (Φ, φ)
end

function get_basis_vector_function(center_sites, kernel_name, shape_parameter )

    kernel_func(r) = eval( Symbol(kernel_name, "_kernel") )(r, shape_parameter)

    function basis_vector_function( eval_site )
        eval_single_basis( center ) = kernel_func( norm( center .- eval_site, 2 ) )
        basis_vector = eval_single_basis.(center_sites)
    end
end
=#
#=
function build_Φ( sites, kernel_name = "multiquadric", shape_parameter = 1)
    φ = get_basis_vector_function( sites, kernel_name, shape_parameter )
    Φ = hcat( ( φ.(sites))... );             # apply φ to all sites and form rbf matrix
    return (Φ, φ)
end

function get_basis_vector_function(center_sites, kernel_name, shape_parameter )

    kernel_func(r) = eval( Symbol(kernel_name, "_kernel") )(r, shape_parameter)

    function basis_vector_function( eval_site )
        eval_single_basis( center ) = kernel_func( norm( center .- eval_site, 2 ) )
        basis_vector = eval_single_basis.(center_sites)
    end
end
=#

function as_second!(destination :: RBFModel, source :: RBFModel )
    @unpack_RBFModel source;
    @pack_RBFModel! destination;
    return nothing
end

@doc raw"Return a function $φ\colon \mathbb R^n \to \mathbb R^{n_c}$ where $n_c$ is the number of training_sites and φ evaluates its input for each basis function."
get_φ( m :: RBFModel )
    kernel = r -> eval( Symbol(m.kernel, "_kernel") )(r, m.shape_parameter)
    φ( x :: Array{Float64, 1} ) = x -> kernel.( norm.( [ x .- center for center ∈ m.training_sites ] ) )
end

# mainly kept for backwards compatibilty
@doc raw"""Use the basis vector function $φ$ to calucate the symmetric $n_c\times n_c$ Matrix $Φ$ evaluating $φ$ at each training site.
Returns Φ,φ.
"""
function build_Φ( m :: RBFModel )
    φ = get_φ( m );
    Φ = hcat( φ.( m.training_sites )... )
    return Φ,φ
end

function φ( m:: RBFModel )

function set_output!( m :: RBFModel, k :: Int64 )
end

function gradient( m :: RBFModel, output_index )
end


function get_basis_vector_derivatives(center_sites, kernel_name, shape_parameter)
    kernel_func(r) = eval( Symbol("∂", kernel_name, "_kernel") )(r, shape_parameter)

    function basis_vector_function( eval_site )
        eval_single_basis( center ) = kernel_func( norm( center .- eval_site, 2 ) )
        basis_vector = eval_single_basis.(center_sites)
    end
end



function build_rbf_gradients( m :: RBFModel)
    ∂̂φ = get_basis_vector_derivatives( m.training_sites, m.kernel, m.shape_parameter ) # maps n-vector to n_centers-vector of rbf derivative valuations
    gradient_prefixes(site) = 2 .* [ site .- center for center in m.training_sites ] # n_centers array containing n-vectors of differences
    ∂φ(site) = vcat( (∂̂φ(site) .* gradient_prefixes(site))'... )     # each row corresponds to gradient of a single rbf basis function
end

function get_poly_function( poly_degree, coefficient_matrix, n_sites )
    # assume x to be a coordinate column vector
    # return a function handle that maps x to a scalar or a k-Vector (considering k outputs)
    if poly_degree == -1
        return x -> 0.0;
    elseif poly_degree == 0
        return x -> coefficient_matrix[end, :];    # simply output one coefficient for each output index
    elseif poly_degree == 1
        return x -> [x'; 1.0]'coefficient_matrix[ n_sites + 1 : end, : ];
    end
end

function get_poly_function( poly_degree, coefficient_matrix, n_sites, output_index )
    # assume x to be a coordinate column vector
    # return a function handle that maps x to a scalar or a k-Vector (considering k outputs)
    if poly_degree == -1
        return x -> 0.0;
    elseif poly_degree == 0
        return x -> coefficient_matrix[end, output_index];    # simply output one coefficient for each output index
    elseif poly_degree == 1
        return x -> [x; 1.0]'coefficient_matrix[ n_sites + 1 : end, output_index ];
    end
end

function get_poly_jacobian( poly_degree, coefficient_matrix, n_sites )
    if -1 <= poly_degree <= 0
        return x -> 0.0;
    else
        return x -> coefficient_matrix[n_sites + 1 : end - 1, :]';
    end
end

function get_poly_gradient( poly_degree, coefficient_matrix, n_sites, output_index)
    if -1 <= poly_degree <= 0
        return x -> 0.0;
    else
        return x -> coefficient_matrix[n_sites + 1 == end - 1 ? n_sites + 1 : n_sites + 1 : end - 1, output_index];
    end
end

function get_Π(m::RBFModel)
    # return augmented site coefficient matrix for computation of polynomial coeffients

    n_sites = length(m.training_sites);
    if m.polynomial_degree == -1
        Π = Matrix{Float64}(undef, 0, n_sites );
    elseif m.polynomial_degree == 0
        Π = ones(1, n_sites);
    elseif m.polynomial_degree == 1
        #Π = [ hcat( [π - m.training_sites[1] for π = m.training_sites]... ); ones(1, n_sites) ]; #a
        Π = [ hcat( m.training_sites...); ones(1,n_sites) ];       #b
    end
end

function solve_rbf_problem( Π, Φ, f )

    Φ_augmented = [ [Φ Π']; [Π zeros(size(Π,1),size(Π,1) )] ];
    f_augmented = [ f;
                    zeros( size(Π,1), size(f,2) ) ];
    local Φ_fact;   # factorized matrix (for min norm solution if Φ is singular)
    try
        Φ_fact = lu( Φ_augmented );
    catch SingularException
        println("\tWARNING RBF Matrix singular. Using QR decomposition.")
        Φ_fact = qr(Φ_augmented, Val(true))   # use QR to still obtain a (minimal norm) solution if Φ is near singular
    end
    coefficients = Φ_fact \ f_augmented;   # obtain model coefficients as min norm solution to LGS
end

function set_output_funcs!(m, φ, coeff, n_sites, n_out )
    # sets a function handle that maps n_vars-vector to k-vector
    # and an array of k scalar-valued function handles
    poly_func = get_poly_function( m.polynomial_degree , coeff, n_sites );
    handle_array = Array{Any,1}();
    for ℓ = 1 : n_out
        output_ℓ = x -> φ(x)'coeff[1:n_sites, ℓ] .+ get_poly_function( m.polynomial_degree, coeff, n_sites, ℓ)(x)
        push!(handle_array, output_ℓ)
    end

    m.output_handles = handle_array;
    m.function_handle = x-> φ(x)'coeff[1:n_sites, :] .+ get_poly_function( m.polynomial_degree, coeff, n_sites)(x) #x -> [ m.output_handles[ℓ] for ℓ = 1 : n_out ];
end

function set_derivative_funcs!( m, coefficients, n_sites, n_out )
    ∂φ = build_rbf_gradients( m )
    m.gradient_handles = [];
    for ℓ = 1 : n_out
        poly_grad = get_poly_gradient( m.polynomial_degree, coefficients, n_sites, ℓ)
        grad = x -> begin
            retVal = ∂φ(x)'vec(coefficients[1:n_sites, ℓ])  .+ poly_grad(x);
            if length(x) == 1 retVal = retVal[1] end
            end
        push!(m.gradient_handles,grad);
    end

    poly_jacobian = get_poly_jacobian( m.polynomial_degree, coefficients, n_sites)
    m.jacobian_handle = x -> coefficients[1:n_sites, :]'∂φ(x) .+ poly_jacobian(x)
end

@doc "Train (and fully instanciate) a RBFModel instance to best fit its training data."
function train!( m::RBFModel )
    n_sites = length( m.training_sites );
    n_out = length( m.training_values[1] );

    Φ,φ = build_Φ( m );
    Π = get_Π(m);
    f = hcat( m.training_values... )';

    coefficients = solve_rbf_problem( Π, Φ, f)

    set_output_funcs!(m, φ, coefficients, n_sites, n_out );
    set_derivative_funcs!( m, coefficients, n_sites, n_out )

    return m
end

# same as train! but using a null space method with data available from 'additional_points!' method
function train!(m::RBFModel, Π, Q, R, Z, L)
    if m.polynomial_degree == 1
        n_out = length(m.training_values[1])
        Φ, φ = build_Φ(m);
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
        @show coeff_offset
        coefficients[end, :] .-= coeff_offset

        set_output_funcs!(m, φ, coefficients, n_sites, n_out );
        set_derivative_funcs!( m, coefficients, n_sites, n_out )

        @show m.training_values
        @show m.function_handle( m.training_sites[1] )

        return m
    else
        train!(m)
    end
end
=#
