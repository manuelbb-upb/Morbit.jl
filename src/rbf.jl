
function (m::RBFModel)(eval_site::Array{Float64, 1})
    m.function_handle(eval_site)  # return an array of arrays for ease of use in other methods
end

const exp_kernel(r,s) = exp(-(r/s)^2);
const multiquadric_kernel(r,s) = -sqrt( 1 + (s*r)^2);
const cubic_kernel(r,s) = (r*s)^3;
const thin_plate_spline_kernel(r,s) = r == 0 ? 0.0 : r^2 * log(r);

function as_second!(destination :: RBFModel, source :: RBFModel )
    @unpack_RBFModel source;
    @pack_RBFModel! destination;
    return nothing
end

function get_basis_vector_function(center_sites, kernel_name, shape_parameter )

    kernel_func(r) = eval( Symbol(kernel_name, "_kernel") )(r, shape_parameter)

    function basis_vector_function( eval_site )
        eval_single_basis( center ) = kernel_func( norm( center .- eval_site, 2 ) )
        basis_vector = eval_single_basis.(center_sites)
    end
end

function build_Φ( sites, kernel_name = "multiquadric", shape_parameter = 1)
    φ = get_basis_vector_function( sites, kernel_name, shape_parameter )
    Φ = hcat( ( φ.(sites))... );             # apply φ to all sites and form rbf matrix
    return (Φ, φ)
end

function build_Φ( m::RBFModel )
    return build_Φ( m.training_sites, m.kernel, m.shape_parameter)
end

function get_poly_function( poly_degree, coefficient_matrix, n_sites )
    if poly_degree == -1
        return x -> 0;
    elseif poly_degree == 0
        return x -> coefficient_matrix[end, :];    # simply output one coefficient for each output index
    elseif poly_degree == 1
        return x -> vec([x' 1.0] * coefficient_matrix[ n_sites + 1 : end, : ]);
    end
end

function get_Π(m::RBFModel)
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


@doc "Train (and fully instanciate) a RBFModel instance to best fit its training data."
function train!( m::RBFModel )
    n_sites = length( m.training_sites );
    n_out = length( m.training_values[1] );

    Φ,φ = build_Φ( m );
    Π = get_Π(m);
    f = hcat( m.training_values... )';

    coefficients = solve_rbf_problem( Π,Φ, f)

    rbf_function(x) = φ(x)'coefficients[1:n_sites, :];
    poly_function = get_poly_function( m.polynomial_degree, coefficients, n_sites)

    m.function_handle = x -> vec(rbf_function(x)) .+ poly_function(x); #b
    #m.function_handle = x -> vec(rbf_function(x)) .+ poly_function(x + m.training_sites[1]); #a  # vec used for backwards compatibility

    return m
end

# same as train! but using a null space method with data available from 'additional_points!' method
function train!(m::RBFModel, Π, Q, R, Z, L)
    if m.polynomial_degree == 1
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

        rbf_function(x) = φ(x)'coefficients[1:n_sites, :];
        poly_function = get_poly_function( 1, coefficients, n_sites)

        m.function_handle = x -> vec(rbf_function(x)) .+ poly_function(x - m.training_sites[1]);

        return m
    else
        train!(m)
    end
end
