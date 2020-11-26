using Morbit
using DynamicPolynomials
using FileIO,JLD2

using Logging
logger = ConsoleLogger(stdout, Logging.Warn)
global_logger(logger);

# This is not really a Test script but rather a utility...
# Pre-generate beautiful interpolation sets for Lagrange DynamicPolynomial
# Save a dictionary mapping the number of variables to the corresponding 
# set of Lagrange basis polynomials and sites...

# SETTINGS
SAVE_PATH = joinpath( @__DIR__, "..", "src", "data");

NVARS = collect(6:40);
D = 2;  # Lagrange polynomial degree
Λ = 1.2;

ε_accept = 1e-6; 

Threads.@threads for n ∈ NVARS
    println("NUM VAR $n")

    p = binomial( n + D, n)
    # Generate the canonical basis for space of polynomial of degree at most D
    canonical_basis = Any[];
    @polyvar χ[1:n];
    for d = 0 : D
        for multi_exponent in Morbit.non_negative_solutions( d, n )
            poly = 1/Morbit.multifactorial(multi_exponent) * prod( χ.^multi_exponent );
            push!(canonical_basis, poly)
        end
    end

    # Find a nice point set in the unit hypercube
    X = .5 .* ones(n); # center point

    # find poiset set in hypercube
    new_sites, recycled_indices, lagrange_basis = Morbit.find_poised_set( ε_accept, 
        canonical_basis, [X,], X, .5, 1, zeros(n), ones(n) );

    # make Λ poised
    Morbit.improve_poised_set!(lagrange_basis, new_sites, recycled_indices, Λ, 
        [X,], zeros(n), ones(n) );
        # save pre-calculated Lagrange basis in config

    stencil_sites = [ [X,][recycled_indices]; new_sites ];
    println( joinpath( SAVE_PATH, "lagrange_basis_$(n)_vars.jld2" ) );
    save( joinpath( SAVE_PATH, "lagrange_basis_$(n)_vars.jld2" ), Dict( 
        "Λ" => Λ,
        "lagrange_basis" => lagrange_basis, 
        "sites" => stencil_sites
    ));
end