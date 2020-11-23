include("Polynomials.jl")
using .Polynomials

n = 2 # number of variables
D = 2 # degree of polynomials
p = binomial( n + D, n)     # required interpolation points

using NLopt

# %%
@doc """
Return array of solution vectors [x_1, …, x_len] to the equation
``x_1 + … + x_len = rhs``
where the variables must be non-negative integers.
"""
function non_negative_solutions( rhs :: Int64, len :: Int64 )
    if len == 1
        return rhs
    else
        solutions = [];
        for i = 0 : rhs
            for shorter_solution ∈ non_negative_solutions( i, len - 1)
                push!( solutions, [ rhs-i; shorter_solution ] )
            end
        end
        return solutions
    end
end

# Generate the canonical basis for space of polynomial of degree at most D
canonical_basis = Vector{Polynomial{n}}();
for d = 0 : D
    for sol in non_negative_solutions( d, n )
        m = Polynomials.MultiIndex( sol )
        poly = Polynomial( 1 / factorial( m ), m )
        push!(canonical_basis, poly)
    end
end

# %%

LB = [0.95, 0.0]
UB = [1.0, 1.0]
W = UB .- LB;
rand_box_point = () -> LB .+ W .* rand(length(LB))

p_ini = p
Y_init = [ LB .+ (UB .- LB) .* rand(2) for i = 1 : p_ini]
Y_poised = Vector{Vector{Float64}}()

# Algorithm 6.2 (p. 95, Conn)
# # select or generate a poised set suited for interpolation
# # also computes the lagrange basis functions
lagrange_basis = copy( canonical_basis )
ε_accept = 1e-5
for i = 1:p
    # Point selection
    lyᵢ, jᵢ = isempty(Y_init) ? (0,0) : findmax( abs.( lagrange_basis[i].(Y_init)) )
    if lyᵢ > ε_accept
        yᵢ = Y_init[jᵢ]
    else
        @info "It. $i: Computing a poised point by Optimization."
        opt = Opt(:LN_BOBYQA, 2)
        opt.lower_bounds = LB
        opt.upper_bounds = UB
        opt.min_objective = (x,g) -> lagrange_basis[i](x)
        y₀ = rand_box_point()
        (_, yᵢ, _) = optimize(opt, y₀)
    end
    if jᵢ > 0 deleteat!( Y_init, jᵢ ) end
    push!(Y_poised, yᵢ)

    # Normalization
    lagrange_basis[i] /= lagrange_basis[i](yᵢ)

    # Orthogonalization
    for j = 1:p
        if j≠i
            lagrange_basis[j] -= (lagrange_basis[j](yᵢ) * lagrange_basis[i])
        end
    end
end

Y_poised = Y_poised[ [isassigned(Y_poised, i) for i=eachindex(Y_poised) ] ]

scatter_x, scatter_y = let Ymat = hcat( Y_poised... );
    Ymat[1,:], Ymat[2,:]
end
using Plots
scatter( scatter_x, scatter_y; label = "poised" )

#%%
# Algorithm 6.3 (p.96 Conn)
Λ = 1.4
while true
    # 1) Λ calculation
    lagrange_abs_vals = Vector{Float64}( undef, p )
    lagrange_abs_maximizers = Vector{Vector{Float64}}(undef, p)
    for i = 1 : p
        opt = Opt(:LN_BOBYQA, 2)
        opt.lower_bounds = LB
        opt.upper_bounds = UB
        opt.max_objective = (x,g) -> abs(lagrange_basis[i](x))
        y₀ = rand_box_point()
        (abs_lᵢ, yᵢ, _) = optimize(opt, y₀)
        lagrange_abs_vals[i] = abs_lᵢ
        lagrange_abs_maximizers[i] = yᵢ
    end
    Λₖ₋₁, iₖ = findmax( lagrange_abs_vals )
    @show Λₖ₋₁

    # 2) Point swap
    if Λₖ₋₁ ≥ Λ
        Y_poised[ iₖ ] = lagrange_abs_maximizers[ iₖ ]
    else
        break;
    end

    # 3) Lagrange Basis update
    ## Normalization
    lagrange_basis[iₖ] /= lagrange_basis[iₖ]( Y_poised[iₖ] )

    # Orthogonalization
    for j = 1 : p
        if j ≠ iₖ
            lagrange_basis[ j ] -= ( lagrange_basis[j](Y_poised[iₖ])* lagrange_basis[ iₖ ] )
        end
    end
end


scatter_x, scatter_y = let Ymat = hcat( Y_poised... );
    Ymat[1,:], Ymat[2,:]
end
scatter!( scatter_x, scatter_y; label = "Λ-poised" )
