# Miscellanuous functions that don't fit anywhere else

# some utily functions to draw samples from specific pdfs
randquad(n) = -( 1 .- rand(n) ).^(1/3) .+ 1; # sample randomly according to pdf 3(x-1)^2
randquad() = randquad(1)[end];
randquart(n) = - ( 1 .- rand(n) ).^(1/5) .+ 1;
randquart() = - ( 1 - rand() )^(1/5) + 1

@doc "Return an enlargment factor `θ` that is sensible for [0,1]^n constrained problems."
function sensible_θ( constrained :: Val{true}, θ :: Float64,
        x :: Vector{Float64}, Δ :: Float64 )
    # Define the effective trust region radius `effective_Δ` as
    # the smallest box radius so some step of length at most `Δ` can be taken
    # whilst honoring the global box constraints [0,1]^n
    effective_Δ = min( Δ, max( maximum( x ), maximum(1.0 .- x )) )
    θ = min(  θ, effective_Δ / Δ )
end

# the above function is not necessary in unconstrained problems...
sensible_θ(::Val{false}, θ::Float64, x::Vector{Float64}, Δ::Float64 ) = θ

@doc "Return indices of sites in `sites_array` so that `x .- Δ <= site <= x .+ Δ`
and exclude index of `x` if contained in `sites_array`."
function find_points_in_box( x, Δ, sites_array, filter_x :: Val{true} )
   x_lb = x .- Δ;
   x_ub = x .+ Δ;
   candidate_indices = findall( site -> all( x_ub .>= site .>= x_lb ) && !isapprox(x, site, rtol = 1e-14), sites_array);   # TODO check isapprox relative tolerance
end

@doc "Return indices of sites in `sites_array` so that `x .- Δ <= site <= x .+ Δ`
and assume that `sites_array` does not contain `x`."
function find_points_in_box( x, Δ, sites_array, filter_x :: Val{false} )
   x_lb = x .- Δ;
   x_ub = x .+ Δ;
   candidate_indices = findall( [ all( x_lb .<= site .<= x_ub ) for site ∈ sites_array ] )
end

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

@doc "Evaluate the objective functions (referenced in `config_struct.problem`) at sites `additional_sites`
and push the results to `config_struct.iter_data` arrays. Return indices of results in `sites_db`."
function eval_new_sites( config_struct :: AlgoConfig, additional_sites :: Vector{Vector{Float64}})
   @unpack iter_data, problem = config_struct;

   @info("\t\tEvaluating at $(length(additional_sites)) new sites.")
   additional_values = eval_all_objectives.(problem, additional_sites)  # automatically unscales

   additional_indices = let nold = length(iter_data.sites_db), nnew = length(additional_sites);
       nold + 1 : nold + nnew
   end

   push!(iter_data.sites_db, additional_sites...)
   push!(iter_data.values_db, additional_values...)
   return additional_indices
end
