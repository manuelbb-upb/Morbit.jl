include("RBFBase.jl");
import .RBF

@with_kw mutable struct RbfModel <: SurrogateModel
    model :: RBFModel

    # matrices for the sampling and improvement algorithm
    Y :: RMat = Matrix{Real}(undef, 0, 0);
    Z :: RMat = Matrix{Real}(undef, 0, 0);
    
    fully_linear :: Bool = false;
end

@with_kw mutable struct RbfConfig <: SurrogateConfig
    kernel :: Symbol = :cubic;
    shape_parameter :: Union{AbstractString, Real} = 1;
    polynomial_degree :: Int64 = 1;

    θ_enlarge_1 :: Real = 2;
    θ_enlarge_2 :: Real = 5;  # reset
    θ_pivot :: Real = 1 / (2 * θ_enlarge_1);
    θ_pivot_cholesky :: Real = 1e-7;

    require_linear :: Bool = false;

    max_model_points :: Int64 = -1; # is probably reset in the algorithm
    use_max_points :: Bool = false;

    sampling_algorithm :: Symbol = :orthogonal # :orthogonal or :monte_carlo

    constrained :: Bool = false;    # restrict sampling of new sites

    max_evals :: Int64 = typemax(Int64);

    @assert sampling_algorithm ∈ [:orthogonal, :monte_carlo] "Sampling algorithm must be either `:orthogonal` or `:monte_carlo`."
    @assert kernel ∈ Symbol.(["exp", "multiquadric", "cubic", "thin_plate_spline"]) "Kernel '$kernel' not supported yet."
    @assert kernel != :thin_plate_spline || shape_parameter isa Int && shape_parameter >= 1
    #@assert θ_enlarge_1 >=1 && θ_enlarge_2 >=1 "θ's must be >= 1."
end

# meta data object to be used during *sophisticated* sampling
@with_kw mutable struct RbfMeta <: SurrogateMeta
    center_index :: Int = 1;
    round1_indices :: Vector{Int} = [];
    round2_indices :: Vector{Int} = [];
    round3_indices :: Vector{Int} = [];
    round4_indices :: Vector{Int} = [];
    fully_linear :: Bool = false;   
end

max_evals( cfg :: RbfConfig ) :: Int = cfg.max_evals;
function max_evals!( cfg :: RbfConfig, N :: Int) :: Nothing
    cfg.max_evals = N;
    nothing
end

fully_linear( rbf :: RbfModel ) :: Bool = rbf.fully_linear;

combinable( cfg :: RbfConfig ) :: Bool = true;

combine(cfg1 :: RbfConfig, :: RbfConfig) :: RbfConfig = cfg1;

eval_models( rbf :: RbfModel, x̂ :: RVec ) = RBF.output( rbf.model, x̂ );
eval_models( rbf :: RbfModel, x̂ :: RVec, ℓ :: Int ) = RBF.output( rbf.model, ℓ, x̂ );
get_gradient( rbf :: RbfModel, x̂ :: RVec, ℓ :: Int  ) = RBF.grad( rbf.model, ℓ, x̂ );
get_jacobian( rbf :: RbfModel, x̂ :: RVec) = RBF.jac( rbf.model, x̂ );

function _init_model( cfg ::RbfConfig, objf:: AbstractObjective, mop:: AbstractMOP, 
    id :: AbstractIterData ) :: Tuple{ RbfModel, RbfMeta }
    inner_model = RBF.RBFModel(;
        n_in = num_vars( mop ),
        kernel = cfg.kernel,
        polynomial_degree = cfg.polynomial_degree,     
    );
    rbf = RbfModel 
end

function update_model( rbf :: RbfModel, objf :: AbstractObjective, rmeta:: RbfMeta, mop :: AbstractMOP, 
    id :: AbstractIterData) :: Tuple{ RbfModel, RbfMeta }
    rbf, rmeta 
end

# optional helper
num_outputs( rbf :: RbfModel ) :: Int = Rbf.num_outputs( rbf.model );

function parse_shape_param_string( Δ, expr_str)
    ex = Meta.parse(expr_str)
    return @eval begin
        let Δ=$Δ
            $ex
        end
    end 
end


## Construction

#%%
using LinearAlgebra: qr, I, norm

function _orthogonal_complement_matrix( Y :: RMat )
    Q, _ = qr(Y);
    Z = Q[:, size(Y,2) + 1 : end];
    if size(Z,2) > 0
        Z ./= norm.( eachcol(Z), Inf )';
    end
    return Z
end 

function affinely_independent_points( list_of_points :: RVecArr, x₀ :: RVec, 
    Y :: RMat = Matrix{Real}(undef, 0, 0), Z :: RMat = Matrix{Real}(undef, 0, 0),
    θ_pivot :: Real = 1e-3 ) :: Tuple{Vector{<:Int},RMat,RMat}
    
    n_vars = length(x₀);

    if isempty( list_of_points )
        return Int[], Matrix{Real}(undef, n_vars, 0), Matrix{Int}(I(n_vars))
    else
        num_points = length(list_of_points);

        rY, cY = size(Y);
        if rY == 0
            Y = Matrix{Real}(undef, n_vars, 0)
            Z = Matrix{Int}(I(n_vars));
        end

        # TODO remove this
        @assert size(Y,1) == n_vars;
        @assert size(Y,2) + size(Z,2) == n_vars;
        @assert all( isapprox.(norm.(eachcol(Z), Inf),1 ) )

        num_missing = n_vars - size(Y,2);

        accepted_indices = Int[];
        candidate_indices = collect(eachindex(list_of_points));
        for i = 1 : num_points
            # mapping to translate a vector χ and project it 
            # onto the orthogonal complement of Y, spanned by columns of Z
            proj =  χ -> norm(Z*(Z'*(χ.-x₀)),Inf)
            # find maximizer of projections 
            # NOTE: this inner loop is not mandatory
            best_val, best_index = findmax( proj.( list_of_points[ candidate_indices ] ) );

            if best_val > θ_pivot
                true_index = candidate_indices[best_index];
                push!(accepted_indices, true_index );
                deleteat!(candidate_indices, best_index)
                Y = hcat( Y, list_of_points[ true_index ] .- x₀ );
                Z = _orthogonal_complement_matrix(Y);
            end
            if length(accepted_indices) == num_missing 
                break;
            end
        end# outer for 
        return accepted_indices, Y, Z
    end# if list_of_points empty   
end
