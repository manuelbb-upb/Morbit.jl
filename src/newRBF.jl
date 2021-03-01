include("RBFBase.jl");
import .RBF
using LinearAlgebra: qr, I, norm


@with_kw mutable struct RbfModel <: SurrogateModel
    model :: Union{Nothing,RBF.RBFModel} = nothing;

    # matrices for the sampling and improvement algorithm
    Y :: RMat = Matrix{Real}(undef, 0, 0);
    Z :: RMat = Matrix{Real}(undef, 0, 0);
    # matrices for the sampling and improvement algorithm
    Y₂ :: RMat = Matrix{Real}(undef, 0, 0);
    Z₂ :: RMat = Matrix{Real}(undef, 0, 0);
    
    fully_linear :: Bool = false;

    # for gathering training data in form of `Results`
    # dict key is the "construction round" for metadata (plotting)
    # dict values are vectors of training data    
    tdata :: Dict{Int,Vector{<:Result}} = Dict( i => Result[] for i = 1:4);
end

function RbfModel( model :: RBF.RBFModel, n_vars :: Int )
    Y = Matrix{Real}(undef, n_vars, 0);
    Z = Matrix{Int}(I(n_vars));
    return RbfModel(; 
        model = model,
        Y = Y, 
        Y₂ = Y,
        Z = Z,
        Z₂ = Z
    );
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
    round1_indices :: Vector{NothInt} = [];
    round2_indices :: Vector{NothInt} = [];
    round3_indices :: Vector{NothInt} = [];
    round4_indices :: Vector{NothInt} = [];
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
    id :: AbstractIterData, ac :: AbstractConfig ) :: Tuple{ RbfModel, RbfMeta }
    
    n_vars = num_vars(mop);

    # let `update_model(…)` do all the work
    rbf = RbfModel();
    rmeta = RbfMeta();

    return update_model( rbf, objf, rmeta, mop, id, ac; ensure_fully_linear = true)
end

function update_model( rbf :: RbfModel, objf :: AbstractObjective, rmeta:: RbfMeta, mop :: AbstractMOP,
    id :: AbstractIterData, ac :: AbstractConfig;
    ensure_fully_linear :: Bool = false ) :: Tuple{ RbfModel, RbfMeta }
    @info "UPDATE RBF"
    cfg = model_cfg(objf);
    DB = db(id);
    Δ = Δᵗ(id);
    x = xᵗ(id);
    n_vars = length(x);

    # we completly start from scratch here, nothing to recycle 
    # this way, tdata is empty and properly initialized
    inner_model = RBF.RBFModel(;
        n_in = n_vars,
        kernel = cfg.kernel,
        polynomial_degree = cfg.polynomial_degree,
        shape_parameter = _get_shape_param(cfg, id),
    );

    rbf =  RbfModel(inner_model, n_vars);

    rmeta = RbfMeta();

    # find affinely independent training points in radius Δ₁
    find_box_independent_points1!( rbf, cfg, id, mop, ac );
       
    n_missing = n_vars - length(rbf.tdata[1]);
    if n_missing > 0 && !(ensure_fully_linear || cfg.require_linear)
        # the model can be not fully linear
        # Search for more points in trust region of maximum possible radius
       find_box_independent_points2!(rbf, cfg, id, mop, ac)
    end

    n_missing -= length(rbf.tdata[2]);
    n_new = min( n_missing, min( max_evals(ac), max_evals(cfg) ) - 1 );

    add_new_sites!(rbf, cfg, mop, id; max_new = n_new);

    if length(rbf.tdata[3]) < n_new 
        return rebuild_model( rbf, objf, rmeta, id, mop, ac);
    end# if (rebuild model) 
    
    rbf.fully_linear = isempty(rbf.tdata[2]) && length(rbf.tdata[1]) + length(rbf.tdata[3]) == n_vars
    @info "RBF Model is $(rbf.fully_linear ? "" : "not ")fully linear."

    #Evaluate resuts...
    _eval_new_sites!( rbf, rmeta, objf, mop, id );
    RBF.train!(rbf.model)

    # TODO REMOVE!!
    #=
    for (s,v) ∈ zip(rbf.model.training_sites, rbf.model.training_values)
        @show eval_models(rbf, s ) .- v
    end
    =#

    return rbf, rmeta 
end

function improve_model( rbf:: RbfModel, objf::AbstractObjective, rmeta :: RbfMeta,
    mop ::AbstractMOP, id :: AbstractIterData, ac:: AbstractConfig;
    ensure_fully_linear :: Bool = false ) :: Tuple{RbfModel, RbfMeta}
    @show size(rbf.Z)
    if size(rbf.Z, 2) > 0
        @info "Performing an improvement step..."

        x = xᵗ(id);
        Δ = Δᵗ(id);
        
        dir = copy( Z[:,1] );
        Δ₁ = cfg.θ_enlarge_1 * Δ;
        pivot = cfg.pivot * Δ;
        
        if abs(len) > pivot
            len = intersect_bounds( mop, x, Δ₁, dir; return_vals = :absmax)
            offset = len .* dir;
            push!(rbf.tdata[3], init_res( Res, x .+ offset));
        
            rbf.Y = hcat( rbf.Y, offset );
            rbf.Z = rbf.Z[:, 2:end];

            rbf.fully_linear = size(rbf.Z, 2) == 0;
            _eval_new_sites!(rbf, rmeta, objf, mop, id);
            RBF.train!(rbf.model);        
        else
            return rebuild_model( rbf, objf, rmeta, mop, id, ac);
        end        
    end
    return rbf, rmeta;
end

function rebuild_model( rbf :: RbfConfig, objf :: AbstractObjective, rmeta :: RbfMeta, 
        mop :: AbstractMOP, id :: AbstractIterData, ac :: AbstractConfig ) :: Tuple{RbfModel, RbfMeta}
    @info "Rebuild model along coordinates..."

    x = xᵗ(id);
    n_vars = length(x);
    Δ = Δᵗ(id);
    Δ₁ = cfg.θ_enlarge_1 * Δ;

    max_new = min(min( max_evals(ac), max_evals(cfg) ) - 1, n_vars);
    rbf = RbfModel(;
        model = rbf.model,
        Y = Matrix{Real}(undef, n_vars, 0),
        Z = Matrix{Int}(I( n_vars )),
    );

    for i = 1 : max_new
        dir = zeros(Int, n_vars);
        dir[i] = 1;
        len = intersect_bounds( mop, x, Δ₁, dir; return_vals = :absmax)
        offset = len .* dir;
        push!(rbf.tdata[3], init_res( Res, x .+ offset));
        rbf.Y = hcat( rbf.Y, offset );
        rbf.Z = rbf.Z[:, 2:end];
        if abs(len) <= cfg.θ_pivot * Δ 
            @warn("Cannot make model fully linear!")
        end
    end#for 

    _eval_new_sites!(rbf, rmeta, objf, mop, id );
    RBF.train!(rbf.model);
    return rbf, rmeta
end


# optional helper
num_outputs( rbf :: RbfModel ) :: Int = Rbf.num_outputs( rbf.model );


# evaluate at new sites
# set `training_sites` and `training_values` for inner model
# set meta data
function _eval_new_sites!( rbf :: RbfModel, rmeta :: RbfMeta, objf :: AbstractObjective,
    mop :: AbstractMOP, id :: AbstractIterData )  :: Nothing

    basis_results = vcat( (rbf.tdata[i] for i = 1:4 )... )
    _eval_and_store_new_results!(id, basis_results, mop);

    NotXint = Union{Nothing,Int};
    rmeta.center_index = convert( NotXint, xᵗ_index(id) );
    rmeta.round1_indices = [ convert( NotXint, get_id( res ) ) for res ∈ rbf.tdata[1] ] ;
    rmeta.round2_indices = [ convert( NotXint, get_id( res ) ) for res ∈ rbf.tdata[2] ] ;
    rmeta.round3_indices = [ convert( NotXint, get_id( res ) ) for res ∈ rbf.tdata[3] ] ;
    rmeta.round4_indices = [ convert( NotXint, get_id( res ) ) for res ∈ rbf.tdata[4] ] ;

    # "transfer" training data to inner model 
    empty!(rbf.model.training_sites)
    push!(rbf.model.training_sites, xᵗ(id));
    push!(rbf.model.training_sites, [ get_site(res) for res ∈ basis_results ]... ) ;
    oi = output_indices( objf, mop )
    empty!(rbf.model.training_values)
    push!(rbf.model.training_values, fxᵗ(id)[oi]);
    push!(rbf.model.training_values, [ get_value(res)[oi] for res ∈ basis_results ]... );

    nothing 
end

function parse_shape_param_string( Δ, expr_str)
    ex = Meta.parse(expr_str)
    return @eval begin
        let Δ=$Δ
            $ex
        end
    end 
end

"Get real-valued shape parameter for RBF model from current iter data.
`cfg` allows for a string expression which would be evaluated here."
function _get_shape_param( cfg :: RbfConfig, id :: AbstractIterData ) :: Real
    if isa(cfg.shape_parameter, AbstractString)
        return parse_shape_param_string( Δᵗ(id), cfg.shape_parameter )
    elseif isa(cfg.shape_parameter, Real)
        return cfg.shape_parameter
    end
end


## Construction

function _orthogonal_complement_matrix( Y :: RMat )
    Q, _ = qr(Y);
    Z = Q[:, size(Y,2) + 1 : end];
    if size(Z,2) > 0
        Z ./= norm.( eachcol(Z), Inf )';
    end
    return Z
end 

"""
Find affinely independent results in database box of radius `Δ` around `x`
Results are saved in `rbf.tdata[tdata_index]`. 
Both `rbf.Y` and `rbf.Z` are changed.
"""
function find_box_independent_points1!( rbf :: RbfModel, cfg :: RbfConfig, id :: AbstractIterData,
    mop :: AbstractMOP, :: AbstractConfig ) :: Nothing

    res, Y, Z = _find_box_independent_points( 
        rbf, id, mop, xᵗ(id), Δᵗ(id) * cfg.θ_enlarge_1; 
        θ_pivot = cfg.θ_pivot * Δᵗ(id) 
    );

    rbf.tdata[1] = res;
    rbf.Y = Y;
    rbf.Z = Z;
    nothing 
end

function find_box_independent_points2!( rbf :: RbfModel, cfg :: RbfConfig, id :: AbstractIterData,
    mop :: AbstractMOP, ac :: AbstractConfig )

    Δ₂ = Δᵘ(ac) * cfg.θ_enlarge_1;

    res, Y, Z = _find_box_independent_points( 
        rbf, id, mop, xᵗ(id), Δ₂; 
        θ_pivot = (Δᵗ(id) * cfg.θ_enlarge_1 / Δ₂) * cfg.θ_pivot,
        exclude_indices = [ xᵗ_index(id); get_id.( rbf.tdata[1] ) ]
    );

    rbf.tdata[2] = res;
    rbf.Y₂ = Y;
    rbf.Z₂ = Z;
    nothing 
end

# if we had more than two rounds we would have to pass Y & Z here too
function _find_box_independent_points( rbf :: RbfModel, 
    id :: AbstractIterData, mop :: AbstractMOP, x :: RVec, Δ :: Union{Real,RVec};
    θ_pivot = 1e-3, exclude_indices :: Vector{<:NothInt} = NothInt[] ) :: Tuple{Vector{<:Result},RMat,RMat}

    lb, ub = local_bounds(mop, x, Δ );
    box_indices = find_points_in_box( id, lb, ub; exclude_indices );
    
    box_results = get_result.( id, box_indices );

    affinely_independent_box_indices, Y, Z = _affinely_independent_points( 
        box_results, 
        x, 
        rbf.Y, 
        rbf.Z, 
        θ_pivot = θ_pivot
    )

    @info("\tFound $(length(affinely_independent_box_indices)) points in box of radius $(Δ) with pivot $(θ_pivot).");
    return (box_results[affinely_independent_box_indices], Y, Z);
end
  

function _affinely_independent_points( list_of_results :: Vector{<:Result}, x₀ :: RVec, 
    Y :: RMat = Matrix{Real}(undef, 0, 0), Z :: RMat = Matrix{Real}(undef, 0, 0);
    θ_pivot :: Real = 1e-3 ) :: Tuple{Vector{<:Int},RMat,RMat}
    return _affinely_independent_points( get_site.(list_of_results), x₀, Y, Z; θ_pivot )
end
    
 
function _affinely_independent_points( list_of_points :: RVecArr, x₀ :: RVec, 
    Y :: RMat = Matrix{Real}(undef, 0, 0), Z :: RMat = Matrix{Real}(undef, 0, 0);
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

function add_new_sites!(rbf :: RbfModel, cfg :: RbfConfig, mop :: AbstractMOP, 
    id :: AbstractIterData; max_new :: Int = typemax(Int) ) :: Nothing 
    return add_new_sites!(rbf, cfg, mop, id, Val( cfg.sampling_algorithm ); max_new )
end

function add_new_sites!(rbf :: RbfModel, cfg :: RbfConfig, mop :: AbstractMOP,
    id :: AbstractIterData, ::Val{:orthogonal}; max_new :: Int ) :: Nothing
    x = xᵗ(id);
    Δ = Δᵗ(id);
    n_vars = length(x);
    
    Δ₁ = Δ * cfg.θ_enlarge_1;
    for dir = eachcol( rbf.Z )
        @assert norm(dir, Inf) ≈ 1;
        len = intersect_bounds( mop, x, Δ₁, copy(dir); return_vals = :absmax)
        if abs(len) > cfg.θ_pivot * Δ 
            # pivot big enough, accept new site
            offset = len .* dir;
            push!(rbf.tdata[3], init_res( Res, x .+ offset));
            rbf.Y = hcat( rbf.Y, offset );
            rbf.Z = rbf.Z[:, 2:end];
        else
            # could not sample far enough from current iterate, break 
            break;
        end
        if length(rbf.tdata[3]) >= max_new 
            break;
        end
    end
    return nothing;    
end

function add_old_sites!(rbf :: RbfModel, cfg :: RbfConfig, id :: AbstractIterData;
    max_new :: Int ) :: Nothing
    return add_old_sites!(rbf,cfg,id,Val( cfg.sampling_algorithm); max_new )
end
function add_old_sites!(rbf :: RbfModel, cfg :: RbfConfig, id :: AbstractIterData, 
    ::Val{:orthogonal}; max_new :: Int) :: Nothing
    nothing
end