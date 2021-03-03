include("RBFBase.jl");
import .RBF
using LinearAlgebra: qr, I, norm, givens, Hermitian, cholesky, inv
using LinearAlgebra: diag # remove

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
    tdata :: Dict{Int,Vector{<:Result}} = Dict( i => Result[] for i = 0:4);
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

tdata_length( rbf :: RbfModel ) :: Int = sum( length.(Base.values(rbf.tdata) ) )
db_indices( rbf :: RbfModel, i :: Int ) = convert( Vector{NothInt}, get_id.( rbf.tdata[i]) ) # WHY DO I HAVE TO CONVERT???
db_indices( rbf :: RbfModel ) = vcat( [ db_indices(rbf, i) for i ∈ Base.keys(rbf.tdata) ]... );
tdata_results(rbf :: RbfModel ) = vcat( (rbf.tdata[i] for i ∈ keys(rbf.tdata) )... );

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
    sampling_algorithm2 :: Symbol = :standard_rand

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
    
    @logmsg loglevel3 "Updating RBF with indices $(output_indices(objf, mop))."
    cfg = model_cfg(objf);
    DB = db(id);
    Δ = Δᵗ(id);
    x = xᵗ(id);
    n_vars = length(x);
    
    MAX_EVALS = min( max_evals(ac), max_evals(cfg) ) - 1 ;
    # we completly start from scratch here, nothing to recycle 
    # this way, tdata is empty and properly initialized
    inner_model = RBF.RBFModel(;
        n_in = n_vars,
        kernel = cfg.kernel,
        polynomial_degree = cfg.polynomial_degree,
        shape_parameter = _get_shape_param(cfg, id),
    );

    rbf =  RbfModel(inner_model, n_vars);
    push!(rbf.tdata[0], get_result( id, xᵗ_index(id) ));

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
    n_new = max(0, min( n_missing, MAX_EVALS) );

    add_new_sites!(rbf, cfg, mop, id; max_new = n_new);

    if length(rbf.tdata[3]) < n_new 
        return rebuild_model( rbf, objf, rmeta, id, mop, ac);
    end
    
    rbf.fully_linear = isempty(rbf.tdata[2]) && length(rbf.tdata[1]) + length(rbf.tdata[3]) == n_vars
    @logmsg loglevel4 "RBF Model is $(rbf.fully_linear ? "" : "not ")fully linear."

    n_evals_left = MAX_EVALS - n_new;
    add_old_sites!( rbf, cfg, mop, id; max_new = n_evals_left);

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
    if size(rbf.Z, 2) > 0
        @logmsg loglevel3 "Performing an improvement step for RBF with indices $(output_indices(objf, mop))."
        cfg = model_cfg(objf);

        x = xᵗ(id);
        Δ = Δᵗ(id);
        
        dir = copy( rbf.Z[:,1] );
        Δ₁ = cfg.θ_enlarge_1 * Δ;
        pivot = cfg.θ_pivot * Δ;
        
        len = intersect_bounds( mop, x, Δ₁, dir; return_vals = :absmax)
        if abs(len) > pivot
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
    @logmsg loglevel3 "Rebuild model along coordinates..."

    x = xᵗ(id);
    n_vars = length(x);
    Δ = Δᵗ(id);
    Δ₁ = cfg.θ_enlarge_1 * Δ;

    MAX_EVALS = min( max_evals(ac), max_evals(cfg) ) - 1;
    max_new = min(MAX_EVALS, n_vars);

    rbf = RbfModel(;
        model = rbf.model,
        Y = Matrix{Real}(undef, n_vars, 0),
        Z = Matrix{Int}(I( n_vars )),
    );

    push!(rbf.tdata[0], get_result( id, xᵗ_index(id) ));

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
    rbf.Y₂ = rbf.Y;
    rbf.Z₂ = rbf.Z;

    n_evals_left = MAX_EVALS - max_new;
    add_old_sites!( rbf, cfg, mop, id; max_new = n_evals_left);

    _eval_new_sites!(rbf, rmeta, objf, mop, id );
    RBF.train!(rbf.model);
    return rbf, rmeta
end


# optional helper
num_outputs( rbf :: RbfModel ) :: Int = Rbf.num_outputs( rbf.model );

function _transfer_training_sites!( rbf :: RbfModel ) :: Nothing
    basis_results = tdata_results(rbf);
    empty!(rbf.model.training_sites)
    push!(rbf.model.training_sites, [ get_site(res) for res ∈ basis_results ]... ) ;
    nothing 
end

function _transfer_training_values!( rbf :: RbfModel, out_indices :: Vector{Int}) :: Nothing
    basis_results = tdata_results(rbf);
    empty!(rbf.model.training_values)
    push!(rbf.model.training_values, [ get_value(res)[out_indices] for res ∈ basis_results ]... );
    nothing
end

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
    _transfer_training_sites!(rbf);
    _transfer_training_values!(rbf, output_indices( objf, mop));  
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
        θ_pivot = cfg.θ_pivot
    );
    push!(rbf.tdata[1], res...);
    rbf.Y₂ = rbf.Y = Y;
    rbf.Z₂ = rbf.Z = Z;
    nothing 
end

function find_box_independent_points2!( rbf :: RbfModel, cfg :: RbfConfig, id :: AbstractIterData,
    mop :: AbstractMOP, ac :: AbstractConfig )

    Δ₂ = Δᵘ(ac) * cfg.θ_enlarge_1;

    res, Y, Z = _find_box_independent_points( 
        rbf, id, mop, xᵗ(id), Δ₂; 
        θ_pivot = (Δᵗ(id) * cfg.θ_enlarge_1 / Δ₂) * cfg.θ_pivot,
        exclude_indices = [ xᵗ_index(id); db_indices(rbf, 1) ]
    );

    push!(rbf.tdata[2], res...);
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

    pivot = Δ * θ_pivot;

    affinely_independent_box_indices, Y, Z = _affinely_independent_points( 
        box_results, 
        x, 
        rbf.Y, 
        rbf.Z, 
        piv_val = pivot
    )

    @logmsg loglevel4 "Found $(length(affinely_independent_box_indices)) points in box of radius $(Δ) with pivot $(pivot)."
    return (box_results[affinely_independent_box_indices], Y, Z);
end
  

function _affinely_independent_points( list_of_results :: Vector{<:Result}, x₀ :: RVec, 
    Y :: RMat = Matrix{Real}(undef, 0, 0), Z :: RMat = Matrix{Real}(undef, 0, 0);
    piv_val :: Real = 1e-3 ) :: Tuple{Vector{<:Int},RMat,RMat}
    return _affinely_independent_points( get_site.(list_of_results), x₀, Y, Z; piv_val )
end
    
 
function _affinely_independent_points( list_of_points :: RVecArr, x₀ :: RVec, 
    Ŷ :: RMat = Matrix{Real}(undef, 0, 0), Ẑ :: RMat = Matrix{Real}(undef, 0, 0);
    piv_val :: Real = 1e-3 ) :: Tuple{Vector{<:Int},RMat,RMat}
    
    n_vars = length(x₀);
    Y = copy(Ŷ);    # because we are possible modifying the matrices
    Z = copy(Ẑ);

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

            if best_val > piv_val
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
    if max_new > 0
        x = xᵗ(id);
        Δ = Δᵗ(id);
        n_vars = length(x);
        
        Δ₁ = Δ * cfg.θ_enlarge_1;
        for dir = eachcol( rbf.Z )
            if length(rbf.tdata[3]) >= max_new 
                break;
            end
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
            
        end
    end
    return nothing;    
end

function add_old_sites!(rbf :: RbfModel, cfg :: RbfConfig, mop :: AbstractMOP, id :: AbstractIterData; kwargs... ) :: Nothing
    return add_old_sites!(rbf, cfg, mop, id, Val( cfg.sampling_algorithm2); kwargs...)
end
function add_old_sites!(rbf :: RbfModel, cfg :: RbfConfig, mop :: AbstractMOP, 
    id :: AbstractIterData, ::Val{:standard_rand}; 
    max_new :: Int = typemax(Int) ):: Nothing
    x = xᵗ(id);
    Δ₂ = cfg.θ_enlarge_2 * Δᵗ(id);
    n_vars = length(x);
    
    chol_pivot = cfg.θ_pivot_cholesky;
    
    N₀ = N = tdata_length( rbf )
    MAX_POINTS = cfg.max_model_points <= 0 ? 2*n_vars+1 : cfg.max_model_points;

    if N < MAX_POINTS
        # make sure, training sites are set
        _transfer_training_sites!(rbf);

        # prepare matrices as by Wild, R has to be augmented by rows of zeros
        Φ = RBF.get_Φ( rbf.model );
        Π = RBF.get_Π( rbf.model );
        Q, R = qr( transpose(Π) );
        R = [
            Matrix(R);
            zeros( size(Q,1) - size(R,1), size(R,2) )
        ];
        Z = Q[:, RBF.min_num_sites(rbf.model) + 1 : end ];

        ZΦZ = Hermitian(Z'Φ*Z);
        L = cholesky( ZΦZ ).L     # should also be empty at this point
        Lⁱ = inv(L);

        φ₀ = Φ[1,1]; # constant value

        # TEST FOR ADDITIONAL POINTS
        # we reverse the order of database indices because in our 
        # implementation this favors recently added points, i.e.,
        # points that are near the current iterate. other heuristics 
        # are very much possible
        DB = db(id);
        all_ids = sort( collect( eachindex( DB ) ); rev = true );
        training_ids = db_indices( rbf ) 

        lb, ub = local_bounds(mop, x, Δ₂);

        for id ∈ all_ids
            if N >= MAX_POINTS 
                break;
            end
            if id ∉ training_ids 
                old_res = get_result(DB, id);
                x̂ = get_site( old_res );
                if _point_in_box( x̂, lb, ub)
                    test_retval = _test_new_cholesky_site( rbf.model, x̂, φ₀, Q, R, Z, L, Lⁱ, Φ, Π, chol_pivot );
                    if !isnothing(test_retval)
                        Q, R, Z, L, Lⁱ, Φ, Π = test_retval;
                        push!(rbf.tdata[4], old_res );
                        push!(rbf.model.training_sites, x̂ ); # so that the _test… works properly
                        N +=1 ; # increase point counter
                    end
                end
            end
        end
        @logmsg loglevel4 "Found $(N - N₀) additional sites to add to the model."
        if max_new > 0 && cfg.use_max_points && N < MAX_POINTS
            n_new = min(MAX_POINTS, max_new);
            max_tries = 10*n_new;
            n_tries = 0;
            @logmsg loglevel4 "`use_max_points` … trying to find $(n_new - N) new sites."
            while N < n_new && n_tries < max_tries;
                # simple hit-or-miss, not very effective nor good for model quality
                x̂ = _rand_box_point( lb, ub )
                test_retval = _test_new_cholesky_site( rbf.model, x̂, φ₀, Q, R, Z, L, Lⁱ, Φ, Π, chol_pivot );
                if !isnothing(test_retval)
                    Q, R, Z, L, Lⁱ, Φ, Π = test_retval;
                    push!(rbf.tdata[4], init_res(Res, x̂));
                    push!(rbf.model.training_sites, x̂ ); # so that the _test… works properly
                    N +=1 ; # increase point counter
                end
                n_tries += 1;
            end
        end
    end
    nothing
end


function _test_new_cholesky_site(m :: RBF.RBFModel, y :: RVec, φ₀ :: Real, 
    Q :: AbstractArray, R :: AbstractArray,  Z :: AbstractArray, L :: AbstractArray, 
    Lⁱ :: AbstractArray, Φ :: AbstractArray, Π :: AbstractArray,  θ_pivot_cholesky :: Real )
    
    # evaluate RBF basis at new site
    φy = RBF.φ(m, y)
    Φy = [
        [Φ φy];
        [φy' φ₀]
    ]

    # get polynomial basis column and stack transpose below R 
    πy = RBF.Π_col( m, y );
    Ry = [
        R ;
        πy'
    ]

    # perform givens rotations to turn last row in Ry to zeros
    row_index = size( Ry, 1)
    G = Matrix(I, row_index, row_index) # whole orthogonal matrix
    for j = 1 : size(R,2) 
        # in each column, take the diagonal as pivot to turn last elem to zero 
        g = givens( Ry[j,j], Ry[row_index, j], j, row_index )[1];
        Ry = g*Ry;
        G = g*G;
    end
    # now, from G we can update the other matrices 
    Gᵀ = transpose(G);
    g̃ = Gᵀ[1 : end-1, end];
    ĝ = Gᵀ[end, end];

    Qg = Q*g̃;
    v_y = Z'*( Φ*Qg + φy .* ĝ );
    σ_y = Qg'*Φ*Qg + (2*ĝ) * φy'*Qg + ĝ^2*φ₀;

    τ_y² = σ_y - norm( Lⁱ * v_y, 2 )^2 
    # τ_y (and hence τ_y^2) must be bounded away from zero 
    # for the model to remain fully linear
    if τ_y² > θ_pivot_cholesky
        τ_y = sqrt(τ_y²)

        Qy = [
            [ Q zeros( size(Q,1), 1) ];
            [ zeros(1, size(Q,2)) 1 ]
        ] * Gᵀ


        z = [
            Q * g̃;
            ĝ
        ]

        Zy = [
            [
                Z;
                zeros(1, size(Z,2))
            ] z
        ]

        Lyⁱ = [
            [Lⁱ zeros(size(Lⁱ,1),1)];
            [ -(v_y'Lⁱ'Lⁱ)./τ_y 1/τ_y ]
        ];

        Ly = [
            [ L zeros(size(L,1), 1) ];
            [ v_y'Lⁱ' τ_y ]
        ]

        Πy = [ Π πy ];

        @assert diag(Ly * Lyⁱ) ≈ ones(size(Ly,1))
    
        return Qy, Ry, Zy, Ly, Lyⁱ, Φy, Πy
    else
        return nothing
    end
end