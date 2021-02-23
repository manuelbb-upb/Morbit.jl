module Morbit

# steepest descent
using LinearAlgebra: norm
import JuMP;
import OSQP;

using Parameters: @with_kw, @unpack, @pack!
using MathOptInterface;
const MOI = MathOptInterface;
using Memoize: @memoize, memoize_cache;
import UUIDs;

import FiniteDiff#erences
const FD = FiniteDiff#erences

import ForwardDiff
const AD = ForwardDiff

include("shorthands.jl");

include("Interfaces.jl");

include("diff_wrappers.jl");

# implementations
include("VectorObjectiveFunction.jl");
include("MixedMOP.jl");

include("Surrogates.jl");

include("objectives.jl");

############################################

Broadcast.broadcastable( db :: AbstractDB ) = Ref( db );
Broadcast.broadcastable( id :: AbstractIterData ) = Ref( id );

############################################
# AbstractDB
Base.length( db :: AbstractDB ) = length(sites(db)) :: Int;

init_db( :: Type{<:AbstractDB} ) = nothing :: AbstractDB;
init_db( :: Nothing ) = nothing :: Nothing;

# array of all evaluation sites
sites( :: AbstractDB ) = RVec[] :: RVecArr;
values( :: AbstractDB ) = RVec[] :: RVecArr;

# index of current iterate 
function xᵗ_index( ::AbstractDB ) :: Int
   -1
end

xᵗ_index!(::AbstractDB, ::Int) :: Nothing = nothing;

add_site!( :: AbstractDB, :: RVec ) :: Int = -1;
add_value!( :: AbstractDB, :: RVec ) :: Int = -1;

Base.eachindex( db :: AbstractDB ) = eachindex( sites(db) );

change_site!( :: AbstractDB, :: Int, :: RVec) = nothing :: Nothing;
change_value!( :: AbstractDB, :: Int, :: RVec) = nothing :: Nothing;

stamp_x_index!( :: AbstractDB )::Nothing = nothing;

# DERIVED AND DEFAULTS
function stamp!(db::AbstractDB) ::Nothing 
    stamp_x_index!(db)
end

function get_site( db :: AbstractDB, pos :: Int) :: RVec
    let sites = sites(db);
        isempty(sites) ? Real[] : sites[pos]
    end
end
function get_value( db :: AbstractDB, pos :: Int) :: RVec
    let values = values(db);
        isempty(values) ? Real[] : values[pos]
    end
end

function get_result( db :: AbstractDB, pos :: Int ) :: Tuple{RVec, RVec}
    return ( get_site(db,pos), get_value(db,pos) )
end

function add_result!( db :: AbstractDB, x̂ :: RVec, ŷ :: RVec ) :: Int 
    site_id = add_site!(db, x̂);
    val_id = add_value!(db, ŷ);
    @assert site_id == val_id;
    return site_id;
end

"Add the next iterate `x̂` and its values `ŷ` to the database."
function add_iter_result!( db :: AbstractDB, x̂ :: RVec, ŷ :: RVec ) :: Int 
    pos = add_result!(db, x̂, ŷ);
    xᵗ_index!(db, pos);
    return pos; 
end

function change_result!( db :: AbstractDB, pos :: Int, 
    new_site :: RVec, new_value :: RVec) :: Nothing 
    change_site!(db, pos, new_site);
    change_value!(db, pos, new_value);
end

function find_result( db :: AbstractDB, x :: RVec, y :: RVec ) :: Int
    for pos ∈ eachindex(db)
        if get_site(db, pos) ≈ x 
            if get_value(db, pos) ≈ y
                change_result!(db, pos, x, y)
                return pos 
            end
        end
    end
    return -1
end

function contains_result( db :: AbstractDB, x :: RVec, y :: RVec ) :: Bool 
    return find_result(db, x, y) >= 1
end

function ensure_contains_result!( db :: AbstractDB, x :: RVec, y :: RVec ) :: Nothing 
    x_pos = find_result(db, x, y);
    if x_pos < 1
        x_pos = add_result!(db,x,y)
    end
    xᵗ_index!(db, x_pos)
    nothing 
end

function scale!( db :: AbstractDB, mop :: AbstractMOP ) :: Nothing
    for pos ∈ eachindex(db)
        x = get_site( db, pos);
        change_site!( db, pos, scale(x, mop) )
    end
    nothing
end

function unscale!( db :: AbstractDB, mop :: AbstractMOP ) :: Nothing
    for pos ∈ eachindex(db)
        x̂ = get_site( db, pos);
        change_site!( db, pos, unscale(x̂, mop) )
    end
    nothing
end

function apply_internal_sorting!( db :: AbstractDB, mop :: AbstractMOP ) :: Nothing
    for pos ∈ eachindex(db)
        y = get_value( db, pos );
        change_value!( db, pos, apply_internal_sorting( y, mop ) );
    end
    nothing 
end

function reverse_internal_sorting!( db :: AbstractDB, mop :: AbstractMOP ) :: Nothing 
    for pos ∈ eachindex(db)
        @show ŷ = get_value(db, pos)
        change_value!( db, pos, reverse_internal_sorting( ŷ, mop) )
    end
    nothing  
end

############################################
# AbstractIterData

# current iterate and values
xᵗ( :: AbstractIterData ) = nothing :: RVec;
fxᵗ( :: AbstractIterData ) = nothing :: RVec;
# trust region radius
Δᵗ( :: AbstractIterData ) = nothing :: Union{Real, RVec};

db( :: AbstractIterData ) = nothing :: Union{AbstractDB,Nothing};

# setters
function xᵗ!( id :: AbstractIterData, x̂ :: RVec ) :: Nothing 
    nothing
end

function fxᵗ!( id :: AbstractIterData, ŷ :: RVec ) :: Nothing 
    nothing
end

function Δᵗ!( id :: AbstractIterData, Δ :: Union{Real, RVec} ) :: Nothing
    nothing
end

# generic initializer
init_iter_data( ::Type{<:AbstractIterData}, x :: RVec, fx :: RVec, Δ :: Union{Real, RVec}, 
    db :: Union{AbstractDB,Nothing}) = nothing :: AbstractIterData;

function set_next_iterate!( id :: AbstractIterData, x̂ :: RVec, 
    ŷ :: RVec, Δ :: Union{Real, RVec} ) :: Nothing
    xᵗ!(id, x̂);
    fxᵗ!(id, ŷ);
    Δᵗ!(id, Δ);

    add_iter_result!(db(id), x̂, ŷ)
    nothing 
end

function keep_current_iterate!( id :: AbstractIterData, x̂ :: RVec, ŷ :: RVec,
      Δ :: Union{Real, RVec}) :: Nothing
    Δᵗ!(id, Δ);
    add_result!(db(id), x̂, ŷ );
    nothing
end

############################################
# Implementations

####### MockDB
struct NoDB <: AbstractDB end 
init_db( :: Type{NoDB} ) = NoDB();

####### ArrayDB
@with_kw mutable struct ArrayDB <: AbstractDB
    sites :: RVecArr = RVec[];   # could also use a Dict here …
    values :: RVecArr = RVec[]; # … but what about equal sites; also `push!` is easy
    x_index :: Int = 0;

    iter_indices :: Int = [];
end

init_db( :: Type{ArrayDB} ) = ArrayDB();
sites( db :: ArrayDB ) = db.sites :: RVecArr;
values( db :: ArrayDB ) = db.values :: RVecArr;
xᵗ_index( db :: ArrayDB ) = db.x_index :: Int;

function stamp_x_index!(db :: ArrayDB)::Nothing
    push!(db.iter_indices,x_index)
    nothing
end

function xᵗ_index!( db :: ArrayDB, N :: Int ) :: Nothing
    db.x_index = N
    nothing
end

function add_site!( db :: ArrayDB, x̂ :: RVec ) :: Int
    push!( db.sites, x̂ )
    return length( sites(db) )
end

function add_value!( db :: ArrayDB, ŷ :: RVec ) :: Int
    push!( db.values, ŷ )
    return length( sites(db) )
end

function change_site!( db :: ArrayDB, pos :: Int, new_site :: RVec) :: Nothing
    db.sites[pos][:] = new_site;
    nothing 
end

function change_value!( db :: ArrayDB, pos :: Int, new_val :: RVec) :: Nothing
    db.values[pos][:] = new_val;
    nothing 
end

####### IterData
@with_kw mutable struct IterData <: AbstractIterData
    sites :: RVecArr = RVec[];
    values :: RVecArr = RVec[];
    x :: Union{Nothing,RVec} = nothing;
    fx :: Union{Nothing, RVec} = nothing;
    Δ :: Union{Nothing,Real} = nothing;
    db :: Union{Nothing,AbstractDB} = nothing;    
end

xᵗ( id :: IterData ) = id.x :: Union{RVec, Nothing};
fxᵗ( id :: IterData ) = id.fx :: Union{RVec, Nothing};
Δᵗ( id :: IterData ) = id.Δ :: Union{Real, Nothing};
db( id :: IterData ) = id.db :: Union{AbstractDB, Nothing};

# setters
function xᵗ!( id :: IterData, x̂ :: RVec ) :: Nothing 
    id.x = x̂;
    nothing
end

function fxᵗ!( id :: IterData, ŷ :: RVec ) :: Nothing 
    id.fx = ŷ;
    nothing
end

function Δᵗ!( id :: IterData, Δ :: Union{Real, RVec} ) :: Nothing
    id.Δ = Δ;
    nothing
end

function init_iter_data( ::Type{IterData}, x :: RVec, fx :: RVec, Δ :: Union{Real, RVec}, 
    db :: Union{AbstractDB,Nothing}) :: IterData
    IterData(; x = x, fx = fx, Δ = Δ, db = db);
end

include("ConfigImplementations.jl")
include("descent.jl")

function _budget_okay( mop :: AbstractMOP, ac :: AbstractConfig ) :: Bool
    for objf ∈ list_of_objectives(mop)
        if num_evals(objf) >= min( max_evals(objf), max_evals(ac) ) - 1
            return false;
        end
    end
    return true
end

"True if stepsize or radius too small."
function _rel_tol_test_decision_space( Δ :: Union{Real,RVec}, steplength :: Real, ac :: AbstractConfig) :: Bool 
    return all(Δ .<= Δₗ(ac)) || all( Δ .<= Δ_crit(ac) ) && all( steplength .<= stepsize_crit(ac) );
end

function shrink_radius( ac :: AbstractConfig, Δ :: Real, steplength :: Real) :: Real
    if radius_update_method(ac) == :standard
        return Δ * γ_shrink(ac);
    elseif radius_update_method(ac) == :steplength
        return steplength * γ_shrink(ac);
    end
end
function shrink_radius_much( ac :: AbstractConfig, Δ :: Real, steplength :: Real) :: Real
    if radius_update_method(ac) == :standard
        return Δ * γ_shrink_much(ac);
    elseif radius_update_method(ac) == :steplength
        return steplength * γ_shrink_much(ac);
    end
end
function grow_radius( ac :: AbstractConfig, Δ :: Real, steplength :: Real) :: Real
    if radius_update_method(ac) == :standard
        return min( Δᵘ(ac), γ_grow(ac) * Δ )
    elseif radius_update_method(ac) == :steplength
        return min( Δᵘ(ac), ( γ_grow + steplength/Δ ) * Δ );
    end
end

using Printf: @sprintf
function _prettify( vec :: RVec, len :: Int = 5) :: AbstractString
    return string(
        "[",
        join( 
            [@sprintf("%.5f",vec[i]) for i = 1 : min(len, length(vec))], 
            ", "
        ),
        length(vec) > len ? ", …" : "",
        "]"
    )
end

############################################
function optimize( mop :: AbstractMOP, x⁰ :: RVec, 
    fx⁰ :: RVec = Real[]; algo_config :: AbstractConfig = EmptyConfig(), 
    populated_db :: Union{AbstractDB,Nothing} = nothing )

    # parse fix configuration parameters
    ν_succ = ν_success( algo_config );
    ν_acc = ν_accept( algo_config );
    mu = μ( algo_config );
    beta = β( algo_config );
    eps_crit = ε_crit( algo_config );
    gamma_crit = γ_crit( algo_config );

    # TODO warn here 
    reset_evals!( mop );

    # initialize first iteration site
    @assert !isempty( x⁰ );
    x = scale( x⁰, mop );
    
    # initalize first objective vector 
    if isempty( fx⁰ )
        # if no starting function value was provided, eval objectives
        @show fx = eval_all_objectives( mop, x );
    else
        fx = apply_internal_sorting( y, mop );
    end

    # initiliza database
    if !isnothing(populated_db)
        # has a database been provided? if yes, prepare
        data_base = populated_db;
        scale!( data_base, mop );
        apply_internal_sorting( data_base, mop );
    else
        data_base = init_db(use_db(algo_config));
    end
    ensure_contains_result!(data_base, x, fx);
    stamp!(data_base);
    @assert data_base isa AbstractDB;

    iter_data = init_iter_data(IterData, x, fx, Δ⁰(algo_config), data_base);

    # initialize surrogate models
    sc = init_surrogates( mop );
    sc.surrogates

    IMPROVEMENT_STEP_FLAG = false;
    n_iterations = 0
    MAX_ITER = max_iter(algo_config)
    steplength = Inf;   # set here for initial stopping test
    while n_iterations < MAX_ITER
        # read iter data to handy variables
        x = xᵗ(iter_data);
        fx = fxᵗ(iter_data);
        Δ = Δᵗ(iter_data);
 
        # check other stopping conditions (could also be done in head of while-loop,
        # but looks a bit more tidy here
        if !_budget_okay(mop, algo_config)
            @info "Stopping. Computational budget is exhausted."
            break;
        end
        
        # relative stopping (decision space)
        if _rel_tol_test_decision_space( Δ, steplength, algo_config)
            @info("""\n
                Stopping. Radius or stepsize too small.
                Δ = $(Δ), stepsize = $(steplength).
                Δ_min = $(Δₗ(algo_config)), Δ_crit = $(Δ_crit(algo_config)).
                stepsize_crit = $(stepsize_crit).
            """
            );
            break;
        end

        # set iteration counter
        if !IMPROVEMENT_STEP_FLAG || count_nonlinear_iterations(algo_config) 
            n_iterations += 1
        end

        @info("""\n
            |--------------------------------------------
            |Iteration $(n_iterations).
            |--------------------------------------------
            |  Current trust region radius is $(Δ).
            |  Current number of function evals is $(num_evals(mop)).
            |  Iterate is $(_prettify(x))
            |  Values are $(_prettify(fx))
            |--------------------------------------------
        """);

        # update surrogate models
        if n_iterations > 1
            if IMPROVEMENT_STEP_FLAG 
                improve_surrogates!( sc, mop, iter_data; ensure_fully_linear = false );
            else
                update_surrogates!( sc, mop, iter_data; ensure_fully_linear = false );
            end
        end

        # calculate descent step and criticality value
        ω, x₊, mx₊, steplength = compute_descent_step( algo_config, mop, iter_data, sc )
        @info "Criticality is ω = $(ω)."

        # Criticallity test
        _fully_linear = fully_linear(sc)
        if ω <= eps_crit && (!_fully_linear || all(Δ .> mu * ω))
            @info "Entered Criticallity Test."
            if !_fully_linear
                @info "Ensuring all models to be fully linear."
                update_surrogates!( sc, mop, iter_data; ensure_fully_linear = true );
                
                ω, x₊, mx₊, steplength = compute_descent_step(algo_config,mop,iter_data,sc);
                if !fully_linear(sc)
                    @info "Could not make all models fully linear. Trying one last descent step."
                    @goto MAIN;
                end
            end
            num_critical_loops = 0;
            
            while all(Δᵗ(iter_data) .> mu * ω)
                @info "Criticality loop $(num_critical_loops + 1)." 
                if num_critical_loops >= max_critical_loops(algo_config)
                    @info "Maximum number ($(max_critical_loops(algo_config))) of critical loops reached. Exiting..."
                    @goto EXIT_MAIN
                end
                if !_budget_okay(mop, algo_config)
                    @info "Computational budget exhausted. Exiting…"
                    @goto EXIT_MAIN
                end
                
                # shrink radius
                Δᵗ!( iter_data, Δᵗ(iter_data) .* gamma_crit );
                # make model linear 
                update_surrogates!( sc, mop, iter_data; ensure_fully_linear = true );
                # (re)calculate criticality
                # TODO make backtracking optional and don't do here
                ω, x₊, mx₊, steplength = compute_descent_step(algo_config,mop,iter_data,sc);

                if _rel_tol_test_decision_space( Δᵗ(iter_data), steplength, algo_config)
                    @info "Radius or stepsize too small. Exiting…"
                    @goto EXIT_MAIN 
                end

                if !fully_linear(sc)
                    @info "Could not make all models fully linear. Trying one last descent step."
                    @goto MAIN;
                end

                num_critical_loops += 1;
            end
            "Exiting after $(num_critical_loops) loops with ω = $(ω) and Δ = $(Δᵗ(iter_data))."
            @goto MAIN
            @label EXIT_MAIN 
            break;
        end# Crit test if 

        @label MAIN # re-entry point after criticality test 

        mx = eval_models(sc, x);
        fx₊ = eval_all_objectives(mop, x₊);
        
        if strict_acceptance_test( algo_config )
            ρ = minimum( (fx .- fx₊) ./ (mx .- mx₊) )
        else
            ρ = (maximum(fx) - maximum( fx₊ ))/(maximum(mx)-maximum(mx₊))
        end
        
        @info """\n
        Attempting descent of length $steplength.
        | f(x)  | $(_prettify(fx))
        | f(x₊) | $(_prettify(fx₊))
        | m(x)  | $(_prettify(mx))
        | m(x₊) | $(_prettify(mx₊))
        The error betwenn f(x) and m(x) is $(sum(abs.(fx .- mx))).
        $(strict_acceptance_test(algo_config) ? "All" : "One") of the components must decrease.
        Thus, ρ is $ρ.
        """
        @assert all( mx .>= mx₊ )

        ACCEPT_TRIAL_POINT = false
        ρ = isnan(ρ) ? -Inf : ρ;
        Δ = Δᵗ(iter_data);  # if it was changed in criticality test
        if ρ >= ν_succ
            if Δ < beta * ω
                new_Δ = grow_radius(algo_config, Δ, steplength);
            else
                new_Δ = Δ;
            end
            ACCEPT_TRIAL_POINT = true;
        else
            if fully_linear(sc)
                if ρ < ν_acc
                    new_Δ = shrink_radius_much(algo_config, Δ, steplength);
                else
                    new_Δ = shrink_radius(algo_config, Δ, steplength);
                    ACCEPT_TRIAL_POINT = true;
                end
            else
                IMPROVEMENT_STEP_FLAG = true;
                new_Δ = Δ
            end
        end

        if ACCEPT_TRIAL_POINT
            set_next_iterate!(iter_data, x₊, fx₊, new_Δ);
        else
            keep_current_iterate!(iter_data, x₊, fx₊, new_Δ);
        end
        @info """\n
            The step is $(ACCEPT_TRIAL_POINT ? (ρ >= ν_succ ? "very sucessfull!" : "acceptable.") : "unsucessfull…")
            Moreover, the radius was updated as below:
            old radius : $Δ
            new radius : $new_Δ ($(round(new_Δ/Δ * 100;digits=1)) %)
        """

        stamp!(data_base)
        @assert all(isapprox.(x₊,xᵗ(iter_data)))
    end

    ret_x = unscale(xᵗ(iter_data),mop);
    ret_fx = reverse_internal_sorting(fxᵗ(iter_data),mop);
    @info("""\n
        |--------------------------------------------
        | FINISHED
        |--------------------------------------------
        | No. iterations:  $(n_iterations) 
        | No. evaluations: $(num_evals(mop))
        | final unscaled vectors:
        | iterate: $(_prettify(ret_x, 10))
        | value:   $(_prettify(ret_fx, 10))
    """);

    # unscale sites and re-sort values to return to user
    if !isnothing( db(iter_data) )
        unscale!( db(iter_data) , mop);
        reverse_internal_sorting!( db(iter_data), mop);
    end

    return ret_x, ret_fx, iter_data

end# function optimize

end#module