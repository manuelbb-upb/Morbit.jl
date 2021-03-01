# This file defines the required data structures and methods for vector-valued
# Taylor models.
# For Differentiation the `diff_wrappers.jl` are needed.
# Also, abstract types SurrogateConfig, SurrogateModel, SurrogateMeta must be defined.

@with_kw mutable struct TaylorConfig <: SurrogateConfig
    degree :: Int64 = 1;

    gradients :: Union{Symbol, Nothing, Vector{<:Function}, Function } = :fdm
    hessians ::  Union{Symbol, Nothing, Vector{<:Function}, Function } = gradients

    # alternative to specifying individual gradients
    jacobian :: Union{Symbol, Nothing, Function} = nothing

    max_evals :: Int64 = typemax(Int64);
    @assert !( ( isa(gradients, Vector) && isempty( gradients ) ) && isnothing(jacobian) ) "Provide either `gradients` or `jacobian`."
    @assert !( isnothing(gradients) && isnothing(jacobian) ) "Provide either `gradients` or `jacobian`."
    @assert 1 <= degree <= 2 "Can only construct linear and quadratic polynomial Taylor models."
end

@with_kw mutable struct TaylorModel <: SurrogateModel
    degree :: Int64 = 2;
    # reference to mop to have unscaling availabe;
    mop :: AbstractMOP
    # reference to objective(s) to evaluate 
    objf :: AbstractObjective
    diff_fn :: Union{DiffFn,Nothing} = nothing
    hess_fn :: Union{HessFromGrads, DiffFn, HessWrapper, Nothing} = nothing

    # expansion point and value 
    x0 :: RVec = Real[];
    fx0 :: RVec = Real[];
    # gradient(s) at x0
    g :: RVecArr = RVec[];
    H :: Vector{<:RMat} = RMat[];
end

struct TaylorMeta <: SurrogateMeta end   # no construction meta data needed

max_evals( cfg :: TaylorConfig ) = cfg.max_evals;
function max_evals!( cfg :: TaylorConfig  ) :: Nothing 
    cfg.max_evals = 0;
    nothing 
end 

fully_linear( tm :: TaylorModel ) = true;
combinable( :: TaylorConfig ) = false;      # TODO think about this 

# Same method as for ExactModel; duplicated for tidyness...
"Modify/initialize thec exact model `mod` so that we can differentiate it later."
function set_gradients!( mod :: TaylorModel, objf :: AbstractObjective, mop :: AbstractMOP ) :: Nothing
    cfg = model_cfg(objf);
    if isa( cfg.gradients, Symbol )
        if cfg.gradients == :autodiff
            mod.diff_fn = AutoDiffWrapper( objf )
        elseif cfg.gradients == :fdm 
            mod.diff_fn = FiniteDiffWrapper( objf );
        end
    else
        if isa(cfg.gradients, Vector)
            @assert length(cfg.gradients) == num_outputs(objf) "Provide as many gradient functions as the objective has outputs."
        elseif isa(cfg.gradients, Function)
            @assert num_outputs(objf) == 1 "Only one gradient provided for $(num_outputs(objf)) outputs."
        end
        mod.diff_fn = GradWrapper(mop, cfg.gradients, cfg.jacobian )
    end
    nothing
end

function set_hessians!( mod :: TaylorModel, objf :: AbstractObjective, mop :: AbstractMOP) :: Nothing
    cfg = model_cfg(objf);
    if isa( cfg.hessians, Symbol )
        if isa( mod.diff_fn, GradWrapper )
            mod.hess_fn = HessFromGrads( mod.diff_fn, cfg.hessians );
        else 
            if cfg.hessians == :autodiff
                if isa( mod.diff_fn, AutoDiffWrapper )
                    mod.hess_fn = mod.diff_fn 
                else
                    mod.hess_fn = AutoDiffWrapper(objf);
                end
            elseif cfg.hessians == :fdm 
                if isa( mod.diff_fn, FiniteDiffWrapper)
                    mod.hess_fn = mod.diff_fn
                else
                    mod.hess_fn = FiniteDiffWrapper( objf );
                end
            end
        end
    else
        if isa(cfg.hessians, Vector)
            @assert length(cfg.hessians) == num_outputs(objf) "Provide as many hessian functions as the objective has outputs."
        elseif isa(cfg.hessians, Function)
            @assert num_outputs(objf) == 1 "Only one hessian function provided for $(num_outputs(objf)) outputs."
        end
        mod.hess_fn = HessWrapper(mop, cfg.hessians )
    end
    nothing
end

@doc "Return a TaylorModel build from a VectorObjectiveFunction `objf`."
function _init_model( cfg ::TaylorConfig, objf :: AbstractObjective, 
    mop :: AbstractMOP, id :: AbstractIterData, ac :: AbstractConfig ) :: Tuple{TaylorModel, TaylorMeta}
    tm0 = TaylorModel(; mop = mop, objf = objf );
    set_gradients!( tm0, objf, mop );
    if cfg.degree >= 2
        set_hessians!( tm0, objf, mop );
    end
    tmeta0 = TaylorMeta()
    return update_model( tm0, objf, tmeta0, mop, id, ac);    
end

function update_model( tm :: TaylorModel, objf :: AbstractObjective, tmeta :: TaylorMeta,
    mop :: AbstractMOP, id :: AbstractIterData, :: AbstractConfig; ensure_fully_linear :: Bool = false ) :: Tuple{TaylorModel,TaylorMeta}
    @info "Building Taylor model(s)."
    tm.x0 = xᵗ(id);
    tm.fx0 = fxᵗ(id)[output_indices(objf,mop)];
    
    # set gradients
    empty!(tm.g)
    for ℓ = 1 : num_outputs(objf)
        push!(tm.g, get_gradient(tm.diff_fn, tm.x0, ℓ))
    end
    
    # and hessians if needed
    if !isnothing(tm.hess_fn)
        empty!(tm.H)
        for ℓ = 1 : num_outputs(objf)
            hess_mat = Matrix(get_hessian(tm.hess_fn, tm.x0, ℓ));
            push!(tm.H, hess_mat);
        end
    end
    @info "Done building Taylor model(s)."
    return tm, tmeta
end

function improve_model(tm::TaylorModel, ::AbstractObjective, tmeta :: TaylorMeta,
    ::AbstractMOP, id :: AbstractIterData, :: AbstractConfig;
    ensure_fully_linear :: Bool = false ) :: Tuple{TaylorModel, TaylorMeta}
    tm, tmeta 
end

function _eval_models( tm :: TaylorModel, h :: RVec, ℓ :: Int ) :: Real
    ret_val = tm.fx0[ℓ] + tm.g[ℓ]'h
    if !isempty(tm.H)
        ret_val += .5 * h'tm.H[ℓ]*h 
    end
    return ret_val
end

function eval_models( tm :: TaylorModel, x̂ :: RVec, ℓ :: Int ) :: Real
    h = x̂ .- tm.x0;
    return _eval_models( tm, h, ℓ);
 end

function eval_models( tm :: TaylorModel, x̂ :: RVec ) :: RVec
    h = x̂ .- tm.x0;
    return vcat( [_eval_models(tm, h, ℓ) for ℓ=1:num_outputs(tm.objf)]... )
end

function get_gradient( tm :: TaylorModel, x̂ :: RVec, ℓ :: Int) :: RVec
    if isempty(tm.H)
        return tm.g[ℓ]
    else
        h = x̂ .- tm.x0;
        return tm.g[ℓ] .+ .5 * ( tm.H[ℓ]' + tm.H[ℓ] ) * h
    end
end

function get_jacobian( tm :: TaylorModel, x̂ :: RVec )
    grad_list = [get_gradient(tm, x̂, ℓ) for ℓ=1:num_outputs( tm.objf ) ]
    return transpose( hcat( grad_list... ) )
end

