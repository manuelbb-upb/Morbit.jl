* Delete the `apply_internal_sorting` and `reverse_interal_sorting`.
* Rethink database ( tuples of function indices as keys )
* Rethink `prepare_XXXX` signatures
* Finish `init_surrogates` etc in `SurrogatesImplementation.jl`
  - `init_wrapper` has changed signature 
  - make `init_model` return a surrogate wrapper?

* Replace `eval_models(sc,…)` by `eval_objectives_surrogates`.
* Replace `get_optim_handle` stuff, especially in PS descent calculation.

* `output_indices(objf,mop)` now is `get_objective_positions(mop, output_index)`;
  similarly for `get_eq_constraint_positions` and `get_ineq_constraint_positions`
  Defined in AbstratMOPInterface.jl. Do we actually need this?
* `_init_model` -> `init_model`

* Reactivate SMOP in `initialize_data`

* Switch AbstractDB/ArrayDB to SuperDB
* !REWRITE MixedMOP/StaticMOP

* Remove `eval_handle` and have derivatives be part of `AbstractObjective`

* Remove "untransforming" of databases at end of routine and change Docs

* `get_saveable_type(::SurrogateMeta)` -> `get_saveable_type(::SurrogateConfig, x, y)`
* check iter_data for additional type fields

# MOP interface

* `output_indices( mop, func_indices )`
  REMOVE old usages
