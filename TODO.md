TODO
====

* Make `ensure_precision` also convert to static array, e.g. for use in `IterData`?

* Support half open intervals and individual unconstrained variables (should be relatively easy)
⇒ use a weighted trust region norm and a vector of Δ values
* Saving/logging
* Python Bridge (and OpenFoam example)

* Improve Docs

## Done
* Warm start capabilities (work with populated AlgoConfig) [ does kind of work now ]u
* Allow for the passing of user defined gradients for cheap functions (instead of requiring Autodiff)
* Approximation of local ideal point:  
  - Distinguish between cheap and expensive functions
  - If function is expensive: pass model output and gradient directly
  - If function is cheap and has user defined gradient: use user defined gradient
* change function signatures: config_struct::AlgoConfig now contains a field storing problem::MixedMOP.  
  It is not necessary to pass problem properties to functions that already take config_struct.
* Randomize sampling of new model points instead of requiring strict maximization along unexplored axes.  
  Perhaps port sampling used in old MATLAB implementation.