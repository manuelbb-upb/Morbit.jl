TODO
====

* Acceptance test: check only expensive (aka surrogated) objectives???
* Acceptance test: min-max-scaling on values?
* Warm start capabilities (work with populated AlgoConfig) [ does kind of work now ]
* Saving/logging
* Randomize sampling of new model points instead of requiring strict maximization along unexplored axes.  
  Perhaps port sampling used in old MATLAB implementation.
* Python Bridge (and OpenFoam example)

# Lower Priority
* Documentation (use corresponding Julia Package?)  
* Introduce convenience functions to extract iteration information from optimized AlgoConfig struct.
* More plotting recipes:  
  - 3D plotting (decision space & objective space)
  - Plotting of critical values.
  - Animation of iteration and model construction.

## Done
* Allow for the passing of user defined gradients for cheap functions (instead of requiring Autodiff)
* Approximation of local ideal point:  
  - Distinguish between cheap and expensive functions
  - If function is expensive: pass model output and gradient directly
  - If function is cheap and has user defined gradient: use user defined gradient
* change function signatures: config_struct::AlgoConfig now contains a field storing problem::MixedMOP.  
  It is not necessary to pass problem properties to functions that already take config_struct.