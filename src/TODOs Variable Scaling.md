# TODOs Variable Scaling

## `AbstractVarScaler`
* ~~Define functions to return the scaled variable bounds 
  (like `full_bounds_internal` does right now for `AbstractMOP`s)~~
* Build boundary safeguard into the var scalers?
* ~~Define `jacobian_of_unscaling` !!!~~

* Enable the usage of prior surrogate models to calculate new iteration dependent scalers.

## `AbstractMOP`
* ~~What functions can we get rid of?~~
  ~~I am thinking of `full_bounds_internal` and the `scale` and `unscale` functions~~
* ~~Remove all evaluation functions for `scaled_site`s.~~
* Replace `scale\(.*(?:mop)\*?\)` with scale(x,scal)
* Replace `eval_mop…` and `eval_vec_mop`a functions
* Replace `full_bounds_internal\(.*(?:mop).*?\)`

## `AbstractIterData`
* store unscaled `x` instead of scaled `x` to return with `get_x(id)`
* store current `AbstractVarScaler` and define `get_var_scaler(id)`
* store `x_scaled` and define `_get_x_scaled(id)` and 
  `_set_x_scaled!(id,_x)`
* store the `AbstractVarScaler` that was used to transform `_get_x_scaled(x)` and define `get_last_var_scaler(id)`
* Then, define function `get_x_scaled(id)` à la 
  ```julia
	function get_x_scaled(id :: AbstractIterData )
		var_scaler = get_var_scaler(id)
		if get_last_var_scaler(id) == var_scaler
			return _get_x_scaled(id)
		else
			x_scaled = transform(id, var_scaler)
			_set_x_scaled!(id, x_scaled)
		return x_scaled
	end
  ```

## `AbstractDB`
* Like for `AbstractIterData`, store the `AbstractVarScaler` when `transform!` is called on the database.
  Make it retrievable with `get_last_var_scaler(db)` or something.
* Change `is_tansformed` to check whether the data is transformed according to `get_last_var_scaler(db)`.
  If not: Untransform and transform with new scaler.

Why don't store only untransformed data? 
Because we assume that the surrogate model construction benefits from data scaling.
The intricate `is_transformed` check then avoids redundant transformations for multiple different model types.

## Surrogate models

* The current `AbstractVarScaler` should be passed to the model building functions.
* Store the current scaler in `ExactModel` and adapt the `eval`
 functions.