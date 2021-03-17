var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = Morbit","category":"page"},{"location":"#Morbit","page":"Home","title":"Morbit","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [Morbit]","category":"page"},{"location":"#Morbit.TransformerFn-Tuple{Array{R,1} where R<:Real}","page":"Home","title":"Morbit.TransformerFn","text":"Unscale the point x̂ from internal to original domain.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit._add!-Tuple{Morbit.AbstractMOP,Morbit.AbstractObjective,Union{Nothing, Array{Int64,1}}}","page":"Home","title":"Morbit._add!","text":"Add an objective function to MOP with specified output indices.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit._backtrack-Tuple{Array{R,1} where R<:Real,Array{R,1} where R<:Real,Real,Real,Morbit.SurrogateContainer,Bool}","page":"Home","title":"Morbit._backtrack","text":"Perform a backtracking loop starting at x with an initial step of step_size .* dir and return trial point x₊, the surrogate value-vector m_x₊ and the final step s = x₊ .- x.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit._del!-Tuple{Morbit.AbstractMOP,Morbit.AbstractObjective}","page":"Home","title":"Morbit._del!","text":"Remove an objective function from MOP.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit._get_shape_param-Tuple{RbfConfig,Morbit.AbstractIterData}","page":"Home","title":"Morbit._get_shape_param","text":"Get real-valued shape parameter for RBF model from current iter data. cfg allows for a string expression which would be evaluated here.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit._init_model-Tuple{ExactConfig,Morbit.AbstractObjective,Morbit.AbstractMOP,Morbit.AbstractIterData,Morbit.AbstractConfig}","page":"Home","title":"Morbit._init_model","text":"Return an ExactModel build from a VectorObjectiveFunction objf.  Model is the same inside and outside of criticality round.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit._init_model-Tuple{TaylorConfig,Morbit.AbstractObjective,Morbit.AbstractMOP,Morbit.AbstractIterData,Morbit.AbstractConfig}","page":"Home","title":"Morbit._init_model","text":"Return a TaylorModel build from a VectorObjectiveFunction objf.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit._intersect_bounds-NTuple{4,Array{R,1} where R<:Real}","page":"Home","title":"Morbit._intersect_bounds","text":"Return smallest positive and biggest negative and σ₊ and σ₋ so that x .+ σ± .* d stays within bounds.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit._local_bounds-Tuple{Array{R,1} where R<:Real,Union{Real, Array{R,1} where R<:Real},Array{R,1} where R<:Real,Array{R,1} where R<:Real}","page":"Home","title":"Morbit._local_bounds","text":"Return lower and upper bound vectors combining global and trust region constraints.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit._multifactorial-Tuple{Array{Int64,1}}","page":"Home","title":"Morbit._multifactorial","text":"Factorial of a multinomial.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit._objf_index-Tuple{Morbit.AbstractObjective,Morbit.AbstractMOP}","page":"Home","title":"Morbit._objf_index","text":"Position of objf in list_of_objectives(mop).\n\n\n\n\n\n","category":"method"},{"location":"#Morbit._rel_tol_test_decision_space-Tuple{Union{Real, Array{R,1} where R<:Real},Real,Morbit.AbstractConfig}","page":"Home","title":"Morbit._rel_tol_test_decision_space","text":"True if stepsize or radius too small.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit._wrap_func-Tuple{Type{var\"#s30\"} where var\"#s30\"<:Morbit.AbstractObjective,Function,Morbit.SurrogateConfig,Int64,Int64}","page":"Home","title":"Morbit._wrap_func","text":"A general constructor.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.add_objective!","page":"Home","title":"Morbit.add_objective!","text":"add_objective!( mop :: MixedMOP, func :: T where{T <: Function}, type :: Symbol = :expensive, n_out :: Int64 = 1, can_batch :: Bool = false )\n\nAdd scalar-valued objective function func to mop structure. func must take an RVec as its (first) argument, i.e. represent a function f ℝ^n  ℝ. type must either be :expensive or :cheap to determine whether the function is replaced by a surrogate model or not.\n\nIf type is :cheap and func takes 1 argument only then its gradient is calculated by ForwardDiff. A cheap function func with custom gradient function grad (representing f  ℝ^n  ℝ^n) is added by\n\nadd_objective!(mop, func, grad)\n\nThe optional argument n_out allows for the specification of vector-valued objective functions. This is mainly meant to be used for expensive functions that are in some sense inter-dependent.\n\nThe flag can_batch defaults to false so that the objective function is simply looped over a bunch of arguments if required. If can_batch == true then the objective function must be able to return an array of results when provided an array of input vectors (whilst still returning a single result, not a singleton array containing the result, for a single input vector).\n\nExamples\n\n# Define 2 scalar objective functions and a MOP ℝ^2 → ℝ^2\n\nf1(x) =  x[1]^2 + x[2]\n\nf2(x) = exp(sum(x))\n∇f2(x) = exp(sum(x)) .* ones(2);\n\nmop = MixedMOP()\nadd_objective!(mop, f1, :cheap)     # gradient will be calculated using ForwardDiff\nadd_objective!(mop, f2, ∇f2 )       # gradient is provided\n\n\n\n\n\n","category":"function"},{"location":"#Morbit.add_objective!-Tuple{MixedMOP,Function,Function}","page":"Home","title":"Morbit.add_objective!","text":"add_objective!( mop :: MixedMOP, func :: T where{T <: Function}, grad :: T where{T <: Function})\n\nAdd scalar-valued objective function func and its vector-valued gradient grad to mop struture.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.add_objective!-Tuple{MixedMOP,Function,Morbit.SurrogateConfig}","page":"Home","title":"Morbit.add_objective!","text":"Add a scalar objective to mop::MixedMOP modelled according to model_config.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.add_vector_objective!-Tuple{MixedMOP,Function,Morbit.SurrogateConfig}","page":"Home","title":"Morbit.add_vector_objective!","text":"Add a vector objective to mop::MixedMOP modelled according to model_config.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.combine-Tuple{Morbit.AbstractObjective,Morbit.AbstractObjective}","page":"Home","title":"Morbit.combine","text":"Combine two objectives. Only needed if combinable can return true.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.combine-Union{Tuple{T}, Tuple{F}, Tuple{F,T}} where T<:Function where F<:Function","page":"Home","title":"Morbit.combine","text":"Get a new function function handle stacking the output of func1 and func2.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.delete_surrogate!-Tuple{Morbit.SurrogateContainer,Int64}","page":"Home","title":"Morbit.delete_surrogate!","text":"Delete surrogate wrapper at position si from list sc.surrogates.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.eval_all_objectives-Tuple{Morbit.AbstractMOP,Array{R,1} where R<:Real}","page":"Home","title":"Morbit.eval_all_objectives","text":"(Internally) Evaluate all objectives at site x̂::RVec. Objective order might differ from order in which they were added.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.eval_and_sort_objectives-Tuple{Morbit.AbstractMOP,Array{R,1} where R<:Real}","page":"Home","title":"Morbit.eval_and_sort_objectives","text":"Evaluate all objectives at site x̂::RVec and sort the result according to the order in which objectives were added.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.eval_models-Tuple{Morbit.ExactModel,Array{R,1} where R<:Real,Int64}","page":"Home","title":"Morbit.eval_models","text":"Evaluate output ℓ of the ExactModel em at scaled site x̂.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.eval_models-Tuple{Morbit.ExactModel,Array{R,1} where R<:Real}","page":"Home","title":"Morbit.eval_models","text":"Evaluate the ExactModel em at scaled site x̂.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.eval_models-Tuple{Morbit.SurrogateContainer,Morbit.AbstractMOP,Array{R,1} where R<:Real,Int64}","page":"Home","title":"Morbit.eval_models","text":"Return model value for output l of sc at x̂. Index l is assumed to be an internal index in the range of 1,…,nobjfs, where nobjfs is the total number of (scalarized) objectives stored in sc.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.eval_objf-Tuple{Morbit.AbstractObjective,Array{R,1} where R<:Real}","page":"Home","title":"Morbit.eval_objf","text":"Evaluate the objective at unscaled site(s). and increase counter.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.eval_objf_at_site-Tuple{Morbit.AbstractObjective,Union{Array{R,1} where R<:Real, Array{var\"#s12\",1} where var\"#s12\"<:(Array{R,1} where R<:Real)}}","page":"Home","title":"Morbit.eval_objf_at_site","text":"Evaluate objective at scaled site.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.find_box_independent_points1!-Tuple{Morbit.RbfModel,RbfConfig,Morbit.AbstractIterData,Morbit.AbstractMOP,Morbit.AbstractConfig}","page":"Home","title":"Morbit.find_box_independent_points1!","text":"Find affinely independent results in database box of radius Δ around x Results are saved in rbf.tdata[tdata_index].  Both rbf.Y and rbf.Z are changed.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.find_points_in_box-Tuple{Morbit.AbstractIterData,Array{R,1} where R<:Real,Array{R,1} where R<:Real}","page":"Home","title":"Morbit.find_points_in_box","text":"Indices of sites in database that lie in box with bounds lb and ub.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.full_lower_bounds_internal-Tuple{Morbit.AbstractMOP}","page":"Home","title":"Morbit.full_lower_bounds_internal","text":"Return lower variable bounds for scaled variables.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.full_upper_bounds_internal-Tuple{Morbit.AbstractMOP}","page":"Home","title":"Morbit.full_upper_bounds_internal","text":"Return upper variable bounds for scaled variables.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.get_gradient-Tuple{Morbit.ExactModel,Array{R,1} where R<:Real,Int64}","page":"Home","title":"Morbit.get_gradient","text":"Gradient vector of output ℓ of em at scaled site x̂.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.get_gradient-Tuple{Morbit.SurrogateContainer,Morbit.AbstractMOP,Array{R,1} where R<:Real,Int64}","page":"Home","title":"Morbit.get_gradient","text":"Return a gradient for output l of sc at x̂. Index 4 is assumed to be an internal index in the range of 1,…,nobjfs, where nobjfs is the total number of (scalarized) objectives stored in sc.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.get_jacobian-Tuple{Morbit.ExactModel,Array{R,1} where R<:Real}","page":"Home","title":"Morbit.get_jacobian","text":"Jacobian Matrix of ExactModel em at scaled site x̂.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.get_optim_handle-Tuple{Morbit.SurrogateContainer,Morbit.AbstractMOP,Int64}","page":"Home","title":"Morbit.get_optim_handle","text":"Return a function handle to be used with NLopt for output l of sc. Index l is assumed to be an internal index in the range of 1,…,nobjfs, where nobjfs is the total number of (scalarized) objectives stored in sc.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.get_optim_handle-Tuple{Morbit.SurrogateModel,Int64}","page":"Home","title":"Morbit.get_optim_handle","text":"Return a function handle to be used with NLopt for output ℓ of model. That is, if model is a surrogate for two scalar objectives, then ℓ must  be either 1 or 2.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.inc_evals!","page":"Home","title":"Morbit.inc_evals!","text":"Increase evaluation count by N\n\n\n\n\n\n","category":"function"},{"location":"#Morbit.init_surrogates-Tuple{Morbit.AbstractMOP,Morbit.AbstractIterData,Morbit.AbstractConfig}","page":"Home","title":"Morbit.init_surrogates","text":"Return a SurrogateContainer initialized from the information provided in mop.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.local_bounds-Tuple{Morbit.AbstractMOP,Array{R,1} where R<:Real,Union{Real, Array{R,1} where R<:Real}}","page":"Home","title":"Morbit.local_bounds","text":"Local bounds vectors lb_eff and ub_eff using scaled variable constraints from mop.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.max_evals!-Tuple{Morbit.AbstractObjective,Int64}","page":"Home","title":"Morbit.max_evals!","text":"Set upper bound of № evaluations to N\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.max_evals-Tuple{Morbit.AbstractObjective}","page":"Home","title":"Morbit.max_evals","text":"(Soft) upper bound on the number of function calls. \n\n\n\n\n\n","category":"method"},{"location":"#Morbit.model_cfg-Tuple{Morbit.AbstractObjective}","page":"Home","title":"Morbit.model_cfg","text":"Return surrogate configuration used to model the objective internally.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.non_negative_solutions-Tuple{Int64,Int64}","page":"Home","title":"Morbit.non_negative_solutions","text":"Return array of solution vectors [x1, …, xlen] to the equation x_1 +  + x_len = rhs where the variables must be non-negative integers.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.num_evals!-Tuple{Morbit.AbstractObjective,Int64}","page":"Home","title":"Morbit.num_evals!","text":"Set evaluation counter to N.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.num_evals-Tuple{Morbit.AbstractObjective}","page":"Home","title":"Morbit.num_evals","text":"Number of calls to the original objective function.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.num_objectives-Tuple{Morbit.AbstractMOP}","page":"Home","title":"Morbit.num_objectives","text":"Number of scalar-valued objectives of the problem.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.pop_objf!-Tuple{Morbit.AbstractMOP,Morbit.AbstractObjective}","page":"Home","title":"Morbit.pop_objf!","text":"Remove objf from list_of_objectives(mop) and return its output indices.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.ps_polish_algo-Tuple{Morbit.AbstractConfig}","page":"Home","title":"Morbit.ps_polish_algo","text":"Specify local algorithm to polish Pascoletti-Serafini solution. Uses 1/4 of maximum allowed evals.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.reset_evals!-Tuple{Morbit.AbstractMOP}","page":"Home","title":"Morbit.reset_evals!","text":"Set evaluation counter to 0 for each VectorObjectiveFunction in m.vector_of_objectives.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.reverse_internal_sorting-Tuple{Array{R,1} where R<:Real,Morbit.AbstractMOP}","page":"Home","title":"Morbit.reverse_internal_sorting","text":"Sort an interal objective vector so that the objectives are in the order in which they were added.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.reverse_internal_sorting_indices-Tuple{Morbit.AbstractMOP}","page":"Home","title":"Morbit.reverse_internal_sorting_indices","text":"Return index vector so that an internal objective vector is sorted according to the order the objectives where added.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.scale-Tuple{Array{R,1} where R<:Real,Morbit.AbstractMOP}","page":"Home","title":"Morbit.scale","text":"Scale variables fully constrained to a closed interval to [0,1] internally.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.set_gradients!-Tuple{Morbit.ExactModel,Morbit.AbstractObjective,Morbit.AbstractMOP}","page":"Home","title":"Morbit.set_gradients!","text":"Modify/initialize thec exact model mod so that we can differentiate it later.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.set_gradients!-Tuple{Morbit.TaylorModel,Morbit.AbstractObjective,Morbit.AbstractMOP}","page":"Home","title":"Morbit.set_gradients!","text":"Modify/initialize thec exact model mod so that we can differentiate it later.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.strict_backtracking-Tuple{Morbit.AbstractConfig}","page":"Home","title":"Morbit.strict_backtracking","text":"Require a descent in all model objective components.  Applies only to backtracking descent steps, i.e., :steepest_descent.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.unscale-Tuple{Array{R,1} where R<:Real,Morbit.AbstractMOP}","page":"Home","title":"Morbit.unscale","text":"Reverse scaling for fully constrained variables from [0,1] to their former domain.\n\n\n\n\n\n","category":"method"}]
}
