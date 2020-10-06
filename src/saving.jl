using JLD2: @save, @load
using Random: randstring
export save_config, load_config

function save_config( src :: AlgoConfig, filename :: Union{Nothing, String} = nothing )
  if isnothing( filename )
    filename = joinpath( pwd(), string( randstring(12), ".jld" ) )
  end
  println("Trying to save AlgoConfig object as \n\t$filename")
  try
    opt = deepcopy(src)
    opt.problem.vector_of_expensive_funcs = [];
    opt.problem.vector_of_cheap_funcs = [];
    opt.problem.vector_of_gradient_funcs = [];
    @save filename opt
    println("Success.")
    return filename
  catch e
    println("Could NOT save object as \n\t$filename")
  end
end

function load_config( filename :: String)
  local opt :: AlgoConfig;
  #Core.eval_models(Main, :(import Morbit: AlgoConfig))
  #Core.eval_models( Main, :(using Morbit))
  if isfile(filename)
    @load filename opt
    return opt
  else
    println("$filename is not a file.")
  end
end
