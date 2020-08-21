using JLD2: @save, @load
using Random: randstring
export save_config, load_config

function save_config( opt :: AlgoConfig, filename :: Union{Nothing, String} = nothing )
  if isnothing( filename )
    filename = joinpath( pwd(), string( randstring(12), ".jld" ) )
  end
  println("Trying to save AlgoConfig object as \n\t$filename")
  try
    @save filename opt
    println("Success.")
    return filename
  catch e
    println("Could NOT save object as \n\t$filename")
  end
end

function load_config( filename :: String)
  local opt :: AlgoConfig;
  if isfile(filename)
    @load filename opt
    return opt
  else
    println("$filename is not a file.")
  end
end
