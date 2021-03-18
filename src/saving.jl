# This file specifies some saving functions for configuration 
# and iteration data.
# I am not yet sure whether to make the below methods in the 
# interface definitions eventually...

import FileIO;

function _jld2_filename( filename :: AbstractString ) 
  fn, ext = splitext(filename)
  if isempty(ext) || ext ≠ ".jld2"
    return string(fn, ".jld2")
  else
    return filename
  end
end

"""
    save_config(filename, ac :: AbstractConfig )

Save the configuration object `ac` at path `filename`.
Ensures, that the file extension is `.jld2`.
The fieldname to retrieve the database object is `database`.

Returns the save path if successful and `nothing` else.
"""
function save_config( filename :: AbstractString, ac :: AbstractConfig ) :: Union{Nothing, AbstractString}
  fn = _jld2_filename(filename)  
  try 
    FileIO.save(fn, Dict("abstract_config" => ac))
    return fn
  catch e
    @warn "Could not save algorithm configuration:" exception=(ex,bt)
    return nothing
  end
end

"""
    load_config(fn :: AbstractString)
Load and return the `AbstractConfig` that was saved previously 
with [`save_config`](@ref).
Return `nothing` on error.
"""
function load_config( filename :: AbstractString ) :: Union{AbstractConfig, Nothing}
  try
    return FileIO.load( filename )["abstract_config"]
  catch e
    @warn "Could not load algorithm configuration:" exception=(ex,bt)
    return nothing
  end
end

"""
    save_database(filename, id :: AbstractIterData )

Save the database that is referenced by `db(id)` at path `filename`.
Ensures, that the file extension is `.jld2`.
The fieldname to retrieve the database object is `database`.

Returns the save path if successful and `nothing` else.
"""
function save_database(filename::AbstractString, id :: AbstractIterData):: Union{Nothing, AbstractString}
  return save_database(filename, db(id))
end

"""
    save_database(filename, DB :: AbstractDB )

Save the database `DB` at path `filename`.
Ensures, that the file extension is `.jld2`.
The fieldname to retrieve the database object is `database`.

Returns the save path if successful and `nothing` else.
"""
function save_database( filename :: AbstractString, DB :: AbstractDB ) :: Union{Nothing, AbstractString}
  fn = _jld2_filename(filename)  
  try
    FileIO.save(fn, Dict("database" => DB))
    return fn
  catch e
    @warn "Could not save evaluation database:" exception=(ex,bt)
    return nothing
  end
end

"""
    load_database(fn :: AbstractString)
Load and return the `<:AbstractDB` that was saved previously 
with [`save_database`](@ref).
Return `nothing` on error.
"""
function load_database( filename :: AbstractString ) :: Union{AbstractDB, Nothing}
  try
    return FileIO.load( filename )["database"]
  catch e
    @warn "Could not load evaluation database:" exception=(ex,bt)
    return nothing
  end
end

"""
    save_iter_data(filename, id :: AbstractIterData )

Save the whole object `id` at path `filename`.
Ensures, that the file extension is `.jld2`.
The fieldname to retrieve the database object is `database`.

Returns the save path if successful and `nothing` else.
"""
function save_iter_data( filename :: AbstractString, id :: AbstractIterData ) :: Union{Nothing, AbstractString}
  fn = _jld2_filename(filename)  
  try
    FileIO.save(fn, Dict("iter_data" => id))
    return fn
  catch e
    @warn "Could not save iteration data object:" exception=(ex,bt)
    return nothing
  end
end

"""
    load_iter_data(fn :: AbstractString)
Load and return the `<:AbstractIterData` that was saved previously 
with [`save_iter_data`](@ref).
Return `nothing` on error.
"""
function load_iter_data( filename :: AbstractString ) :: Union{AbstractDB, Nothing}
  try
    return FileIO.load( filename )["iter_data"]
  catch e
    @warn "Could not load iteration data object:" exception=(ex,bt)
    return nothing
  end
end
