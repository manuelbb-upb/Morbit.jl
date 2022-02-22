```@meta
EditURL = "<unknown>/../src/custom_logging.jl"
```

# Printing Debug Info

We provide a custom formatter method and define our own log levels.
The user can choose, how much information is printed and it should
look nicer this way.

Log messages are only displayed if they have a `LogLevel` that
is ≥ than a minimum log-level defined for the current logger.
The current minimum log-level can be determined with
```julia
Logging.min_enabled_level( Logging.current_logger() )
```
For more information see the [docs](https://docs.julialang.org/en/v1/stdlib/Logging/).
Usually, the minimum log level is -1.

We have the following `LogLevel`s and they can be referred to as `Morbit.loglevel1` ect.:

````julia
const loglevel1 = LogLevel(-1);
const loglevel2 = LogLevel(-2);
const loglevel3 = LogLevel(-3);
const loglevel4 = LogLevel(-4);
nothing #hide
````

The can be made visible by setting one of these levels with a custom logger.
For example, to see the most detailled messages, do something like this:
```julia
logger = Logging.ConsoleLogger( stderr, Morbit.loglevel4 )
Logging.global_logger(logger)
```
Or use `with_logger(logger) do … end` to leave the global logger unchanged.

For prettier output, we define custom colors and indented prefixes:

````julia
const printDict = Dict(
    loglevel1 => (:blue, "Morbit"),
    loglevel2 => (:cyan, "Morbit "),
    loglevel3 => (:green, "Morbit  "),
    loglevel4 => (:green, "Morbit   ")
)
````

These are used in the `morbit_formatter`.
The `morbit_formatter` can be enabled for a logger, such as `Logging.ConsoleLogger`,
by passing the keyword argument `meta_formatter`, i.e.,
```julia
Logging.ConsoleLogger( stderr, Morbit.loglevel4; meta_formatter = morbit_formatter )
```
Note, that `morbit_formatter` is exported.

````julia
function morbit_formatter(level::LogLevel, _module, group, id, file, line)
    @nospecialize
	global printDict
    if level in keys(printDict)
        color, prefix = printDict[ level ]
        return color, prefix, ""
    else
        return Logging.default_metafmt( level, _module, group, id, file, line )
    end
end

function get_morbit_logger( level = Morbit.loglevel4 )
    Logging.ConsoleLogger( stderr, level; meta_formatter = morbit_formatter )
end
````

## Shorthand Function
The following (unexported) function sets the global logger to print everything:

````julia
function print_all_logs()
    Logging.global_logger( get_morbit_logger() )
end
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

