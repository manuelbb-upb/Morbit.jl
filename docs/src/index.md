```@meta
CurrentModule = Morbit
```

# Morbit

The package `Morbit.jl` provides a local derivative-free solver for multiobjective optimization problems with possibly expensive objectives.
It is meant to find a **single** Pareto-critical point, not a good covering of the global Pareto Set.

“Morbit” stands for **M**ultiobjective **O**ptimization by **R**adial **B**asis **F**unction **I**nterpolation **i**n **T**rust-regions. 
The name was chosen so as to pay honors to the single objective algorithm ORBIT by Wild et. al.  

We have a [paper](https://www.mdpi.com/2297-8747/26/2/31) explaining the algorithm!

This was my first project using Julia and there have been many messy rewrites.
Nonetheless, the solver should now work sufficiently well to tackle most problems. 
I hope to rewrite the custom types soonish. At the moment they are weakly typed and the performance suffers.

**To get started, see the examples, e.g. [Two Parabolas](@ref).

This project was founded by the European Region Development Fund.

```@raw html
<img src="https://www.efre.nrw.de/fileadmin/Logos/EU-Fo__rderhinweis__EFRE_/EFRE_Foerderhinweis_englisch_farbig.jpg" width="45%"/>
```

```@raw html
<img src="https://www.efre.nrw.de/fileadmin/Logos/Programm_EFRE.NRW/Ziel2NRW_RGB_1809_jpg.jpg" width="45%"/>
```

```@index
```

```@autodocs
Modules = [Morbit]
```
