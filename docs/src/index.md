```@meta
CurrentModule = Morbit
```

# Morbit

The package `Morbit.jl` provides a local derivative-free solver for multiobjective optimization problems with possibly expensive objectives.
It is meant to find a **single** Pareto-critical point, not a good covering of the global Pareto Set.

“Morbit” stands for **M**ultiobjective **O**ptimization by **R**adial **B**asis **F**unction **I**nterpolation **i**n **T**rust-regions. 
The name was chosen so as to pay honors to the single objective algorithm ORBIT by Wild et. al.  
There is a [preprint in the arXiv](https://arxiv.org/abs/2102.13444) that explains what is going on inside.
It has been submitted to the MCA journal.

This was my first project using Julia and there have been many messy rewrites.
Nonetheless, the solver should now work sufficiently well to tackle most problems.

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
