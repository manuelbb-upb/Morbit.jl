# This module provides some crude (and not necessarily
# efficient) methods for multivariate polynomials.
# They are intended for the construction of polynomial
# interpolation and regression models.

module Polynomials

using Parameters: @with_kw
using Combinatorics: multinomial
export Polynomial, gradient, hessian, non_negative_solutions

############################### MultiIndex ###############################
@doc """
MultiIndex{N} has one field `vals :: NTuple{N, Int64}`
and provides methods to treat this N-vector as a multitindex.
"""
struct MultiIndex{N}
    vals :: NTuple{N, Int64}
end

function MultiIndex( vals :: Vector{Int64} )
    return MultiIndex( Tuple(vals) )
end

Base.length(α :: MultiIndex{N}) where N = N;

@doc "Simply return `i`-th value of multi-index."
function (α::MultiIndex{N} where N)(i :: Int64)
    α.vals[i]
end

@doc "Evaluate the monomial ``x^α`` where `x` is a real vector and `α` a multiindex."
function (α::MultiIndex{N})(x::Vector{R} where R<:Real) where N
    prod( x[i]^α(i) for i = 1:N)
end

@doc "Return ``α₁! · … · αₙ!``."
function Base.factorial( α :: MultiIndex{N} ) where N
    prod( factorial(α(i)) for i = 1:N )
end

@doc "Return ``Σᵢⁿ αᵢ.``"
function Base.sum( α :: MultiIndex{N} ) where N
    sum( α(i) for i = 1 : N )
end

############################### Monomials ###############################

@doc """
A multivariate monomial that simply combines as real `factor` with a
multiindex `α` and provides methods to evaluate and differentiate the
monomial at sites `x`∈ ℝⁿ.
"""
@with_kw mutable struct Monomial{N}
    factor :: R where R<: Real = 0.0
    α :: MultiIndex{N} = MultiIndex(Tuple(Zeros(Int64, N)))

    # cache for the 'symbolical' gradient
    grad :: Union{Nothing,Vector{Monomial}} = nothing;
    # cache for the 'symbolical' hessian
    hessian :: Union{Nothing, Vector{Vector{Monomial}}} = nothing;
end

# helper: minimal positional constructor
function Monomial( factor::R where R<:Real, α :: MultiIndex )
    return Monomial(;
        factor = factor,
        α = α
    )
end

@doc "Evaluate monomial `m` at site `x`"
function (m::Monomial{N} where N)(x :: Vector{R} where R<:Real)
    return m.factor * m.α(x)
end

function (M::Vector{Monomial})(x :: Vector{R} where R<:Real)
    return [ M[i](x) for i = eachindex(M) ]
end

@doc """
Return the vector `G` of Monomials that results from "symbolically"
differentiating `m` with respect to each coordinate.

This method does **not** evaluate the vector `G` at any site.
Note also, that the `factor` of each entry of `G` does **not** contain `m.factor`.
"""
function gradient( m :: Monomial{N} ) where N
    G = Vector{ Monomial }( undef, N );
    for i = 1 : N
        β = [ m.α.vals... ]
        β[i] = max( β[i] - 1, 0 );
        G[i] = Monomial(
            factor = m.α(i),
            α = MultiIndex(β),
        )
     end
     return G
end

@doc "Set the field `m.grad` to contain the symbolical gradient of `m`."
function gradient!( m::Monomial{N} ) where N
    m.grad = gradient(m)
end

@doc "Return the gradient of `m` evaluated at site `x`."
function gradient( m :: Monomial{N}, x :: Vector{R} ) where{N, R<:Real}
    if isnothing(m.grad)
        gradient!(m)
    end
    if m.factor == 0.0
        return zeros(R, N)
    else
        #return m.factor .* [ m.grad[i](x) for i = eachindex(m.grad) ]
        return m.factor .* m.grad(x)
    end
end

@doc """
Set the fiel `m.hessian` to contain a list of vectors so that
the `i`-th vector in the list contains those monomials that result from
differentiating the `i`-th component of `m.grad`.
Hence, evaluating the list entries gives the rows of the Hessian of `m`
(again, up to a scaling by `m.factor` and the factors of `m.grad`)
"""
function hessian!( m :: Monomial{N} ) where N
    if isnothing(m.grad)
        gradient!(m)
    end
    m.hessian = Vector{Vector{Monomial{N}}}( undef, N );
    for i = 1 : N
        m.hessian[i] = gradient( m.grad[i] )
    end
end

@doc "Return the hessian of `m` evaluated at site `x`."
function hessian( m :: Monomial{N}, x :: Vector{R} ) where{N, R<:Real}
    if isnothing(m.hessian)
        hessian!(m)
    end
    H = zeros(R, (N,N) )
    for i = 1 : N
        H[i,:] =  iszero( m.factor ) || iszero(m.grad[i].factor) ? zeros(R, N) :
            m.factor .* m.grad[i].factor .* m.hessian[i]( x );
    end
    return H
end

############################### Polynomials ###############################

@doc "A Polynomial is simply a list of multivariate `monomials`."
@with_kw mutable struct Polynomial{N}
    monomials :: Vector{Monomial{N}} = [];
end

# helper constructor to define polynomial from single monomial
function Polynomial( m :: Monomial{N} where N)
    return Polynomial( monomials = [m,] )
end

# helper constructor to define polynomial from a factor and a tuple
function Polynomial( factor :: R, v :: Union{ Vector{Int64},
        NTuple{N, Int64}} ) where{R<:Real, N}
    m = Monomial(;
        factor = factor,
        α = MultiIndex( v )
    )
    return Polynomial( m )
end

# helper constructor to define polynomial from a factor and a MultiIndex
function Polynomial( factor :: R, α :: MultiIndex{N}) where{R<:Real, N}
    m = Monomial(;
        factor = factor,
        α = α
    )
    return Polynomial( m )
end


@doc "Evaluate `p` at `x` by summing up the evaluations of all monomials"
function (p::Polynomial)( x :: Vector{R} ) where R<:Real
    isempty(p.monomials) ? R(0) : sum( m(x) for m in p.monomials )
end

import Base: *, /
@doc "Multiply a `p` by scalar `λ`, i.e. scale the `factor`s of all `p.monoials`."
function *( p :: Polynomial{N} where N, λ :: R where R<:Real)
    pλ = deepcopy(p)
    for monomial in pλ.monomials
        monomial.factor *= λ
    end
    return pλ
end
*(λ :: R where R<:Real, p :: Polynomial{N} where N) = p * λ
/(p :: Polynomial{N} where N, λ :: R where R<:Real) = p * (1/λ)

import Base: +, -
#=
# NOTE I suppose this is a bit of an overkill.
# It would suffice to concatenate the arrays `p1.monomials` and `p2.monomials`
@doc """
Add `p1` and `p2` symbolically:
If there is a pair of monomials in `p1.monomials` and `p2.monomials` with the
same exponents (multi-indices), then combine them by adding their factors.
All other monomials are kept without change.
"""
function +(p1 :: Polynomial{N}, p2 :: Polynomial{N} )  where N
    new_monomial_list = Vector{Monomial{N}}();
    p1_accept = ones(Bool, length(p1.monomials) )
    p2_accept = ones(Bool, length(p2.monomials) )
    for (m1_index, m1) ∈ enumerate(p1.monomials)
        for (m2_index, m2) ∈ enumerate(p2.monomials)
            if p2_accept[m2_index]
                if m1.α == m2.α
                    new_m = Monomial(
                        factor = m1.factor + m2.factor,
                        α = m1.α
                    )
                    push!(new_monomial_list, new_m)
                    p1_accept[m1_index] = p2_accept[m2_index] = false
                    break;
                end
            end
        end
    end
    push!( new_monomial_list, p1.monomials[ p1_accept ]... )
    push!( new_monomial_list, p2.monomials[ p2_accept ]... )
    return Polynomial( new_monomial_list )
end
=#
function +(p1 :: Polynomial{N}, p2 :: Polynomial{N} )  where N
    new_monomial_list = [ p1.monomials; p2.monomials ];
    return Polynomial( new_monomial_list );
end
-(p1 :: Polynomial{N} where N, p2 :: Polynomial{N} where N ) = p1 + (p2 * (-1))


@doc "Evaluate and return the gradient of `p` at site `x`."
function gradient( p :: Polynomial{N} where N, x :: Vector{R} where R<:Real )
    sum( gradient(m, x) for m ∈ p.monomials )
end

@doc "Evaluate and return the hessian of `p` at site `x`."
function hessian( p :: Polynomial{N} where N, x :: Vector{R} where R<:Real )
    sum( hessian(m, x) for m ∈ p.monomials )
end

# helper function
@doc """
Return array of solution vectors [x_1, …, x_len] to the equation
``x_1 + … + x_len = rhs``
where the variables must be non-negative integers.
"""
function non_negative_solutions( rhs :: Int64, len :: Int64 )
    if len == 1
        return rhs
    else
        solutions = [];
        for i = 0 : rhs
            for shorter_solution ∈ non_negative_solutions( i, len - 1)
                push!( solutions, [ rhs-i; shorter_solution ] )
            end
        end
        return solutions
    end
end

end

#=
# POOR MAN'S UNIT TESTING
using .Polynomials

p1 = Morbit.Polynomial(1/3, (3, 0))    # x_1^2
p2 = Polynomial(2.0, (1, 1))    # 2x_1x_2

# ∇p1 = [ x_1^2; 0 ]
# Hp3 =
#       | x_1   0 |
#   2.* | 0     0 |

# ∇p2 = 2 .* [ x_2; x_1 ]
# Hp2 =
#          | 0     1  |
#   2 .*   | 1     0  |


x1 = ones(2)
x2 = [1;0]
x3 = [0;1]

@show p1(x1)
@show p2(x1)
@show p2(x2)

@show gradient(p1, x1)
@show gradient(p2, x1)
@show hessian(p2, x1)
=#
