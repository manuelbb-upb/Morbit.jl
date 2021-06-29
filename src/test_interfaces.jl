
using Printf: @sprintf

# steepest descent and directed search
using LinearAlgebra: norm, pinv
import JuMP;
import OSQP;

# LagrangeModels and PS
import NLopt;

using Parameters: @with_kw, @unpack, @pack!
using MathOptInterface;
const MOI = MathOptInterface;

import ThreadSafeDicts: ThreadSafeDict;
using Memoize;

import UUIDs;

import FiniteDiff#erences
const FD = FiniteDiff#erences

import ForwardDiff
const AD = ForwardDiff

import Logging: LogLevel, @logmsg
import Logging

include("custom_logging.jl")

include("shorthands.jl");
include("Interfaces.jl");
include("diff_wrappers.jl");

# implementations (order should not matter)
include("VectorObjectiveFunction.jl");
include("MixedMOP.jl");

include("ResultImplementation.jl")
include("DataBaseImplementation.jl")
include("IterDataImplementation.jl")
include("SurrogatesImplementation.jl");

include("ConfigImplementations.jl")

# utilities
include("adding_objectives.jl");
include("descent.jl")
include("saving.jl")
include("utilities.jl")