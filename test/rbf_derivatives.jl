include(joinpath(@__DIR__, "..", "src", "RBFBase.jl"))
using .RBF
using ForwardDiff: gradient, hessian, jacobian     # to automatically differentiate the model outputs for comparison
using Test
import FiniteDiff;
fd = FiniteDiff;
m = RBF;

# RBFModel with 2 input and 2 outputs

f1(x) = 1 .+ sum(x.^2);
f2(x) = sin(sum(x));
f(x) = [f1(x), f2(x)];

sites = vec( [[x;y] for x = range( -2, 2, length = 5),
                    y = range( -2, 2, length = 5) ] )

sites = [rand(2) for i = 1 : 5]

vals_1 = f1.(sites);
vals_2 = f2.(sites);
vals = f.(sites);

shape_parameters = [ 1 ]
kernels = [:multiquadric, :exp, :cubic, :thin_plate_spline]

println("Looping through different RBF settings.")
for poly_deg ∈ [ -1, 0, 1 ]
    for kernel ∈ kernels
        for sp ∈ shape_parameters


            A = 1e-3    # somewhat loose tolerance because of possibly bad conditioning
            println("Kernel $(kernel), sp = $sp, poly_deg = $poly_deg")
            rbf = m.RBFModel(
                training_sites = sites,
                training_values = vals,
                kernel = kernel,
                shape_parameter = sp,
                polynomial_degree = poly_deg,
            );
            m.train!(rbf);

            # Test interpolation properties
            rbf1 = x -> m.output( rbf, 1, x );
            rbf2 = x -> m.output( rbf, 2, x );

            evals_1 = rbf1.( sites )
            @test all( isapprox.( evals_1 , vals_1,  atol=A ) )

            evals_2 = rbf2.( sites )
            @test all( isapprox.( evals_2 , vals_2,  atol=A ) )

            rbf_output = x -> m.output( rbf, x );
            @test rbf_output( sites[1] ) == rbf.function_handle( sites[1] )

            # Test gradient calculations
            d_rbf1 = x -> m.grad( rbf, 1, x);
            d_rbf2 = x -> m.grad( rbf, 2, x);

            fwd_rbf1 = χ -> gradient( x -> m.output( rbf, 1, x ), χ);
            fwd_rbf2 = χ -> gradient(x -> m.output( rbf, 2, x ), χ);

            dvals_1 = d_rbf1.(sites)
            dvals_2 = d_rbf2.(sites)
            fwdvals_1 = fwd_rbf1.(sites)
            fwdvals_2 = fwd_rbf2.(sites)

            @test all( isapprox.( dvals_1, fwdvals_1, atol = A) )
            @test all( isapprox.( dvals_2, fwdvals_2, atol = A) )

            jacobian_handle = x -> m.jac( rbf, x )
            @test jacobian_handle(sites[1]) ≈ [d_rbf1(sites[1])'; d_rbf2(sites[1])']

            # test hessian correctness
            h_rbf1 = x -> m.hess( rbf, 1, x )

            H_fwd1 = χ -> fd.finite_difference_hessian( x -> m.output( rbf, 1, x), χ )

            X = rand(2)
            @test h_rbf1( X ) ≈ H_fwd1( X ) atol = A

        end
    end
end
