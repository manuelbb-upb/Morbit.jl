using Morbit
F(x) = [ sum(x.^2); exp( sum(x)) ];

function get_data( n, N )
    ub = 4 .* ones(n);
    lb = -ub;
    sites = [ lb .+ (ub .- lb) .* rand(n) for i = 1 : N ]
    vals = F.(sites);
    return sites, vals
end

s,v = get_data( 1, 10 )
m = RBFModel( training_sites = s, training_values = v );
train!(m)
