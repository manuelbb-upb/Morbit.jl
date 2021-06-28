using Morbit
using Test
using Logging

@testset "Save & Load AbstractConfig" begin
    
    # AbstractConfig 
    empty_cfg = Morbit.DefaultConfig();
    fn1 = string( tempname(), ".jld2" )
    pt1 = save_config( fn1, empty_cfg )

    @test fn1 == pt1 
    @test isfile(pt1)

    loaded_empty_cfg = load_config( fn1 )
    @test empty_cfg == loaded_empty_cfg

    algo_cfg = AlgorithmConfig();
    fn2 = tempname()
    pt2 = save_config( fn2, algo_cfg )

    @test string(fn2, ".jld2") == pt2 
    @test isfile(pt2)

    loaded_algo_config = load_config( pt2 )
    CFG_SAME = true
    for field in fieldnames( AlgorithmConfig )
        if getfield( algo_cfg, field ) ≠ getfield( loaded_algo_config, field )
            CFG_SAME = false 
        end
    end
    @test CFG_SAME
    
    # try to load non-existent file 
    # use NullLogger to suppress threatening messages
    with_logger(NullLogger()) do 
        @test isnothing(load_config(fn2))
    end
end

@testset "Save & Load Database" begin
    
    # AbstractConfig 
    empty_db = Morbit.NoDB();
    fn1 = string( tempname(), ".jld2" )
    pt1 = save_database( fn1, empty_db )

    @test fn1 == pt1 
    @test isfile(pt1)

    loaded_empty_db = load_database( pt1 )
    @test empty_db == loaded_empty_db

    fn2 = tempname()
    nonempty_db = Morbit.ArrayDB();
    Morbit.add_result!(nonempty_db,
        Morbit.init_res(
            Morbit.Res,
            rand(2), 
            rand(3)
        )
    )
    pt2 = save_database(fn2, nonempty_db)
    @test string(fn2, ".jld2") == pt2 
    @test isfile(pt2)

    loaded_nonempty_db = load_database( pt2 )

    DB_SAME = true
    for field in fieldnames( Morbit.ArrayDB )
        if getfield( nonempty_db, field ) ≠ getfield( loaded_nonempty_db, field )
            @show getfield(nonempty_db, field)
            @show getfield(loaded_nonempty_db, field)
            DB_SAME = false 
            break;
        end
    end
    @test DB_SAME
end

# TODO test save / load of AbstractIterData