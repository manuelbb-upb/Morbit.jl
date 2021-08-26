
using Pkg
current_env = Base.load_path()[1]
Pkg.activate(@__DIR__)

using Literate
using PlutoSliderServer

example_dir = joinpath(@__DIR__, "..", "examples")
src_dir = joinpath(@__DIR__, "..", "src")

function replace_comments( content )
    content = replace(content, r"^(\h*)#~(.*)$"m => s"\1## \2")
    return content
end    


#%% Functions to convert notebook
function make_notebook_html( notebook_jl_path; execute = true )
    out_dir = joinpath(@__DIR__, "src", "custom_assets") 
    if execute
        PlutoSliderServer.export_notebook( notebook_jl_path; Export_output_dir = out_dir )
    end
    base_name = splitpath(splitext( notebook_jl_path )[1])[end]
    return joinpath( out_dir, base_name * ".html" )
end

function modify_notebook_html( html_name )
    content = open( html_name, "r" ) do html_file 
        read( html_file, String )
    end
    content = replace(content, 
        r"(<script\b[^>]*>[\s\S]*?<\/script>)" => 
        s"\1\n\t<script src='./iframeResizer.contentWindow.min.js'></script>\n";
        count = 1
    )
    open( html_name, "w" ) do html_file
        write( html_file, content )
    end
    return nothing
end

const md_nb_template = """
```@raw html
<iframe id="fdnotebook" src="../custom_assets/HTML_NAME" width="100%"></iframe>
<!--<script src="../custom_assets/iframeResizer.min.js"></srcipt>-->
<script>
const iFrameResizerPath = '../custom_assets/iframeResizer.min.js';

if (require) {
  require([iFrameResizerPath], (iFrameResize) => iFrameResize())
} else {
  const script = document.createElement('script')
  script.onload = () => iFrameResize()
  script.src = iFrameResizerPath
}
</script>
<script>
document.addEventListener('DOMContentLoaded', function(){
	var myIframe = document.getElementById("fdnotebook");
	iFrameResize({log:true}, myIframe);	
});
</script>
```
"""

function create_md_nb_file( html_name )
    global md_nb_template;
    html_filename = splitpath(html_name)[end]    
    content = replace( md_nb_template, "HTML_NAME" => html_filename)
    md_path = joinpath(@__DIR__, "src", splitext( html_filename )[1] * ".md" ) 
    open(md_path, "w") do md_file
        write(md_file, content)
    end
    return md_path
end

function make_notebook_md( notebook_jl_path; execute = true )
    html_path = make_notebook_html( notebook_jl_path; execute )
    modify_notebook_html( html_path )
    return create_md_nb_file( html_path )
end

#%%
Literate.markdown(
    joinpath( example_dir, "example_two_parabolas.jl"), 
    joinpath( @__DIR__, "src" );    
    preprocess = replace_comments,
    codefence = "````julia" => "````",  # disables execution; REMOVE when it works again
    )

Literate.markdown(
    joinpath( example_dir, "example_zdt.jl"), 
    joinpath( @__DIR__, "src" );   
    codefence = "````julia" => "````", # disables execution; REMOVE when it works again
    preprocess = replace_comments
    )

Literate.markdown(
    joinpath( src_dir, "TaylorModel.jl"), 
    joinpath( @__DIR__, "src" );    
    codefence = "````julia" => "````",
)


Literate.markdown(
    joinpath( src_dir, "RbfModel.jl"), 
    joinpath( @__DIR__, "src" );    
    codefence = "````julia" => "````",
)

Literate.markdown(
    joinpath( src_dir, "custom_logging.jl"), 
    joinpath( @__DIR__, "src" );  
    codefence = "````julia" => "````",
)

#%%
#make_notebook_md( joinpath( example_dir, "notebook_finite_differences.jl" ); execute = true )

#%%
Pkg.activate(current_env)