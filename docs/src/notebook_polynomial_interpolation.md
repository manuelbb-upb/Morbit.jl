```@raw html
<iframe id="fdnotebook" src="../custom_assets/notebook_polynomial_interpolation.html" width="100%"></iframe>
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
