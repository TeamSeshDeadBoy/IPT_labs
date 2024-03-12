function loadXMLDoc(filename) {
    if (window.ActiveXObject) {
         xhttp = new ActiveXObject("Msxml2.XMLHTTP");
    } else {
         xhttp = new XMLHttpRequest();
    }
    xhttp.open("GET", filename, false);
    xhttp.send("");
    return xhttp.responseXML;
 }

 
 xml = loadXMLDoc("objects.xml");
 //  xsl = loadXMLDoc("objects_en.xsl");
 var render = function(xsl) {
      if (document.implementation && document.implementation.createDocument) {
           document.getElementById('container').innerHTML = '';
           xsltProcessor = new XSLTProcessor();
           xsltProcessor.importStylesheet(xsl);
           resultDocument = xsltProcessor.transformToFragment(xml, document);
           document.getElementById('container').appendChild(resultDocument);
          }
}


let flag = false;
let parser = loadXMLDoc("objects_en.xsl");
render(parser)
     
var Boo = function(){
          console.log('changed')
          if (flag){
               flag = false;
               parser = loadXMLDoc("objects_en.xsl");
          } else {
               flag = true;
               parser = loadXMLDoc("objects_ru.xsl");
          }
          render(parser)
     }

document.getElementById("a").onclick = Boo;