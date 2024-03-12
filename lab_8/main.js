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
 xsl = loadXMLDoc("objects.xsl");
 if (document.implementation && document.implementation.createDocument) {
    xsltProcessor = new XSLTProcessor();
    xsltProcessor.importStylesheet(xsl);
    resultDocument = xsltProcessor.transformToFragment(xml, document);
    document.getElementById('container').appendChild(resultDocument);
 }