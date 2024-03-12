<?xml version="1.0" encoding="UTF-8"?>

<xsl:stylesheet
  version="1.0"
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
>
<xsl:output
              method="html"
              doctype-public="-//W3C//DTD HTML 4.01//EN"
              doctype-system="http://www.w3.org/TR/html4/strict.dtd"
              indent="yes" />

    <xsl:template match="/">
        <html>
            <body>
                <h1>Lebedev Stepan, XML - XSL Transitions</h1>

                <hr />
                <h2>Tensors:</h2>

                <xsl:for-each select="objects/object[@type='tensor']">
                    <h3>Tensor obj.</h3>
                    <p>
                        <xsl:value-of select="description[@lang='en']" />
                    </p>
                    <table border='1'>
                            <tr>
                                <td></td>
                                <xsl:for-each select="dimensions/dimension">
                                    <td class="blck"><xsl:value-of select="@axis" /></td>
                                </xsl:for-each>
                            </tr>
                            <tr>
                                <td>Dimensions:</td>
                                <xsl:for-each select="dimensions/dimension">
                                    <td class="blck"><xsl:value-of select="." /></td>
                                </xsl:for-each>
                            </tr>
                    </table>
                    <br />
                </xsl:for-each>

                <hr />
                <h2>Matrices obj.:</h2>

                <xsl:for-each select="objects/object[@type='matrix']">
                    <h3>Matrix</h3>
                    <p>
                        <xsl:value-of select="description[@lang='en']" />
                    </p>
                    <table border='1'>
                            <tr>
                                <td>Dimensions:</td>
                                <td></td>
                                <td class="blck"><xsl:value-of select="dimension_X" /></td>
                            </tr>
                            <tr>
                                <td></td>
                                <td class="blck"><xsl:value-of select="dimension_Y" /></td>
                                <td class="blck-out"></td>
                            </tr>
                    </table>
                    <br />
                </xsl:for-each>

                <hr />
                <h2>Arrays:</h2>

                <xsl:for-each select="objects/object[@type='array']">
                        <h3>Flat array</h3>
                        <p>
                            <xsl:value-of select="description[@lang='en']" />
                        </p>
                        <table border='1'>
                            <tr>
                                <td>Length:</td>
                                <td class="blck"><xsl:value-of select="length" /></td>
                            </tr>
                        </table>
                        <br />
                </xsl:for-each>
            </body>
        </html>
    </xsl:template>
</xsl:stylesheet>