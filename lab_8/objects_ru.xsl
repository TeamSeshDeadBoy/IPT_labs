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
                <h1>Лебедев Степан. XSL/XML Преобразования</h1>

                <hr />
                <h2>Тензоры:</h2>

                <xsl:for-each select="objects/object[@type='tensor']">
                    <h3>Тензор-обьект</h3>
                    <p>
                        <xsl:value-of select="description[@lang='ru']" />
                    </p>
                    <table border='1'>
                            <tr>
                                <td></td>
                                <xsl:for-each select="dimensions/dimension">
                                    <td class="blck"><xsl:value-of select="@axis" /></td>
                                </xsl:for-each>
                            </tr>
                            <tr>
                                <td>Размерность:</td>
                                <xsl:for-each select="dimensions/dimension">
                                    <td class="blck"><xsl:value-of select="." /></td>
                                </xsl:for-each>
                            </tr>
                    </table>
                    <br />
                </xsl:for-each>

                <hr />
                <h2>Матрицы:</h2>

                <xsl:for-each select="objects/object[@type='matrix']">
                    <h3>Матрица</h3>
                    <p>
                        <xsl:value-of select="description[@lang='ru']" />
                    </p>
                    <table border='1'>
                            <tr>
                                <td>Размерность:</td>
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
                <h2>Массивы:</h2>

                <xsl:for-each select="objects/object[@type='array']">
                        <h3>Одномерный массив</h3>
                        <p>
                            <xsl:value-of select="description[@lang='ru']" />
                        </p>
                        <table border='1'>
                            <tr>
                                <td>Длина массива:</td>
                                <td class="blck"><xsl:value-of select="length" /></td>
                            </tr>
                        </table>
                        <br />
                </xsl:for-each>
            </body>
        </html>
    </xsl:template>
</xsl:stylesheet>