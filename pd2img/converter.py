import os
import random

import imgkit

css = """
<style type=\"text/css\">
table {
color: #333;
font-family: Helvetica, Arial, sans-serif;
width: 640px;
border-collapse:
collapse; 
border-spacing: 0;
}

td, th {
border: 1px solid transparent; /* No more visible border */
height: 30px;
}

th {
background: #DFDFDF; /* Darken header a bit */
font-weight: bold;
}

td {
background: #FAFAFA;
text-align: center;
}

table tr:nth-child(odd) td{
background-color: white;
}
</style>
"""


def dataframe_to_image(data, outputfile="out.png", format="png"):
    """
    For rendering a Pandas DataFrame as an image.
    data: a pandas DataFrame
    css: a string containing rules for styling the output table. This must
         contain both the opening an closing <style> tags.
    *outputimage: filename for saving of generated image
    *format: output format, as supported by IMGKit. Default is "png"
    """
    fn = str(random.random() * 100000000).split(".")[0] + ".html"
    fn = f'/tmp/{fn}'

    try:
        os.remove(fn)
    except:
        None
    text_file = open(fn, "a")

    # write the CSS
    text_file.write(css)
    # write the HTML-ized Pandas DataFrame
    text_file.write(data.to_html())
    text_file.close()

    # See IMGKit options for full configuration,
    # e.g. cropping of final image
    imgkitoptions = {"format": 'png'}

    imgkit.from_file(fn, outputfile, options=imgkitoptions)
    os.remove(fn)
