import os
import datetime

def docmaker(main_data, timestamp, log):
    tab = "&ensp;&ensp;&ensp;&ensp;"
    #main_data = '<br>\n'.join(main_data.splitlines())
    #main_data = main_data.replace("    ",str(tab))
    #log = '<br>\n'.join(log.splitlines())
    style = r"""
<style>
h1,h3,p {
    font-family: sans-serif;
}
</style> """

    result = f"""
<!DOCTYPE html>
<html>
<head>
<title>🐾 Result</title>
</head>
<body>

<h1>{timestamp}</h1>
<h3>Loss func-ish graph</h3>
<p><img src="{timestamp}_cost.png" alt="Loss func-ish graph"></p>
<h3>Accurancy rate graph</h3>
<p><img src="{timestamp}_accurancy.png" alt="Accurancy graph"></p>
<h3>📝Main Code</h3>
<pre><code>
{main_data}
</code></pre>
<h3>Terminal log</h3>
<pre><code>
{log}
</code></pre>
</body>
</html>

{style}
"""

    path = os.path.join(os.path.dirname(__file__), f'../out/{timestamp}.html')
    #path = os.path.join(os.path.dirname(__file__), '../out/cost.html')
    with open(path, mode='w') as f:
        f.write(result)
