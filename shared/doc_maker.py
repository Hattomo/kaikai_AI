import os
import datetime

now = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
tab = "&ensp;&ensp;&ensp;&ensp;"
result = f"""
<!DOCTYPE html>
<html>
<head>
<title>🐾 Result</title>
</head>
<body>

<h1>{now}</h1>
<h3>Model Data</h3>
<p>
    time stamp: {datetime.datetime.now()}<br>
    model : <br>
    {tab}loss func : <br>
    {tab}count : <br>

</p>
<p>This is a paragraph.</p>
<h3>Loss func-ish graph</h3>
<p><img src="{now}_cost.png" alt="Loss func-ish graph"></p>
<h3>Accurancy rate graph</h3>
<p><img src="{now}_accurancy.png" alt="Accurancy graph"></p>
<h3>Memo</h3>
<h3>Loss func-ish log</h3>
<h3>Accurancy rate log</h3>
</body>
</html>
"""

#path = os.path.join(os.path.dirname(__file__), f'../out/{now}.html')
path = os.path.join(os.path.dirname(__file__), '../out/cost.html')
with open(path, mode='w') as f:
    f.write(result)
