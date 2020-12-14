import os
import datetime
import subprocess

def getdata(nn_name):
    if nn_name == "dnn":
        file_path = os.path.join(os.path.dirname(__file__), f'../dnn/main.py')
    elif nn_name == "cnn":
        file_path = os.path.join(os.path.dirname(__file__), f'../cnn/main.py')
    else:
        pass
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
    with open(file_path, encoding="utf-8") as f:
        s = f.read()
    commitid, branchname = gethash()
    return s, timestamp, commitid, branchname

def gethash():
    cmd = "git rev-parse --short HEAD"
    commitid = subprocess.check_output(cmd.split()).strip().decode('utf-8')
    cmd = "git rev-parse --abbrev-ref HEAD"
    branch_name = subprocess.check_output(cmd.split()).strip().decode('utf-8')
    return commitid, branch_name

def docmaker(main_data, timestamp, log, commitid, branch_name):
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
<h3>git data</h3>
<p>branch name : {branch_name}</p>
<p>base commitid : {commitid}</p>
<h3>Loss func-ish graph</h3>
<p><img src="{timestamp}_cost.png"></p>
<h3>Accurancy rate graph</h3>
<p><img src="{timestamp}_accurancy.png"></p>
<h3>kernelmove</h3>
<p><img src="{timestamp}_kernelmove.png"></p>
<h3>kernelmax&min</h3>
<p><img src="{timestamp}_kernelmax&min.png"></p>
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
    with open(path, mode='w',encoding="utf-8") as f:
        f.write(result)
