import os
import datetime

def docmaker(main_data, timestamp):
    now = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
    main_data = '\n<br>'.join(main_data.splitlines())
    tab = "&ensp;&ensp;&ensp;&ensp;"
    result = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <title>🐾 Result</title>
    </head>
    <body>

    <h1>{now}</h1>
    <h3>Loss func-ish graph</h3>
    <p><img src="{timestamp}_cost.png" alt="Loss func-ish graph"></p>
    <h3>Accurancy rate graph</h3>
    <p><img src="{timestamp}_accurancy.png" alt="Accurancy graph"></p>
    <h3>📝Main Code</h3>
    <p>
    {main_data}
    </p>
    <h3>Loss func-ish log</h3>
    <h3>Accurancy rate log</h3>
    </body>
    </html>
    """

    path = os.path.join(os.path.dirname(__file__), f'../out/{now}.html')
    #path = os.path.join(os.path.dirname(__file__), '../out/cost.html')
    with open(path, mode='w') as f:
        f.write(result)
