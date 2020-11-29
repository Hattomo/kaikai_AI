import os
import time
import glob
import re

def clean():
    now = time.time()
    path = os.path.join(os.path.dirname(__file__), '../out/**/*')
    # a = glob.glob(path, recursive=True)
    files = [p for p in glob.glob(path, recursive=True) if (os.path.isfile(p) and re.search("20\d{18}.html", p))]
    for file_path in files:
        file_name = os.path.splitext(os.path.abspath(file_path))[0]
        # one week passed since created
        if 3600 < now - os.path.getctime(file_path):
            try:
                os.remove(file_name + ".html")
                os.remove(file_name + "_cost.png")
                os.remove(file_name + "__accurancy.png")
                print("delete " + str(file_name))
            except:
                print("file delete failed : " + str(file_name))
