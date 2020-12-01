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
                remove_log_data(file_name)
                print("delete " + str(file_name))
            except:
                print("file delete failed : " + str(file_name))

def remove_log_data(file_name):
    remove_file(file_name + ".html")
    remove_file(file_name + "_cost.png")
    remove_file(file_name + "_accurancy.png")
    remove_file(file_name + "_kernelmove.png")
    remove_file(file_name + "_kernelmax&min.png")

def remove_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
