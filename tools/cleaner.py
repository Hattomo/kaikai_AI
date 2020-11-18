import os
import time
import glob
import re

now = time.time()
path = os.path.join(os.path.dirname(__file__), '../out/**/*')
# a = glob.glob(path, recursive=True)
files = [p for p in glob.glob(path, recursive=True) if (os.path.isfile(p) and re.search("20\d{18}(|_cost|_accurancy).*", p))]
print(files)
for file_name in files:
    # one week passed since created
    if 604800 < now - os.path.getctime(file_name):
        #os.remove(file_name)
        print("delete " + str(file_name))
# new_path = "demo_folder" 
# if not os.path.exists(new_path):
#     os.mkdir(new_path)
