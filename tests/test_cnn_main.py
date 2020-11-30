import sys
import subprocess

import numpy as np
import pytest

sys.path.append("./dataset")
sys.path.append('./dnn')
sys.path.append("./shared")

def test_main():
    main_result = subprocess.run(('python', 'cnn/main.py'))
    assert main_result.returncode == 0