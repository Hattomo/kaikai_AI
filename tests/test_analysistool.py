import sys

import numpy as np
import pytest

sys.path.append('./dnn')
import analysistool as atool

def test_draw():
    y = [9, 0, 8, 1, 7, 2, 6, 3, 5, 4]
    atool.draw(y)
