import sys

import numpy as np

sys.path.append('./shared')
import activationfunction as af

def test_sigmoid():
    data = np.array([-50, 50, 0])
    label = np.array([1e-15, 1.0 - 1e-15, 0.5])
    assert np.allclose(af.sigmoid(data), label), "Sigmoid is test error"

def test_relu():
    data = np.array([-50, 50, 0])
    label = np.array([0, 50, 0])
    assert np.allclose(af.relu(data), label), "ReLU is test error"

def test_tanh():
    data = np.array([-50, 50, 0])
    label = np.array([-1, 1, 0])
    assert np.allclose(af.tanh(data), label), "tanh is test error"

def test_identity():
    data = np.array([-50, 50, 0])
    label = np.array([-50, 50, 0])
    assert np.allclose(af.identity(data), label), "Identity is test error"

def test_swish():
    data = np.array([-1, 0, 1])
    label = np.array([-0.2689414213699951, 0, 0.7310585786300049])
    assert np.allclose(af.swish(data), label), "Swish is test error"

def test_elu():
    data = np.array([0, 4, -50])
    label = np.array([0, 4, -1])
    assert np.allclose(af.elu(data), label), "ELU is test error"

def test_diffsigmoid():
    data = np.array([0, 1])
    label = np.array([0.25, 0.19661193324148185])
    assert np.allclose(af.diffsigmoid(data), label), "diffsigmoid is test error"

def test_diffrelu():
    data = np.array([-50, 50, 0])
    label = np.array([0, 1, 1])
    assert np.allclose(af.diffrelu(data), label), "diffrelu is test error"

def test_difftanh():
    data = np.array([-0.3, 0, 0.9])
    label = np.array([0.9151369618266293, 1, 0.48691736114834144])
    assert np.allclose(af.difftanh(data), label), "difftanh is test error"

def test_diffidentity():
    data = np.array([-50, 50, 0])
    label = np.array([1, 1, 1])
    assert np.allclose(af.diffidentity(data), label), "diffidentity is test error"

def test_diffswish():
    data = np.array([-4, 0, 4])
    label = np.array([-0.05266461, 0.5, 1.05266461])
    assert np.allclose(af.diffswish(data), label), "diffswish is test error"

def test_diffelu():
    data = np.array([-1, 0, 4])
    label = np.array([0.36787944, 1, 1])
    assert np.allclose(af.diffelu(data), label), "diffelu is test error"
