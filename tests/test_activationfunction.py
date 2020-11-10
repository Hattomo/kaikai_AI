import sys
import unittest

import numpy as np

sys.path.append('./shared')
import activationfunction as af

class Test_Activation_Function(unittest.TestCase):

    def setUp(self):
        # init
        pass

    def tearDown(self):
        # dispose
        pass

    def test_sigmoid(self):
        self.assertEqual(af.sigmoid(-50), np.array(1e-15))
        self.assertEqual(af.sigmoid(50), np.array(1.0 - 1e-15))
        self.assertEqual(af.sigmoid(0), np.array(0.5))

    def test_relu(self):
        self.assertEqual(af.relu(-50), np.array(0))
        self.assertEqual(af.relu(50), np.array(50))
        self.assertEqual(af.relu(0), np.array(0))

    def test_tanh(self):
        self.assertEqual(af.tanh(0), np.array(0))
        self.assertEqual(af.tanh(50), np.array(1))
        self.assertEqual(af.tanh(-50), np.array(-1))

    def test_identity(self):
        self.assertEqual(af.identity(50), np.array(50))
        self.assertEqual(af.identity(-50), np.array(-50))
        self.assertEqual(af.identity(0), np.array(0))
        self.assertEqual(af.identity(10e+5), np.array(1e+5))

    def test_swish(self):
        self.assertEqual(af.swish(0), np.array(0))
        self.assertEqual(af.swish(1e+15), np.array(1e+5))

    def test_elu(self):
        self.assertEqual(af.elu(0), np.array(0))
        self.assertEqual(af.elu(4), np.array(4))
        self.assertEqual(af.elu(-50), np.array(-1))

    def test_diffsigmoid(self):
        self.assertEqual(af.diffsigmoid(0), np.array(0.25))
        self.assertLess(abs(af.diffsigmoid(1) - np.array(0.19661193324148185)), 1e5)

    def test_diffrelu(self):
        self.assertEqual(af.diffrelu(-50), np.array(0))
        self.assertEqual(af.diffrelu(0), np.array(1))
        self.assertEqual(af.diffrelu(50), np.array(1))

    def test_difftanh(self):
        self.assertEqual(af.difftanh(0), np.array(1))
        self.assertLess(abs(af.difftanh(4) - np.array(0)), 1e5)
        self.assertLess(abs(af.difftanh(4) - np.array(0)), 1e5)

    def test_diffidentity(self):
        self.assertEqual(af.diffidentity(50), np.array(1))
        self.assertEqual(af.diffidentity(-50), np.array(1))
        self.assertEqual(af.diffidentity(0), np.array(1))
        self.assertEqual(af.diffidentity(10e+5), np.array(1))

    # def test_diffswish(self):
    #     self.assertEqual(af.diffswish(0), np.array(0.5))
    #     self.assertEqual(af.diffswish(1), np.array(1e+5))

    def test_diffelu(self):
        self.assertEqual(af.diffelu(0), np.array(1))
        self.assertEqual(af.diffelu(4), np.array(1))
        self.assertLess(abs(af.diffelu(-1) - np.array(0.36787944)), 1e5)

if __name__ == "__main__":
    unittest.main()
