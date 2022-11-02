import unittest
import numpy as np

from classifier import NMC

# IMPORTANT: to pass the test on github, pythonpath must be updated in python-package.yml
# export PYTHONPATH='src'

class TestNMC(unittest.TestCase):

    def setUp(self):
        self.x = np.zeros(shape=(10, 2))
        self.y = np.zeros(shape=(10,))
        self.clf = NMC()

    def test_fit(self):
        self.clf.fit(self.x, self.y)
        out = self.clf.centroids
        self.assertIsNotNone(out)
        self.assertEqual(out.shape, (1, 2))
        self.assertTrue(out.sum() == 0)