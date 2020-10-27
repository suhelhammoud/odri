from unittest import TestCase
import numpy as np


class Test(TestCase):
    def test_sliding_window(self):
        from medri.m_utils import sliding_window
        result = sliding_window(np.array([1, 2, 3, 4, 5]), 3)
        np.testing.assert_array_equal(result[0], np.array([1, 2, 3]))
        np.testing.assert_array_equal(result[1], np.array([2, 3, 4]))
        np.testing.assert_array_equal(result[2], np.array([3, 4, 5]))
