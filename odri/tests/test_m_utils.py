from unittest import TestCase
import numpy as np
from math import log2


def entropy_simple(a: list):
    max = log2(len(a))
    print(f'max entropy = {max}')
    w = sum(a)
    if w == 0:
        return 0
    b = [(i + 0.0) / w for i in a]
    result = 0
    for i in b:
        if i < 1e-8:
            continue
        result -= i * log2(i)
    print(f'entropy = {result}')
    return max - result


class Test(TestCase):
    def test_sliding_window(self):
        from odri.m_utils import sliding_window
        result = sliding_window(np.array([1, 2, 3, 4, 5]), 3)
        np.testing.assert_array_equal(result[0], np.array([1, 2, 3]))
        np.testing.assert_array_equal(result[1], np.array([2, 3, 4]))
        np.testing.assert_array_equal(result[2], np.array([3, 4, 5]))
