from unittest import TestCase

from medri.m_indices import CumIndices
import numpy as np


class TestCumIndices(TestCase):
    def test_atts_lines(self):
        cs = CumIndices([5, 3, 4])
        att_lines = cs.atts_lines()
        np.testing.assert_array_equal(att_lines[0], np.array([0, 1, 2, 3, 4]))
        np.testing.assert_array_equal(att_lines[1], np.array([5, 6, 7]))
        np.testing.assert_array_equal(att_lines[2], np.array([8, 9, 10, 11]))


    def test_att(self):
        indx = [5, 3, 4]
        cs = CumIndices(indx)

        for i in range(5):
            att = cs.att(i)
            self.assertEqual(0, att)

        for i in range(5, 8):
            self.assertEqual(1, cs.att(i))

        for i in range(8, 12):
            self.assertEqual(2, cs.att(i))

    def test_item(self):
        indx = [5, 3, 4]
        cs = CumIndices(indx)

        for i in range(5):
            item = cs.item(i)
            self.assertEqual(i, item)

        for i, v in enumerate(range(5, 8)):
            self.assertEqual(i, cs.item(v))

        for i, v in enumerate(range(8, 12)):
            self.assertEqual(i, cs.item(v))
