# -*- coding: utf-8 -*-

#
# dataOP of HMM Aligner
# Simon Fraser University
# NLP Lab
#
# This file contains functions that are used in the aligner, which are of data
# data manipulation nature.
#
import unittest
import numpy as np


def keyDiv(x, y):
    """
    This method is no longer used in the actual programme.
    """
    if x.shape[:-1] != y.shape:
        raise RuntimeError("Incorrect size")
    if len(x.shape) == 3:
        for i, j in zip(*y.nonzero()):
            x[i][j] /= y[i][j]
    elif len(x.shape) == 2:
        for i, in zip(*y.nonzero()):
            x[i] /= y[i]
    return x


def extendNumpyArray(array, shape):
    """
    This method extends an array to a shape not smaller in any dimension
    than target shape. The entries created during extension will all have
    zero values.
    @param array: np.array. The original array.
    @param shape: tuple. The target shape.
    @return: np.array. The extended array.
    """
    if not isinstance(array, np.ndarray):
        array = np.zeros(shape)
        return array
    if len(array.shape) != len(shape):
        raise RuntimeError("Array dimensions doesn't match")
    for i in range(len(shape)):
        if array.shape[i] < shape[i]:
            tmp =\
                array.shape[0:i] +\
                (shape[i] - array.shape[i], ) +\
                array.shape[i + 1:]
            array = np.append(array, np.zeros(tmp), axis=i)
    return array


class TestModelBase(unittest.TestCase):
    def testExtendNumpyArray(self):
        arrayA = np.array(range(20)).reshape((4, 5))
        arrayAExtended = np.append(
            np.append(
                arrayA,
                np.zeros((2, 5)),
                axis=0
            ),
            np.zeros((6, 3)),
            axis=1
        )
        np.testing.assert_array_equal(
            arrayAExtended,
            extendNumpyArray(arrayA, (6, 8)))
        return

    def testKeyDiv3D(self):
        import math
        n = 3
        m = 4
        h = 5
        x = np.arange(n * m * h).reshape((n, m, h)) + 1
        y = np.arange(n * m).reshape(n, m) + 1
        with np.errstate(invalid='ignore', divide='ignore'):
            correct = np.array([[[x[i][j][k] / y[i][j] for k in range(h)]
                                 for j in range(m)]
                                for i in range(n)])
        result = keyDiv(x, y)
        for i in range(n):
            for j in range(m):
                for k in range(h):
                    self.assertFalse(math.isnan(result[i][j][k]))
                    self.assertEqual(result[i][j][k], correct[i][j][k])
        return

    def testKeyDiv2D(self):
        import math
        n = 3
        m = 4
        x = np.arange(n * m).reshape((n, m)) + 1
        y = np.arange(n) + 1
        with np.errstate(invalid='ignore', divide='ignore'):
            correct = np.array([[x[i][j] / y[i]
                                 for j in range(m)]
                                for i in range(n)])
        result = keyDiv(x, y)
        for i in range(n):
            for j in range(m):
                    self.assertFalse(math.isnan(result[i][j]))
                    self.assertEqual(result[i][j], correct[i][j])
        return


if __name__ == '__main__':
    unittest.main()
