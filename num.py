import numpy as np
import matplotlib.pyplot as plt

from scipy.io import arff


if __name__ == '__main__':
    filename = 'data/colic.arff'

    data, meta = arff.loadarff(filename)
    for i in meta:
        print(i.type())
    # print(len(data.dtype))
    # print(data.dtype[6])
    #
    #
    # print(f'data {len(data)}')
