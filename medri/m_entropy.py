import math

import numpy as np
from scipy.special import (comb, chndtr, entr, rel_entr, xlogy, ive)
from scipy.stats import entropy


def measure_weighted(a, w=None):
    max_ent = math.log2(a.shape[1])
    e = entropy(a, base=2, axis=1)
    return np.where(np.isnan(e), 0, max_ent - e)


def entsuhel(a: list):
    max = math.log2(len(a))
    print(f'max entropy = {max}')
    w = sum(a)
    if w == 0:
        return 0
    b = [(i + 0.0) / w for i in a]
    result = 0
    for i in b:
        if i < 1e-8:
            continue
        result -= i * math.log2(i)
    print(f'entropy = {result}')
    return max-result


if __name__ == '__main__':
    a = np.array([[8, 4, 2, 2],
                  [0, 0, 1, 20],
                  [0, 0, 2, 40],
                  [0, 1, 1, 48],
                  [0, 0, 0, 1],
                  [8, 8, 8, 7],
                  [0, 0, 0, 0]])

    print(f'result = {r}')
    sm = np.sum(a, axis=1)
    print(f'* = {r * sm}')
