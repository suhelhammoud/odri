import sys

import numpy as np



if __name__ == '__main__':
    c1 = 1999.0
    v1 = 4000000.0
    c2 = 1.0
    v2 = 2.0

    x = v1 * v1

    for c1 in range(1, 100000000):
        # x = sys.maxsize

        left = c1/v1 + c1/x
        right = c2/v2 + c2/x
        if left > right:
            print(f' x = {x}, left = {left}, right= {right}, c1 = {c1}')
            break
    print('end')