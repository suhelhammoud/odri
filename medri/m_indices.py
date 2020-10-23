import numpy as np


class CumIndices:
    def __init__(self, indices):
        idx_sum = np.cumsum(indices)
        self.__base = np.array([0] + list(idx_sum[:-1]))
        # print(f'base = {self.__base}')

        self.__cidx = idx_sum - 1
        # print(f'cidx = {self.__cidx}')

    def att(self, idx):
        return np.searchsorted(self.__cidx, idx)

    def item(self, idx):
        att = self.att(idx)
        return idx - self.__base[att]

    def att_item(self, idx):
        att = np.searchsorted(self.__cidx, idx)
        item = idx - self.__base[att]
        return att, item


if __name__ == '__main__':
    a = [5, 3, 4]
    cs = CumIndices(a)
    for i in range(sum(a)):
        print(i, cs.att_item(i))
