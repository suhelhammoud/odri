# from log_settings import lg
import numpy as np


class CumIndices:
    def __init__(self, indices):
        self.indices = np.array(indices)
        idx_sum = np.cumsum(indices)
        self.__base = np.array([0] + list(idx_sum[:-1]))
        self.__cidx = idx_sum - 1

    def att(self, idx):
        return np.searchsorted(self.__cidx, idx)

    def item(self, idx):
        att = self.att(idx)
        return idx - self.__base[att]

    def att_item(self, idx):
        att = np.searchsorted(self.__cidx, idx)
        item = idx - self.__base[att]
        return att, item

    def atts_lines(self):
        from odri.m_utils import sliding_window

        base = np.array(list(self.__base) + [sum(self.indices)])
        sw = sliding_window(base, 2)
        result = []
        for i in sw:
            rng = np.arange(start=i[0], stop=i[1])
            result.append(rng)
        return result

    def __repr__(self):
        return f'CumIndices({self.indices})\n' \
               f'\t\tbase = {self.__base}\n' \
               f'\t\tidx = {self.__cidx}\n'
