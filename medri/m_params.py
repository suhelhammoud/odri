from log_settings import lg
import numpy as np


class MParam:

    def __init__(self):
        self.labels_weights = np.array([1])
        self.items_weights = np.array([1])
        self.atts_weights = np.array([1])
        self.min_occ = 1

    def __repr__(self):
        return f'MParams[\n' \
               f'\t min_occ ={self.min_occ}\n' \
               f'\t labels_weights ={self.labels_weights}\n' \
               f'\t items_weights ={self.items_weights}\n' \
               f'\t atts_weights ={self.atts_weights}\n'
