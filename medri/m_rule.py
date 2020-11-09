from log_settings import lg
import numpy as np
import sys
from m_max_finder import MaxFinder


class MRule:
    def __init__(self, label):
        self.__label = label
        self.rank = - sys.maxsize
        self.att_indexes = []
        self.att_values = []
        self.cover = 0
        self.correct = 0
        self.errors = 0

    def __len__(self):
        return len(self.att_indexes)

    def containts_att(self, att_index):
        return att_index in self.att_indexes

    def add_test(self, att_index, item):
        if self.containts_att(att_index):
            return False
        self.att_indexes.append(att_index)
        self.att_values.append(item)
        return True

    def update_errors(self, cover, correct):
        # assert self.__label == mx.label_index
        self.correct = correct
        self.cover = cover
        self.errors = self.cover - self.correct

    def is_better_than(self, that):
        if self.rank == that.rank:
            return self.cover > that.cover
        return self.rank > that.rank

    def can_cover_instance(self, cond: np.array):
        if len(self.att_indexes) == 0:
            return True
        return np.array(cond[self.att_indexes], self.att_values)

    def classify(self, cond: np.array):
        return self.__label \
            if self.can_cover_instance(cond) \
            else None

    def __repr__(self):
        return f"""Rule:
        label: {self.__label}
        att  : {self.att_indexes}
        val  : {self.att_values}
        correct: {self.correct}
        covers : {self.cover}
        errors : {self.errors}
        """
