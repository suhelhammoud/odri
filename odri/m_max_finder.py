import sys
import numpy as np


class MaxFinder:
    def __init__(self,
                 rank=-sys.maxsize,
                 att_index=None,
                 item_index=None,
                 label_index=None):
        self.rank = rank
        self.att_index = att_index
        self.item_index = item_index
        self.label_index = label_index
        # From MRule # TODO reduce properties
        # self.__label_index = label_index
        self.att_indexes = []
        self.att_values = []
        self.cover = 0
        self.correct = 0
        self.errors = 0

    def get_label(self):
        return self.label_index

    def w_rank(self, w):
        if w is not None:
            self.rank *= w

    def of_weight(self, w):
        if w is not None:
            self.rank = self.rank * w
        return self

    def max(self, that):
        if self.rank < that.rank:
            return that
        elif self.rank > that.rank:
            return self
        elif self.cover < that.cover:  # TODO what about self.correct? later...
            return that
        else:
            return self

    def __len__(self):
        return len(self.att_indexes)

    def contains_att(self, att_index):
        return att_index in self.att_indexes

    def add_test(self, att_index=None, item=None):
        if self.contains_att(att_index):
            return False
        if att_index is None:
            self.att_indexes.append(self.att_index)
            self.att_values.append(self.item_index)
        else:
            self.att_indexes.append(att_index)
            self.att_values.append(item)
        return True

    def update_errors(self, correct, cover):
        # assert self.__label == mx.label_index
        self.cover = cover
        self.correct = correct
        self.errors = self.cover - self.correct

    def can_cover_instance(self, cond: np.array):
        if len(self.att_indexes) == 0:
            return False
        return np.array(cond[self.att_indexes], self.att_values)

    def classify(self, cond: np.array):
        return self.label_index \
            if self.can_cover_instance(cond) \
            else None

    def __repr__(self):
        return f"""MaxFinder: rank = {self.rank}
        (att, item, label)  = ({self.att_index}, {self.item_index}, {self.label_index})
        (cover,correct,errors)  = ({self.cover},{self.correct},{self.errors})
        atts_indexes = {self.att_indexes}
        att_values   = {self.att_values}
        """

    def __ior__(self, other):
        return self.max(other)


if __name__ == '__main__':
    m = MaxFinder()
    x = MaxFinder(rank=3, att_index=7, item_index=77, label_index=8)
    # m |= x
    x |= m
    x = x.max(m)
    print(x)
