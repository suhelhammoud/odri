import numpy as np
from numpy import linalg as LA

from scipy.stats import entropy

from medri.m_indices import CumIndices


def v_entropy(labels: np.array):
    n = LA.norm(labels, ord=1)[np.newaxis]
    p_normalized = labels / n.T
    result = entropy(p_normalized, base=2)
    return result


class MCounter:

    def __init__(self, unique_labels, num_values_in_atts):
        self.u_labels = np.array(unique_labels)
        self.atts = []
        self.items = []
        self.labels = []
        self.cIndex = CumIndices(num_values_in_atts)

    WEIGHT = np.array([1.0], dtype=float)

    def __iadd__(self, other):
        self.atts += other[0]
        self.items += other[1]
        self.labels += other[2]
        return self

    def __repr__(self):
        return f'Counter :\n' \
               f'\tu_labels = {self.u_labels}\n' \
               f'\tatts = {self.atts}\n' \
               f'\titems = {self.items}\n' \
               f'\tlabels = {self.labels}'

    def mutual_info(self):
        lable_entropy = v_entropy(self.u_labels)
        # print(f'label_entropy = {lable_entropy}')

        att_label = [np.sum(li, axis=1) for li in self.labels]
        # print(f'att_label = {att_label}')
        att_entrop = [v_entropy(i) for i in att_label]
        # print(f'att_entrop = {att_entrop}')
        j_entropy = [v_entropy(it.flat) for it in self.labels]
        # print(f'j_entropy = {j_entropy}')

        # return np.add(lable_entropy, att_entrop) - np.array(j_entropy)
        return np.subtract(att_entrop, j_entropy) + \
               np.array([lable_entropy])

    def w_ranks(self, min_freq=1, weight=WEIGHT):
        lbl_w = np.multiply(np.vstack(self.labels), weight)
        # add support advantage
        rank = entropy(lbl_w, base=2, axis=1) + 10e-8 / np.amax(lbl_w, axis=1)
        not_passed = np.all(lbl_w < min_freq, axis=1)
        # print(f'passed = {not_passed}')
        not_passed_index = np.where(not_passed)
        print(f'not passed indexes = {not_passed_index}')
        rank[not_passed_index] = 10e10
        return rank

    def best_att_item_label(self, min_freq, weight=WEIGHT):
        lbl_w = np.multiply(np.vstack(self.labels), weight)
        # add support advantage
        rank = entropy(lbl_w, base=2, axis=1) + 10e-8 / np.amax(lbl_w, axis=1)
        not_passed = np.all(lbl_w < min_freq, axis=1)
        # print(f'passed = {not_passed}')
        not_passed_index = np.where(not_passed)
        print(f'not passed indexes = {not_passed_index}')
        rank[not_passed_index] = 10e10
        b_index = np.argmin(rank)
        b_label = np.argmax(lbl_w[b_index])
        b_att, b_item = self.cIndex.att_item(b_index)
        print(f'self.items = {self.items}')
        return self.atts[b_att], self.items[b_att][b_item], b_label
