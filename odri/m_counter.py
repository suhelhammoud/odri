from log_settings import lg
import numpy as np
from numpy import linalg as LA
from scipy.stats import entropy
from odri.m_indices import CumIndices


def v_entropy(labels: np.array):
    n = LA.norm(labels, ord=1)[np.newaxis]
    p_normalized = labels / n.T
    result = entropy(p_normalized, base=2)
    return result


# TODO not used anymore, consider deleting all file


class MCounter:
    WEIGHT = np.array([1.0], dtype=float)

    def __init__(self, atts_indexes, att_items, items_labels):
        i_count = [len(i) for i in att_items]
        self.c_index = CumIndices(i_count)
        # self.atts_indexes = np.concatenate([item] * count for item, count in zip(atts_indexes, i_count))
        self.atts_indexes = np.concatenate(atts_indexes)
        self.att_items = np.concatenate(att_items)
        self.items_labels = np.vstack(items_labels)

    def __repr__(self):
        return f'Counter :\n' \
               f'\tc_index = {self.c_index}\n' \
               f'\tatts_indexes = {self.atts_indexes}\n' \
               f'\tatt_items = {self.att_items}\n' \
               f'\titem_labels = {self.items_labels}'

    def mutual_info(self):
        # TODO reimplementing all method
        att_lines = self.c_index.atts_lines()
        atts_labels = [self.items_labels[aline] for aline in att_lines]

        u_labels = [np.sum(alabel, axis=0) for alabel in atts_labels]  # TODO change later for performance
        labels_entropy = np.array([v_entropy(ul) for ul in u_labels])
        lg.debug(f'labels_entropy = {labels_entropy}')

        lg.debug(f'att_lines = {att_lines}')
        # att_label = [np.sum(self.items_labels[a_line], axis=1) for a_line in att_lines]

        for aline, alabel in zip(att_lines, atts_labels):
            lg.debug(f'aline = {aline}, item_lables = {alabel}')

        att_label = [np.sum(alabel, axis=1) for alabel in atts_labels]
        lg.debug(f'att_label = {att_label}')
        att_entrop = [v_entropy(i) for i in att_label]
        lg.debug(f'att_entrop = {att_entrop}')
        j_entropy = [v_entropy(it.flat) for it in atts_labels]
        lg.debug(f'j_entropy = {j_entropy}')
        result = labels_entropy + att_entrop - j_entropy
        lg.debug(f'result = {result}')
        return result
        # return np.subtract(att_entrop, j_entropy) + \
        #        np.array([labels_entropy])

    def w_ranks(self, min_freq=1, weight=WEIGHT):
        # lbl_w = np.multiply(np.vstack(self.labels), weight)
        lbl_w = np.multiply(self.items_labels, weight)
        # add support advantage
        rank = entropy(lbl_w, base=2, axis=1) + 10e-8 / np.amax(lbl_w, axis=1)
        not_passed = np.all(lbl_w < min_freq, axis=1)
        # print(f'passed = {not_passed}')
        not_passed_index = np.where(not_passed)
        print(f'not passed indexes = {not_passed_index}')
        rank[not_passed_index] = 10e10
        return rank

    def best_att_item_label(self, min_freq, weight=WEIGHT):
        lbl_w = np.multiply(self.items_labels, weight)
        # add support advantage
        rank = entropy(lbl_w, base=2, axis=1) + 10e-8 / np.amax(lbl_w, axis=1)
        not_passed = np.all(lbl_w < min_freq, axis=1)
        lg.debug(f'passed = {not_passed}')
        not_passed_index = np.where(not_passed)
        print(f'not passed indexes = {not_passed_index}')
        rank[not_passed_index] = 10e10
        b_index = np.argmin(rank)
        b_label = np.argmax(lbl_w[b_index])
        b_att, b_item = self.c_index.att_item(b_index)
        lg.debug(f'b_att={b_att}, b_item={b_item}, b_label={b_label}')
        lg.debug(f'self.att_items = {self.att_items}')
        return b_att, b_item, b_label

# np.where(v.sum(axis=1) > 0, entropy(v, axis=1, base=2), 7777 )
