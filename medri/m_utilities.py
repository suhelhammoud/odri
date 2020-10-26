from log_settings import lg
import numpy as np
from numpy import linalg as LA
# from medri.utils import *
from medri.arffreader import loadarff

from scipy.stats import entropy
import matplotlib.pyplot as plt

from medri.m_attribute import MAttribute
from medri.m_counter import MCounter
from medri.m_instances import Instances


def labels_of_item(lines, all_labels, num_labels):
    labels = all_labels[lines]
    vals, counts = np.unique(labels, return_counts=True)
    result = np.zeros(num_labels, dtype=int)
    result[vals] = counts
    return result


def lines_to_count_labels(lst_lines, all_labels, num_labels):
    return np.array([np.array(labels_of_item(lines, all_labels, num_labels))
                     for lines in lst_lines], dtype=int)


def labels_in_att(
        att: np.array,
        all_labels,
        num_labels,
        lines=None,
        prune_items=True):
    """
    no missing values allowed in items_indexes, use indexes_no_missing
    :param att: np.array one scalar vector, may contain missing data
    :param lines: indexes to be considered out of att values
    :param prune_items: to remove indexes of missing values. TODO check to remove it alter
    :return: (vals, att_lines)
        u_vals: a sorted array of unique items
        att_lines: list of array indices
    """

    items_indexes = np.where(data != MAttribute.MISSING)[0] \
        if lines is None else lines

    if prune_items:
        items_indexes[data[items_indexes] != MAttribute.MISSING]

    # TODO delete later for performance
    if np.any(att[items_indexes] == MAttribute.MISSING):
        raise Exception(f"Missing values are not allowed in indexes\n"
                        f"att[item_indexes = {att[items_indexes]}")

    items = att[items_indexes]

    idx_sort = np.argsort(items)

    sorted_data = items[idx_sort]

    sorted_indexes = items_indexes[idx_sort]

    item, idx__start = np.unique(sorted_data, return_index=True)

    item_lines = np.split(sorted_indexes, idx__start[1:])

    item_labels = lines_to_count_labels(
        lst_lines=item_lines,
        all_labels=all_labels,
        num_labels=num_labels)
    # return item, item_labels, item_lines
    return item, item_labels  # , item_lines


def count_atts_items_labels(inst,
                            available_atts,
                            available_lines=None):
    all_labels = inst.label_data
    num_lables = inst.num_labels()
    if available_lines is None:
        available_lines = np.arange(inst.num_instances())

    r_att_index = []
    r_item = []
    r_labels = []

    for att_index in available_atts:
        att = inst.nominal_data[att_index]

        item, item_labels = labels_in_att(
            att=att,
            lines=available_lines,
            all_labels=all_labels,
            num_labels=num_lables,
            prune_items=False)
        # lg.debug(f'itemIndex = {att_index}')
        # lg.debug(f'item={item}')
        # lg.debug(f'item_labels=\n{item_labels}')
        # lg.debug(f'item_lines={item_lines}')

        # r_att_index.append(att_index)
        # r_att_index.append(att_index)
        r_att_index.append([att_index] * len(item))
        r_item.append(item)
        r_labels.append(item_labels)

    # return np.concatenate(r_att_index), \
    #        np.concatenate(r_item), \
    #        np.vstack(r_labels)
    return MCounter(r_att_index,
                    r_item,
                    r_labels)
    # return r_att_index, r_item, r_labels


if __name__ == '__main__':
    lg.debug('Starting')
    # filename = '../data/contact-lenses.arff'
    filename = '../data/cl.arff'
    data, meta = loadarff(filename)
    # #
    inst = Instances(*loadarff(filename))
    lg.debug(f'inst = {inst}')
    available_lines = np.arange(inst.num_instances())
    # available_atts = np.array(range(inst.num_nominal_atts()))
    available_atts = np.array([0, 3, 1])
    # m_counter = MCounter(inst.unique_labels, inst.num_values_in_att()[available_atts])

    m_counter = count_atts_items_labels(inst,
                                        available_atts,
                                        available_lines)

    lg.info(m_counter)

    print(m_counter)
    print('end')
    print(m_counter)
    print(m_counter.mutual_info())
    print(f'ranks = {m_counter.w_ranks(min_freq=1)}')
    print(f'best att, item, label = {m_counter.best_att_item_label(min_freq=1)}')
    # print('=====================')
    # print(f'ranks = {m_counter.w_ranks(min_freq=6, weight=np.array([2,1,1]))}')
    # ax2, itms2, lbls2 = labels_in_atts(
    #     inst,
    #     available_atts,
    #     available_lines)
    # print(f'ax2={ax2}')
    # print(f'itms2={itms2}')
    # print(f'lbls2 = {lbls2}')
