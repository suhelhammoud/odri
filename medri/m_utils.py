import math
from scipy.stats import entropy
from log_settings import lg
import numpy as np
from medri.arffreader import loadarff
from medri.m_attribute import MAttribute
from medri.m_counter import MCounter
from medri.m_instances import Instances


def sliding_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def bin_count_labels(lst_lines, all_labels, num_labels):
    return np.array([np.bincount(all_labels[lines], minlength=num_labels)
                     for lines in lst_lines], dtype=int)


def remove_missing_lines(att_values, lines=None):
    if lines is None:
        return np.where(att_values != MAttribute.MISSING)[0]
    else:
        return lines[att_values[lines] != MAttribute.MISSING]


def count_get_max(att,
                  all_labels,
                  num_labels,
                  num_items,
                  lines=None,
                  weights=None):
    lines = remove_missing_lines(att, lines)
    labels2 = att[lines] * num_labels + all_labels[lines]
    # [[item0_labels], [item1_labels] ,.... ]
    result = np.array(np.split(
        np.bincount(labels2, minlength=num_labels * num_items),
        num_items))
    
    return result


def labels_in_att2(
        att: np.array,
        all_labels,
        num_labels,
        num_items,
        lines=None,
        prune_items=True):
    items_indexes = np.where(att != MAttribute.MISSING)[0] \
        if lines is None else lines

    if prune_items:
        items_indexes = items_indexes[att[items_indexes] != MAttribute.MISSING]

    labels2 = att[items_indexes] * num_labels + all_labels[items_indexes]
    result = np.array(np.split(
        np.bincount(labels2, minlength=num_labels * num_items),
        num_items))
    return np.arange(num_items), result


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

    items_indexes = np.where(att != MAttribute.MISSING)[0] \
        if lines is None else lines

    if prune_items:
        items_indexes = items_indexes[att[items_indexes] != MAttribute.MISSING]

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

    item_labels = bin_count_labels(
        lst_lines=item_lines,
        all_labels=all_labels,
        num_labels=num_labels)
    # return item, item_labels, item_lines
    return item, item_labels  # TODO consider returning , item_lines #


def count_atts_items_labels(inst,
                            available_atts,
                            available_lines=None):
    if available_lines is None:
        available_lines = np.arange(inst.num_instances())

    r_att_index = []
    r_item = []
    r_labels = []

    num_items = inst.num_items
    for att_index in available_atts:
        att = inst.nominal_data[att_index]
        # item, item_labels = labels_in_att(
        #     att=att,
        #     lines=available_lines,
        #     all_labels=inst.label_data,
        #     num_labels=inst.num_items_label,
        #     prune_items=False)

        r = count_get_max(
            att=att,
            lines=available_lines,
            all_labels=inst.label_data,
            num_labels=inst.num_items_label,
            num_items=num_items[att_index]
        )

        lg.debug(f'count_get max \n{r}')
        if True:
            return

        item, item_labels = labels_in_att2(
            att=att,
            lines=available_lines,
            all_labels=inst.label_data,
            num_labels=inst.num_items_label,
            num_items=num_items[att_index],
            prune_items=False)
        r_att_index.append([att_index] * len(item))
        r_item.append(item)
        r_labels.append(item_labels)

    return MCounter(r_att_index,
                    r_item,
                    r_labels)


def best_item_label_in_att(
        att: np.array,
        all_labels,
        num_labels,
        num_items,
        lines=None,
        prune_items=True):
    items_indexes = np.where(att != MAttribute.MISSING)[0] \
        if lines is None else lines

    if prune_items:
        items_indexes = items_indexes[att[items_indexes] != MAttribute.MISSING]

    labels2 = att[items_indexes] * num_labels + all_labels[items_indexes]
    result = np.array(np.split(
        np.bincount(labels2, minlength=num_labels * num_items),
        num_items))

    return np.arange(num_items), result


def measure_weighted(a, w=None):
    max_ent = math.log2(a.shape[1])
    a = weight_labels(a, w.labels_weight, w.items_weights)
    occ = np.sum(a, axis=1)
    e = entropy(a, base=2, axis=1)
    return np.where(np.isnan(e) | (occ < w.min_occ), 0, max_ent - e)


def weight_labels(lbl, l_w=np.array([1]), i_w=np.array([1])):
    return lbl * l_w * np.asarray(i_w)[np.newaxis].T


def measure2(a, min_occ=1):
    r = entropy(a, base=2, axis=1)
    occ = np.sum(a, axis=1)
    return np.where(np.isnan(r) | (occ < min_occ), 1000, r)


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
    available_atts = np.array([3, 3, 1])
    # m_counter = MCounter(inst.unique_labels, inst.num_values_in_att()[available_atts])

    # find best
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
