from log_settings import lg
import numpy as np

from medri.m_attribute import MAttribute
from medri.m_counter import MCounter


def sliding_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def bin_count_labels(lst_lines, all_labels, num_labels):
    return np.array([np.bincount(all_labels[lines], minlength=num_labels)
                     for lines in lst_lines], dtype=int)


def labels_in_att2(
        att: np.array,
        all_labels,
        num_labels,
        num_items,
        lines=None,
        prune_items=True
):
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
