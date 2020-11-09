import math
from scipy.stats import entropy
from log_settings import lg
import numpy as np
from medri.arffreader import loadarff
from medri.m_attribute import MAttribute
from medri.m_counter import MCounter
from medri.m_instances import Instances
from medri.m_max_finder import MaxFinder
from medri.m_params import MParam
from medri.m_rule import MRule


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


def measure_weighted(a,
                     max_label=None,
                     params: MParam = None):
    labels_weight = np.array([1]) if params is None else params.labels_weights
    items_weights = np.array([1]) if params is None else params.items_weights
    min_occ = 1 if params is None else params.min_occ

    max_ent = math.log2(a.shape[1])
    a = weight_labels(a, labels_weight, items_weights)
    if max_label is not None:
        lindx = np.argmax(a, axis=1) != max_label
    else:
        lindx = np.array([False])

    occ = np.sum(a, axis=1)
    e = entropy(a, base=2, axis=1)
    return np.where(np.isnan(e)
                    | (occ < min_occ)
                    | lindx,
                    0, max_ent - e)


def weight_labels(lbl, l_w=np.array([1]), i_w=np.array([1])):
    return lbl * l_w * np.asarray(i_w)[np.newaxis].T


def measure2(a, min_occ=1):
    r = entropy(a, base=2, axis=1)
    occ = np.sum(a, axis=1)
    return np.where(np.isnan(r) | (occ < min_occ), 1000, r)


def best_label(v, index=None, l_w=None):
    v_w = v if l_w is None else v * l_w
    label_index = np.argmax(v_w) if index is None else index
    return label_index, v[label_index], np.sum(v)


def get_max_finder(label_count,
                   att_index,
                   max_label=None,
                   params: MParam = None):
    ranks = measure_weighted(label_count, max_label, params)
    rank = np.amax(ranks)
    max_indices = np.where(ranks == rank)[0]
    max_sums = np.sum(label_count[max_indices], axis=1)

    max_item_index = max_indices[np.argmax(max_sums)]
    # max_item_index = np.argmax(ranks)
    max_item_rank = ranks[max_item_index]
    max_label, correct, cover = best_label(label_count[max_item_index],
                                           max_label,
                                           l_w=params.labels_weights)
    mf = MaxFinder(rank=max_item_rank,
                   att_index=att_index,
                   item_index=max_item_index,
                   label_index=max_label)
    mf.correct = correct
    mf.cover = cover
    mf.errors = cover - correct
    return mf


def count_for_att(all_att,
                  all_labels,
                  num_labels,
                  num_items,
                  lines=None):
    """
    get label counts for all items in one attributes
    :param all_att: np.array whole attribute data
    :param all_labels: np.array whole labels data
    :param num_labels: number of labels
    :param num_items: number of items in this attributes
    :param lines: indexes of lines to count for this attribute
    :return: np.array([[labels count for item1], [lables count for item2, ...]])
    """
    lines = remove_missing_lines(all_att, lines)  # TODO can I remove this ?

    # lg.debug(f'lines ={lines}')
    # lg.debug(f'att ={all_att[lines]}')
    # lg.debug(f'labels ={all_labels[lines]}')

    labels2 = (all_att[lines] * num_labels) + all_labels[lines]
    # [[item0_labels], [item1_labels] ,.... ]
    result = np.array(np.split(
        np.bincount(labels2, minlength=num_labels * num_items),
        num_items))
    return result


def lines_of_item(lines, att, item):
    """
    Get item lines for attribute
    :param lines: subset of all lines indexes
    :param att: all attributes data
    :param item: item index or value
    :return: (subset of lines for item value, number of not covered lines)
    """
    item_indexes = np.where(att[lines] == item)[0]  # index of item
    return lines[item_indexes], len(lines) - len(item_indexes)


def partition_covered_for_label(lines, att, labels, item, label):
    """

    :param lines: available lines
    :param att: att values for available lines len(lines) = len(att) = len(lables)
    :param labels:
    :param item:
    :param label:
    :return: tuple(np.array(covred_lines), np.array(not_covered_lines))
    """
    item_indexes = np.where(att == item)[0]  # index of item
    item_labels = labels[item_indexes]
    covered = item_labels == label
    return lines[item_indexes[covered]], lines[item_indexes[~covered]]


def covered_lines_for_item_label(att,
                                 all_labels,
                                 av_lines,
                                 item,
                                 label):
    item_lines_indexes = np.where(att[av_lines] == item)
    covered_indexes = np.where(all_labels[item_lines_indexes] == label)
    return av_lines[item_lines_indexes][covered_indexes]


def count_step(inst: Instances,
               available_atts,
               available_lines=None,
               params: MParam = None,
               max_label=None):
    """

    :param inst:
    :param available_atts: list of remaining attributes
    :param available_lines:
    :param params:
    :param max_label:
    :return:
    """
    # TODO check to count for max_labels found previously
    if available_lines is None:
        available_lines = np.arange(inst.num_instances())

    m_max = MaxFinder()
    for att_index in available_atts:
        labels_count = count_for_att(
            all_att=inst.nominal_data[att_index],
            all_labels=inst.label_data,
            num_labels=inst.num_items_label,
            num_items=inst.num_items[att_index],
            lines=available_lines)
        # lg.debug(f'att_index = {att_index}')
        # lg.debug(f'labels_count =\n\t {labels_count}')
        # apply weights ?
        t_max = get_max_finder(labels_count,
                               att_index,
                               max_label=max_label,
                               params=params)
        #
        att_w = params.get_att_w(att_index)
        m_max = m_max.max(t_max.of_weight(att_w))
        # lg.debug(f't_max = {t_max}')
        # lg.debug(f'm_max = {m_max}')
        # m_max |= t_max.of_weight(att_w)
    return m_max


def get_one_rule(inst: Instances,
                 available_atts=None,
                 available_lines=None,
                 params: MParam = None) -> (MRule, np.array, int):
    """
    calc step and get one rule from available lines
    :param inst: Instances
    :param available_atts: list
    :param available_lines: np.array
    :param params: MParam
    :return: tuple(MRule, item_lines, covered)
    """
    # lg.debug(f'start with available_lines ={available_lines}')
    # lg.debug(f' data =\n {inst.nominal_data.T[available_lines]}')
    # lg.debug(f' data =\n {inst.data[available_lines]}')
    if len(available_lines) < params.min_occ:
        return None, available_lines, 0

    if available_atts is None:
        available_atts = inst.all_nominal_attributes()

    if len(available_atts) == 0:
        return None, available_lines, 0

    entry_lines = available_lines
    max_label = None
    m_rule = None
    while True:
        lg.debug(f'count_step with entry_lines ={len(entry_lines)}, max_label = {max_label}')
        lg.debug(f'current available_atts ={available_atts}')

        t_max = count_step(inst,
                           available_atts,
                           entry_lines,
                           params,
                           max_label)
        if m_rule is None:  # first iteration
            m_rule = MRule(t_max.label_index)
            max_label = t_max.label_index
            lg.debug(f'firt iteration created rule = {m_rule}')

        # if m_rule.rank >= t_max.rank:  # can not enhance rank
        lg.debug(f'new t_max = {t_max}')
        if m_rule.is_better_than(t_max):
            lg.debug(f'm_rule is better than t_max')
            return m_rule, entry_lines, 0

        # better prediction
        m_rule.rank = t_max.rank
        m_rule.add_test(t_max.att_index, t_max.item_index)
        m_rule.update_errors(t_max.cover, t_max.correct)
        entry_lines, num_not_covered = \
            lines_of_item(entry_lines,
                          inst.nominal_data[t_max.att_index],
                          t_max.item_index)
        available_atts.remove(t_max.att_index)

        if m_rule.errors == 0 \
                or len(available_atts) == 0 \
                or num_not_covered == 0 \
                or m_rule.correct < params.min_occ:
            break
    # TODO check of rule is empty
    remaining_lines = np.setdiff1d(available_lines,
                                   entry_lines,
                                   assume_unique=True)
    return m_rule, remaining_lines, len(available_lines) - len(remaining_lines)


def build_classifier(inst: Instances,
                     params: MParam = None,
                     add_default_rule=False):
    rules = []
    lines = inst.all_lines()
    entry_lines = lines
    atts = inst.all_nominal_attributes()

    num_remaining_lines = len(lines)

    while num_remaining_lines > 0:
        rule, r_lines, covered = get_one_rule(inst,
                                              available_atts=list(atts),
                                              available_lines=entry_lines,
                                              params=params)

        entry_lines = r_lines
        if covered == 0:  # DO NOT remove this
            break
        rules.append(rule)
        lg.debug(f'rule = {rule}')
        lg.debug(f'covered = {covered}')
    if add_default_rule and len(entry_lines) > 0:
        labels = inst.label_data[entry_lines]
        lg.debug(f'add default with lines ={entry_lines}')
        lg.debug(f'add default with labels ={labels}')

        default = get_default_rule(inst.label_data[entry_lines],
                                   num_labels=inst.num_items_label,
                                   labels_weights=None)  # TODO set to params.labels_weights
        rules.append(default)

    assert len(rules) > 0
    lg.debug(f'number of found rules = {len(rules)}')
    return rules


def get_default_rule(labels, num_labels, labels_weights=None):
    lg.debug(f'labels = {labels}')
    count = np.bincount(labels, labels_weights, minlength=num_labels)
    label_index = np.argmax(count)
    rule = MRule(label_index)
    rule.update_errors(np.sum(count), count[label_index])
    return rule


def test_classifier():
    pass


if __name__ == '__main__':
    filename = '../data/contact-lenses.arff'
    filename = '../data/tic-tac-toe.arff'
    # filename = '../data/cl.arff'
    lg.debug(f'Starting with file name = {filename}')
    data, meta = loadarff(filename)
    # #
    inst = Instances(*loadarff(filename))
    lg.debug(f'inst num_lines = {inst.num_lines}')
    lg.debug(f'inst num_items = {inst.num_items}')

    params = MParam()
    rules = build_classifier(inst, params=params, add_default_rule=True)
    for i, rule in enumerate(rules):
        print(f'{i} -> rule = {rule}')
    lg.debug('end of application')
