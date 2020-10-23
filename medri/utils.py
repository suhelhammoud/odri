import numpy as np
from medri.arffreader import loadarff
from numpy import linalg as LA

from scipy.stats import entropy
import matplotlib.pyplot as plt

from medri.m_attribute import MAttribute
from medri.m_instances import Instances





# # @np.array_function_dispatch(_unique_dispatcher)
# def count_labels(item_lines, all_labels):
#     labels = all_labels[item_lines]
#     return_index = False
#     return_inverse = False
#     axis = None
#     return_counts = False
#
#     labels = np.asanyarray(labels)
#     # ret = _unique1d(ar, return_index, return_inverse, return_counts)
#     # return _unpack_tuple(ret)
#     #
#     #
#     # def _unique1d(ar, return_index=False, return_inverse=False,
#     #               return_counts=False):
#     """
#     Find the unique elements of an array, ignoring shape.
#     """
#     labels = np.asanyarray(labels).flatten()
#
#     labels.sort()
#     aux = labels
#     mask = np.empty(aux.shape, dtype=np.bool_)
#     mask[:1] = True
#     mask[1:] = aux[1:] != aux[:-1]
#
#     ret = (aux[mask],)
#
#     # if return_counts:
#     idx = np.concatenate(np.nonzero(mask) + ([mask.size],))
#     ret += (np.diff(idx),)
#     return (ret,)


#
# def extract_items_lines_in_attribute(data_att_values, MISSING=-1):
#     """
#
#     :param data_att_values: np.array one scalar vector, may contain missing data
#     :param MISSING: int value for missing
#     :return: (vals, att_lines)
#         u_vals: a sorted array of unique items
#         att_lines: list of array indices
#     """
#
#     idx_sort = np.argsort(data_att_values)
#     sorted_data = data_att_values[idx_sort]
#     att_vals, idx__start = np.unique(sorted_data, return_index=True)
#     att_lines = np.split(idx_sort, idx__start[1:])
#
#     missing_index = np.where(att_vals == MISSING)[0]  # first value in case of -1
#     if missing_index.size != 0:
#         index = missing_index[0]
#         att_vals = np.delete(att_vals, index)
#         att_lines.pop(index)
#     return att_vals, att_lines

#
# def test_extract_items_lines_in_attribute():
#     data = np.array([11, 22, 33, 11, 11, -1, 33, 44, 33, 22], dtype=int)
#     print(f'data.dtype = {data.dtype}')
#     vals, res = extract_items_lines_in_attribute(data)
#     print(f'vals = {vals}')
#     for i in res:
#         print(i, i.dtype)


def extract_items_indexes_in_attribute(
        att: np.array,
        indexes=None,
        prune_items=True):
    """
    no missing values allowed in items_indexes, use indexes_no_missing
    :param att: np.array one scalar vector, may contain missing data
    :param indexes: indexes to be considered out of att values
    :param prune_items: to remove indexes of missing values. TODO check to remove it alter
    :return: (vals, att_lines)
        u_vals: a sorted array of unique items
        att_lines: list of array indices
    """
    items_indexes = indexes_no_missing(att) if indexes is None else indexes

    if prune_items:
        print(f'data = {att[items_indexes]}')
        items_indexes = prune_indexes_from_missing(att, items_indexes)
        print(f'data pruned= {att[items_indexes]}')
        print(f'indexes pruned= {items_indexes}')

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
    return item, item_lines


def test_extract_items_lines_in_attribute():
    att = np.array([11, 22, 44, 11, -1, 11, 33, 33, 33, 22], dtype=int)
    # indexes = indexes_no_missing(att)[::2]
    indexes = np.arange(0, 10, 2, dtype=int)
    print(f'items_indexes = {indexes}')
    print(f'items = {att[indexes]}')
    vals, lines = extract_items_indexes_in_attribute(att, indexes, prune_items=True)

    for i in range(vals.size):
        print(f'index = {i}, val ={vals[i]}, lines={lines[i]}, actual ={att[lines[i]]}')
        if np.any(att[lines[i]] != vals[i]):
            raise Exception(f'Error at val={vals[i]}, lines={lines}, actual={att[lines[i]]}')


def indexes_no_missing(data: np.array):
    return np.where(data != MAttribute.MISSING)[0]


def prune_indexes_from_missing(data: np.array, indexes: np.array):
    return indexes[data[indexes] != MAttribute.MISSING]


def map_labels(lines, all_labels, num_labels=None):
    labels = all_labels[lines]
    vals, counts = np.unique(labels, return_counts=True)
    if num_labels:
        result = np.zeros(num_labels, dtype=int)
        result[vals] = counts
        return result
    else:
        return counts


def test_map_labels():
    all_lines = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    all_labels = np.array([1, 0, 0, 1, 3, 0, 3, 1, 0, 0])

    result = map_labels(all_lines, all_labels, num_labels=4)
    print(result)  # [5 3 0 2]


def get_items_lines(inst: Instances,
                    available_atts: list,
                    lines: np.array = None):
    """

    :param inst:
    :param available_atts:
    :param lines:
    :return: {attribute_index: tuple([items], [[item1_lines], [item2_lines], ... )}
    """
    result = {}
    if lines is None:
        lines = np.arange(inst.num_instances())

    # print(f' lines.size = {lines.size}')
    for att_index in available_atts:
        # print(f'att_index = {att_index}')
        items, items_lines = extract_items_indexes_in_attribute(
            inst.nominal_data[att_index],
            lines
        )

        result[att_index] = (items_lines, items_lines)
    # for  (k,v) in result.items():
    #     print( k, v)
    return result


def ent(a: np.array):
    pa = LA.norm(a, ord=1)
    return -np.sum(pa * np.log2(pa))


def weight_one_entropy(labels: np.array):
    n = LA.norm(labels, ord=1)[np.newaxis]
    p_normalized = labels / n.T
    result = entropy(p_normalized, base=2)
    return result


def m_entropy(labels: np.array):
    n = LA.norm(labels, ord=1, axis=1)[np.newaxis]
    p_normalized = labels / n.T
    result = entropy(p_normalized, base=2, axis=1)
    return result


def test_entropy():
    # entropy vs confidence for each item
    points = np.random.random(size=(500, 3))
    # points = np.array([[2,2,2], [4, 2, 8]])
    n = LA.norm(points, ord=1, axis=1)[np.newaxis]
    p_normalized = points / n.T
    p_max = np.max(p_normalized, axis=1)

    result = entropy(p_normalized, base=2, axis=1)
    x = p_normalized.T[0]
    y = p_normalized.T[1]
    z = result
    ax = plt.axes(projection='3d')
    ax.scatter3D(x, y, z, marker='o', c='red')
    ax.scatter3D(x, y, p_max, marker='^', c='blue')
    ax.set_xlabel('x')
    ax.set_ylabel('Y')
    ax.set_zlabel('z')


def test_get_items_lines():
    filename = '../data/contact-lenses.arff'
    data, meta = loadarff(filename)

    inst = Instances(data, meta)
    available_atts = list(range(inst.num_nominal_atts()))
    result = get_items_lines(inst, available_atts)
    for (k, v) in result.items():
        find_max(k, v, inst.label_data)

    pass


def count_labels_w(lines, all_labels, num_labels, labels_w=None):
    labels = all_labels[lines]
    vals, counts = np.unique(labels, return_counts=True)
    result = np.zeros(num_labels, dtype=int)
    result[vals] = counts
    if labels_w is None:
        return result
    else:
        return np.multiply(result, labels_w)


def map_w_labels(lines, all_labels, lines_w, labels_w, num_labels=None):
    labels = all_labels[lines]
    vals, counts = np.unique(labels, return_counts=True)
    if num_labels:
        result = np.zeros(num_labels, dtype=int)
        result[vals] = counts
        return result
    else:
        return counts


def learn_one_rule(inst: Instances):
    """
    numItemsInAtt
    lines,
    minFreq,
    minConf
    :return: irule, lines_of_irule
    """
    # if lines.size < minFreq return None
    available_atts = set(range(inst.num_nominal_atts()))

    entry_lines = np.arange(inst.num_lines())
    not_covered_lines = None

    while True:
        # count items that survive minFreq and minConf
        # find best next max weighted item

        # generate rule from it

        # available_atts.remove(best.att)

        # get_covered_lines from entryLines

        # accumulate gloval not_covered_lines

        # entry_lines = actual covered lines

        # break if
        # rule.errors ==0
        # or available_atts
        # rule support < minFreq
        # return rule, not_covered_lines
        pass


def find_best_item_att(inst,
                       available_lines,
                       min_freq=1,
                       labels_weights=None):
    num_labels = inst.nominal_attributes_label.sz
    all_labels = inst.label_data
    available_atts = np.array(range(inst.num_nominal_atts()))

    min_w_entropy = 10e50
    min_item = None
    min_item_lines = None
    min_att = None


    for att in available_atts:
        # available_lines = np.array(range(inst.num_lines()))

        items, items_lines = extract_items_indexes_in_attribute(
            inst.nominal_data[att],
            available_lines,
            prune_items=False)

        labels_counts = np.array(
            [count_labels_w(il,
                            all_labels,
                            num_labels,
                            labels_w=labels_weights)
             for il in items_lines])

        print(labels_counts)
        entropies = m_entropy(labels_counts)
        print(entropies)
        min_item_index = np.argmin(entropies)
        print(f'min_item_index = {min_item_index}')

        if True:
            return None

        for item_index in range(len(items)):
            item = items[item_index]
            item_lines = items_lines[item_index]

            label_counts = count_labels_w(item_lines,
                                          all_labels,
                                          num_labels,
                                          labels_weights)
            if np.sum(label_counts) < min_freq:
                continue

            w_entropy = weight_one_entropy(label_counts)
            if w_entropy < min_w_entropy:
                min_w_entropy = w_entropy
                min_item_lines = item_lines
                min_att = att

        return min_att, \
               min_item, \
               min_item_lines, \
               min_w_entropy


def find_best_item(items,
                   items_lines,
                   all_labels,
                   num_labels,
                   min_freq=1,
                   labels_weights=None):
    min_w_entropy = 10e50
    min_item = None
    min_item_lines = None

    for item_index in range(len(items)):
        item = items[item_index]
        item_lines = items_lines[item_index]

        labels_counts = count_labels_w(item_lines,
                                       all_labels,
                                       num_labels,
                                       labels_weights)
        if np.sum(labels_counts) < min_freq:
            continue

        w_entropy = weight_one_entropy(labels_counts)
        if w_entropy < min_w_entropy:
            min_w_entropy = w_entropy
            min_item_lines = item_lines

        return min_item, min_item_lines, min_w_entropy


def build_classifier(inst: Instances,
                     lines: np.array,
                     add_default=None):
    remainin_lines = np.array([])
    rules_result = []
    lines_data_size = lines.size
    while lines.size > 0:
        rule, not_covered_lines = learn_one_rule
        remainin_lines = np.concatenate((remainin_lines, not_covered_lines))
        rules_result += rule
        lines_data_size -= rule.covered_line

    if add_default:
        # default_rule = calc_default(remainin_lines, label_index, num_labels)
        # rules_result += default_rule
        pass

    return rules_result


if __name__ == '__main__':
    pass

    filename = 'data/contact-lenses.arff'
    data, meta = loadarff(filename)
    # #
    inst = Instances(*loadarff(filename))
    rules = []
    while True:
        available_atts = np.array(range(inst.num_nominal_atts()))
        available_lines = np.array(range(inst.num_lines()))

        for att in available_atts:
            items, items_lines = extract_items_indexes_in_attribute(
                inst.nominal_data[att], available_lines, prune_items=False)

    # # available_atts = range(inst.num_nominal_atts())
    # r = get_items_lines(inst, available_atts)
    # # pass

    # filename = '../data/colic.arff'
    #
    # data, meta = loadarff(filename)
    #
    # ins = Instances(data, meta)

    # print(instances.nominal_attributes[4].ids)
    # instances.map_item(0, 0)
    #
    # print(instances.label_data)
    # r = count_labels(instances.nominal_data[0], instances.label_data)

    # v1 = instances.nominal_data[0]
    # print(f'len v1 = {v1.size}')
    # r = np.unique(v1, return_index=True,return_counts=True)
    # print(instances.label_data[np.where(v1 == 0)])
    # print(v1[0])
    # print(instances.nominal_data.shape)
    # print(instances.numeric_data.shape)
    # print(instances.label_data.shape)
