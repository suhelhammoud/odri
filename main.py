import numpy as np
from medri.utils import *
from medri.m_instances import Instances
from log_settings import lg



if __name__ == '__main__':
    lg.debug('wwwwwwwwwwwwwww')
    filename = 'data/contact-lenses.arff'
    data, meta = loadarff(filename)
    # #
    inst = Instances(*loadarff(filename))

    available_lines = np.array(range(inst.num_lines()))

    # att, item, item_lines, m_entropy = \
    result = find_best_item_att(inst,
                                available_lines,
                                min_freq=1,
                                labels_weights=None)

    print(f'att={att}\n'
          f'item={item}\n'
          f'item_lines={item_lines}\n'
          f'm_entropy={m_entropy}')

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
