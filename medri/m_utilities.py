from log_settings import lg
import numpy as np
from medri.arffreader import loadarff
from medri.m_instances import Instances
from medri.m_utils import count_atts_items_labels

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
