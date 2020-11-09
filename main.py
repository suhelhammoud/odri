from log_settings import lg
from odri.arffreader import loadarff
from odri.m_instances import Instances
from odri.m_params import MParam
from odri.m_utils import build_classifier

if __name__ == '__main__':
    filename = 'data/contact-lenses.arff'
    filename = 'data/tic-tac-toe.arff'
    # filename = 'data/cl.arff'
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
