import numpy as np
from odri.m_attribute import MAttribute


def _nominal_numeric(_meta):
    names = np.array(_meta.names())
    types = np.array(_meta.types())
    nominal_indexes = np.where(types == 'nominal')[0]
    nominal_names = names[nominal_indexes]
    numeric_indexes = np.where(types == 'numeric')[0]
    numeric_names = names[numeric_indexes]
    return (nominal_indexes,
            nominal_names,
            numeric_indexes,
            numeric_names)


class Instances:
    """
    Wrapper for MetaData and Data objects retrieved by loadarff
    """

    def __init__(self, _data, _meta):
        self.data = _data
        self.meta = _meta

        (self.nominal_indexes,
         self.nominal_names,
         self.numeric_indexes,
         self.numeric_names) = _nominal_numeric(_meta)

        _nominal_attributes = [(name, _meta[name]) for name in self.nominal_names]
        self.nominal_attributes = np.array([MAttribute(n, a) for n, a in _nominal_attributes])
        self.nominal_attributes_label = self.nominal_attributes[-1]
        self.num_items = np.array([att.num_items for att in self.nominal_attributes])
        self.num_items_label = self.nominal_attributes_label.num_items

        self.nominal_data_str = np.array(
            [[line[i] for i in self.nominal_indexes] for line in _data],
            dtype=np.dtype('<U200')  # TODO type should be set automatically
        ).T

        _z = list(zip(self.nominal_attributes, self.nominal_data_str))
        self.nominal_data = \
            np.array([header.map_all(values) for header, values in _z])

        self.numeric_data = np.array(
            [[line[i] for i in self.numeric_indexes] for line in _data],
            dtype=np.float32
        ).T

        self.label_data_str = self.nominal_data_str[-1]
        self.label_data = self.nominal_data[-1]
        self.num_lines = len(self.label_data)  # self.label_data.size

    def all_lines(self):
        return np.arange(self.num_lines)

    def all_nominal_attributes(self):
        return list(range(self.num_nominal_atts()))

    def __repr__(self):
        return '\n'.join([f'Instances: \tnum_nominal={self.nominal_names.size}',
                          f'\tnum_numeric={self.numeric_names.size}',
                          f'\tnum_instances={self.num_lines}'
                          ])

    def _get_nominal_headers(self):
        _nominal_attributes = [(name, self.meta[name]) for name in self.nominal_names]
        return np.array([MAttribute(n, a) for n, a in _nominal_attributes])

    #
    # def num_labels(self):
    #     return self.nominal_attributes_label.num_items

    def num_instances(self):
        return self.label_data.size

    def num_nominal_atts(self):
        """
        :return: number of nominal attributes that does
        not include the label attribute
        """
        return len(self.nominal_indexes) - 1
