import numpy as np


class MAttribute:
    MISSING = -1

    def __init__(self, name, att):
        """

        :param name:
        :param att: att[0] = nominal, att[1] = dict{name: id}
        """
        self.name = name
        self.data = {k: v for v, k in enumerate(att[1])}
        self.sz = len(self.data)
        # print(f'name = {name}, sz = {self.sz}')
        # self.data['?'] = Attribute.MISSING

    def map_one(self, name):
        return MAttribute.MISSING if name == '?' else self.data[name]
        # return self.data[name]

    def map_all(self, a):
        # return np.array([self.map_one(i) for i in a])
        return np.array(list(map(self.map_one, a)), dtype=np.int)

    def map_labels(self, lines, labels):
        lines = np.ones(3)
        np.vstack()

    def __repr__(self):
        return f'AValue <name: {self.name} = {str(self.data)}>'

    # def str_values_from_lines(self, lines):
    #     return self.
