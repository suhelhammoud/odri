import numpy as np


class MAttribute:
    MISSING = -1

    def __init__(self, name, att):
        """

        :param name:
        :param att: att[0] = nominal, att[1] = dict{name: id}
        """
        self.name = name
        self.data = {v: idx for idx, v in enumerate(att[1])}
        self.num_items = len(self.data)

    def map_one(self, name):
        return self.data[name] if name in self.data else MAttribute.MISSING

    def map_all(self, a):
        # return np.array([self.map_one(i) for i in a], dtype=np.int)
        return np.array(list(map(self.map_one, a)), dtype=np.int)

    def __repr__(self):
        return f'MAttribute <name: {self.name} = {str(self.data)}>'
