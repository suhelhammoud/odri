# import numpy as np
# from numpy import linalg as LA
#
# from scipy.stats import entropy
#
#
# def v_entropy(labels: np.array):
#     n = LA.norm(labels, ord=1)[np.newaxis]
#     p_normalized = labels / n.T
#     result = entropy(p_normalized, base=2)
#     return result
#
#
# if __name__ == '__main__':
#     print(v_entropy([0, 0, 12, 5, 4, 3] ))