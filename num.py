import numpy as np
import random



if __name__ == '__main__':
    lines = list(np.arange(10))
    random.shuffle(lines)
    lines = np.array(lines, dtype='int')

    att = np.array(["a", "a", "b", "c", "a", "b", "c", "a", "a", "b"])
    labels = np.array(["X", "Y", "Y", "Z", "Z", "Z", "X", "Z", "Z", "X"])

    print(f'lines = {lines}')
    print(f'att = {att}')
    print(f'labels = {labels}')

    covered, not_covered = partition_covered(lines, att, labels, "a", "Z")

    print(f'covered = {covered}')
    print(f'item_covered = {att[covered]}')
    print(f'not_covered = {not_covered}')
    print(f'item_not_covered = {att[not_covered]}')

    if True:
        exit()
    item_lines = np.where(att == "a")[0]
    print(f'item_lines = {item_lines}')
    item_labels = labels[item_lines]
    print(f'item_labels = {item_labels}')
    c = item_labels == "Z"
    print(f'c ={c}')
    lns = np.where(item_labels == "Z")[0]
    print(f'lns_covered  = {item_lines[lns]}')
    print(f'lns_covered2 = {item_lines[c]}')
    print(f'lns_not_covered = {item_lines[~c]}')

    covered = item_lines[lns]
    print(item_labels)
    print(lns)
    print(covered)
