import operator

import torch
import gc


def get_free_vram():
    m = torch.cuda.mem_get_info()
    return m, 1 - (m[0] / m[1])


def print_alive_tensors(n=-1, print_list=False, print_total=True):
    t = []

    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                t.append((type(obj), torch.prod(torch.tensor(obj.size())).item() * 8 / 10e6))
        except:
            continue

    t = sorted(t, key=operator.itemgetter(1), reverse=True)
    if n == -1:
        n = len(t)

    if print_list:
        for i in range(n):
            print(t[i])

    print("Number of tensors: {0}".format(len(t)))
    if print_total:
        print("Total: {0} megabytes".format(sum([x[1] for x in t])))
