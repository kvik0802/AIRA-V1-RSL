from collections import OrderedDict


class LRU:
    def __init__(s, cap):
        s.c = OrderedDict()
        s.cap = cap

    def get(s, k):
        if k not in s.c:
            return None
        s.c.move_to_end(k)
        return s.c[k]

    def put(s, k, v):
        s.c[k] = v
        s.c.move_to_end(k)
        if len(s.c) > s.cap:
            s.c.popitem(last=False)